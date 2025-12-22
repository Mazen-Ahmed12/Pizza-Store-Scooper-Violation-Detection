# =============================================
# RabbitMQ Detection Service (detection.py)
# Run this in another terminal: python detection.py
# (After starting frame_reader in separate terminal)
# =============================================
import cv2
import numpy as np
from shapely.geometry import Polygon, box
from ultralytics import YOLO
import pika
import pickle

# ---- Load YOLO model ----
model = YOLO("yolo12m-v2.pt")

# ---- ROI setup ----
CONTAINER_ROI = np.array(
    [[530, 270], [449, 694], [366, 680], [463, 264]], dtype=np.int32
)
ROI_POLY = Polygon(CONTAINER_ROI)


def in_roi(bbox):
    """Check if bounding box intersects ROI polygon."""
    return ROI_POLY.intersects(box(*bbox))


def detection_service(queue_name="frames", out_path="output_cycle_violation.mp4"):

    # --- CRITICAL FIX: Add your credentials here ---
    credentials = pika.PlainCredentials("admin", "strongpassword")
    parameters = pika.ConnectionParameters(host="localhost", credentials=credentials)

    # Connect to RabbitMQ
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    channel.queue_declare(queue=queue_name, durable=True)

    # Prepare video writer (we'll init after first frame)
    out = None
    fps = 25  # Default, will update if possible
    w, h = None, None

    # ---- State ----
    frame_id = 0
    violation_count = 0
    detections = []
    hand_inside = False

    def callback(ch, method, properties, body):
        nonlocal out, fps, w, h, frame_id, violation_count, detections, hand_inside

        data = pickle.loads(body)

        if "end" in data:
            # End of video — finalize
            if out:
                out.release()
            print(f"Video saved: {out_path}")
            print(f"Final violations detected: {violation_count}")
            print("Detection log:", detections)
            ch.stop_consuming()
            return

        # Get frame
        received_frame_id = data["frame_id"]
        frame = data["frame"]

        # Init writer on first frame
        if out is None:
            h, w = frame.shape[:2]
            fps = 25  # Use default since no cap here; adjust if known
            out = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )

        # Run YOLO detection (your original logic)
        results = model.predict(frame, conf=0.2, iou=0.5, verbose=False, device="cuda")

        hands, scoops = [], []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                cls_id = int(box.cls.cpu().numpy()[0])
                label = result.names[cls_id]

                if label == "hand":
                    hands.append((x1, y1, x2, y2))
                    cv2.rectangle(
                        frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2
                    )
                    cv2.putText(
                        frame,
                        "hand",
                        (int(x1), int(y1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 200, 255),
                        2,
                    )
                elif label == "scooper":
                    scoops.append((x1, y1, x2, y2))
                    cv2.rectangle(
                        frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 150, 0), 2
                    )
                    cv2.putText(
                        frame,
                        "scooper",
                        (int(x1), int(y1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 150, 0),
                        2,
                    )

        # Draw ROI
        cv2.polylines(frame, [CONTAINER_ROI], True, (0, 255, 255), 2)

        # State checks (your original logic)
        current_hands_in_roi = sum(in_roi(h) for h in hands)
        scooper_inside = any(in_roi(s) for s in scoops)

        if current_hands_in_roi > 0 and not hand_inside:
            # Hand just entered ROI
            hand_inside = True
            if not scooper_inside:  # scooper didn't move
                violation_count += 1
                msg = f"VIOLATION #{violation_count}: Hand entered ROI without scooper"
                detections.append({"frame_id": received_frame_id, "event": msg})
                cv2.putText(
                    frame, msg, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )
            else:
                msg = "Hand + scooper action (OK)"
                detections.append({"frame_id": received_frame_id, "event": msg})
                cv2.putText(
                    frame, msg, (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )

        if current_hands_in_roi == 0 and hand_inside:
            # Hand exited ROI → reset cycle
            hand_inside = False

        # Overlay totals
        cv2.putText(
            frame,
            f"Total Violations: {violation_count}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        # Write to output
        out.write(frame)
        print(f"Processed frame {received_frame_id}")

        ch.basic_ack(delivery_tag=method.delivery_tag)  # Acknowledge

    channel.basic_qos(prefetch_count=1)  # Fair dispatch
    channel.basic_consume(queue=queue_name, on_message_callback=callback)

    print("Detection service started. Waiting for frames...")
    channel.start_consuming()

    connection.close()
    return {"total_violations": violation_count, "detections": detections}


# Run example
out_path = "output_cycle_violation.mp4"
results = detection_service(out_path=out_path)
print(f"Video saved: {out_path}")
print(f"Final violations detected: {results['total_violations']}")
print("Detection log:", results["detections"])
