import cv2
import numpy as np
from shapely.geometry import Polygon, box
from ultralytics import YOLO
import pika
import pickle
from datetime import datetime, timezone


model = YOLO("yolo12m-v2.pt")

CONTAINER_ROI = np.array(
    [[530, 270], [449, 694], [366, 680], [463, 264]], dtype=np.int32
)
ROI_POLY = Polygon(CONTAINER_ROI)


def in_roi(bbox):
    return ROI_POLY.intersects(box(*bbox))


def encode_frame(frame):
    # JPEGs are already compressed; we send raw bytes to avoid re-compression in DB
    ok, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes() if ok else None


def detection_service(in_queue="frames", out_queue="detection_logs"):
    credentials = pika.PlainCredentials("admin", "strongpassword")
    parameters = pika.ConnectionParameters(host="localhost", credentials=credentials)

    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    channel.queue_declare(queue=in_queue, durable=True)
    channel.queue_declare(queue=out_queue, durable=True)

    violation_count = 0
    hand_inside = False

    def callback(ch, method, properties, body):
        nonlocal violation_count, hand_inside
        data = pickle.loads(body)

        if "end" in data:
            print(f"Final violations detected: {violation_count}")
            ch.stop_consuming()
            return

        received_frame_id = data["frame_id"]
        frame = data["frame"]

        results = model.predict(frame, conf=0.2, iou=0.5, verbose=False, device="cuda")

        hands, scoops = [], []
        for result in results:
            if result.boxes is None:
                continue
            for box_obj in result.boxes:
                x1, y1, x2, y2 = box_obj.xyxy.cpu().numpy()[0]
                cls_id = int(box_obj.cls.cpu().numpy()[0])
                label = result.names[cls_id]
                if label == "hand":
                    hands.append((x1, y1, x2, y2))
                elif label == "scooper":
                    scoops.append((x1, y1, x2, y2))

        current_hands_in_roi = sum(in_roi(h) for h in hands)
        scooper_inside = any(in_roi(s) for s in scoops)

        if current_hands_in_roi > 0 and not hand_inside:
            hand_inside = True
            if not scooper_inside:
                violation_count += 1
                payload = {
                    "frame_id": received_frame_id,
                    "message": "violation",
                    "frame_jpeg": encode_frame(frame),
                    "violation_count": violation_count,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                channel.basic_publish(
                    exchange="", routing_key=out_queue, body=pickle.dumps(payload)
                )
                print(f"Published violation: Frame {received_frame_id}")

        if current_hands_in_roi == 0 and hand_inside:
            hand_inside = False

        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=in_queue, on_message_callback=callback)
    print("Detection service started. Waiting for frames...")
    channel.start_consuming()
    connection.close()


if __name__ == "__main__":
    detection_service()
