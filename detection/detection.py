import os
import cv2
import numpy as np
import pika
import pickle
import uuid
from shapely.geometry import Polygon, box
from ultralytics import YOLO
from datetime import datetime, timezone
from collections import deque

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIOLATIONS_FOLDER = os.path.join(PROJECT_ROOT, "fast_api", "violations")
os.makedirs(VIOLATIONS_FOLDER, exist_ok=True)

model = YOLO("../yolo12m-v2.pt")
CONTAINER_ROI = np.array(
    [[530, 270], [449, 694], [366, 680], [463, 264]], dtype=np.int32
)
ROI_POLY = Polygon(CONTAINER_ROI)


# --- Helpers ---
def in_roi(bbox):
    return ROI_POLY.intersects(box(*bbox))


def iou(b1, b2):
    xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
    xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    area2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def overlaps_any(b, boxes, thresh=0.2):
    return any(iou(b, bb) >= thresh for bb in boxes)


def smooth_history(current, history):
    """Refactored smoothing logic [cite: 161]"""
    if not current and history:
        return history[-1]
    history.append(current)
    return current


def annotate_frame(frame, hands, scoops, pizzas, v_count, active, used):
    cv2.polylines(frame, [CONTAINER_ROI], True, (0, 255, 255), 2)

    # [cite_start]Generic drawer to reduce repetition [cite: 143-147]
    def draw_box(boxes, color, label):
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    draw_box(hands, (0, 0, 255), "hand")
    draw_box(scoops, (255, 0, 0), "scooper")
    draw_box(pizzas, (0, 255, 0), "pizza")

    cv2.putText(
        frame,
        f"Violations: {v_count}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    if active:
        status = "Session: ACTIVE (Scooper OK)" if used else "Session: ACTIVE"
        color = (0, 255, 0) if used else (0, 165, 255)
        cv2.putText(frame, status, (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


# --- Main Service ---
def detection_service(in_queue, db_queue, api_queue, stream_name="default"):
    creds = pika.PlainCredentials("admin", "strongpassword")
    conn = pika.BlockingConnection(
        pika.ConnectionParameters("localhost", credentials=creds)
    )
    ch = conn.channel()

    for q in [in_queue, db_queue, api_queue]:
        ch.queue_declare(queue=q, durable=True, auto_delete=True)
    ch.queue_declare(queue="finish_signals", durable=False, auto_delete=True)
    ch.queue_declare(
        queue=f"annotated_frames_{stream_name}", durable=True, auto_delete=True
    )

    # State
    violation_count = 0
    session_active = False
    scooper_used = False
    hand_touched_pizza = False

    exit_counter = 0
    post_window_counter = 0

    # History Buffers
    last_hands = deque(maxlen=12)
    last_scoops = deque(maxlen=12)
    last_pizzas = deque(maxlen=12)
    roi_frames = deque(maxlen=3)

    def commit_violation(fid, v_frame):
        nonlocal violation_count
        violation_count += 1
        fname = f"{uuid.uuid4()}.jpg"
        cv2.imwrite(
            os.path.join(VIOLATIONS_FOLDER, fname),
            v_frame,
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )

        payload = {
            "frame_id": fid,
            "message": "violation",
            "file_path": fname,
            "violation_count": violation_count,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        for q in [db_queue, api_queue]:
            ch.basic_publish(exchange="", routing_key=q, body=pickle.dumps(payload))
        print(f"[Detection] VIOLATION #{violation_count} saved.")

    def callback(chx, method, props, body):
        nonlocal violation_count, session_active, scooper_used, hand_touched_pizza
        nonlocal exit_counter, post_window_counter
        nonlocal last_hands, last_scoops, last_pizzas, roi_frames

        data = pickle.loads(body)

        # 1. HANDLE STREAM END OR FORCE STOP
        if "end" in data:
            print(f"[Detection] Shutting down. Final count: {violation_count}")

            # Signal the finish_listener in api.py to clean up queues
            chx.basic_publish(
                exchange="",
                routing_key="finish_signals",
                body=pickle.dumps(
                    {"type": "detection_finished", "violation_count": violation_count}
                ),
            )

            # Acknowledge and stop the consumer to kill the thread
            chx.basic_ack(delivery_tag=method.delivery_tag)
            chx.stop_consuming()
            return

        frame_id = data.get("frame_id")
        frame_jpeg_bytes = data.get("frame_jpeg")

        if frame_jpeg_bytes is None:
            chx.basic_ack(delivery_tag=method.delivery_tag)
            return

        frame = cv2.imdecode(
            np.frombuffer(frame_jpeg_bytes, np.uint8), cv2.IMREAD_COLOR
        )
        if frame is None:
            chx.basic_ack(delivery_tag=method.delivery_tag)
            return

        # 2. PREDICTION & SMOOTHING
        results = model.predict(frame, conf=0.15, iou=0.5, verbose=False, device="cuda")
        current_hands, current_scoops, current_pizzas = [], [], []

        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy.cpu().numpy()[0]
                label = r.names[int(b.cls.cpu().numpy()[0])]
                if label == "hand":
                    current_hands.append((x1, y1, x2, y2))
                elif label == "scooper":
                    current_scoops.append((x1, y1, x2, y2))
                elif label == "pizza":
                    current_pizzas.append((x1, y1, x2, y2))

        # Apply History Filling
        hands = (
            current_hands
            if current_hands
            else (list(last_hands)[-1] if last_hands else [])
        )
        last_hands.append(current_hands)
        scoops = (
            current_scoops
            if current_scoops
            else (list(last_scoops)[-1] if last_scoops else [])
        )
        last_scoops.append(current_scoops)
        pizzas = (
            current_pizzas
            if current_pizzas
            else (list(last_pizzas)[-1] if last_pizzas else [])
        )
        last_pizzas.append(current_pizzas)

        # 3. OVERLAP CHECKS
        hand_inside_roi = any(in_roi(hb) for hb in hands)
        scooper_to_pizza_now = (
            any(overlaps_any(sb, pizzas) for sb in scoops) and len(pizzas) > 0
        )
        hand_on_pizza_now = any(overlaps_any(hb, pizzas) for hb in hands)

        if hand_inside_roi:
            roi_frames.append(frame.copy())

        # 4. STATE MACHINE & VIOLATION LOGIC
        if hand_inside_roi and not session_active:
            session_active = True
            scooper_used, hand_touched_pizza = False, False
            exit_counter, post_window_counter = 0, 0
            roi_frames.clear()
            roi_frames.append(frame.copy())

        if session_active:
            if scooper_to_pizza_now:
                scooper_used = True
            if hand_on_pizza_now:
                hand_touched_pizza = True

            if not hand_inside_roi:
                exit_counter += 1
                post_window_counter += 1

                if scooper_to_pizza_now and post_window_counter <= 10:  # Grace period
                    scooper_used = True

                if exit_counter >= 5:  # Stable exit
                    if hand_touched_pizza and not scooper_used and len(roi_frames) > 0:
                        # COMMIT VIOLATION IMMEDIATELY
                        commit_violation(frame_id, roi_frames[-1])

                    session_active = False
                    roi_frames.clear()
            else:
                exit_counter, post_window_counter = 0, 0

        # 5. PUBLISH LIVE STREAM FRAME
        annotated = annotate_frame(
            frame.copy(),
            hands,
            scoops,
            pizzas,
            violation_count,
            session_active,
            scooper_used,
        )
        ok, jpeg_annot = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            ch.basic_publish(
                exchange="",
                routing_key=f"annotated_frames_{stream_name}",
                body=pickle.dumps(
                    {"frame_id": frame_id, "frame_jpeg": jpeg_annot.tobytes()}
                ),
            )

        chx.basic_ack(delivery_tag=method.delivery_tag)

    ch.basic_qos(prefetch_count=10)
    ch.basic_consume(queue=in_queue, on_message_callback=callback)
    print(f"[Detection] Listening on {in_queue}")
    ch.start_consuming()
