# --- Install dependencies in Colab ---
# !pip install ultralytics shapely opencv-python

import cv2
import numpy as np
from shapely.geometry import Polygon, box
from ultralytics import YOLO

# ---- Load YOLO model ----
model = YOLO("yolo12m-v2.pt")  # place model in Colab working dir

# ---- ROI setup ----
CONTAINER_ROI = np.array(
    [[530, 270], [449, 694], [366, 680], [463, 264]], dtype=np.int32
)
ROI_POLY = Polygon(CONTAINER_ROI)


def in_roi(bbox):
    """Check if bounding box intersects ROI polygon."""
    return ROI_POLY.intersects(box(*bbox))


def process_video(video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Could not open video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # ---- State ----
    frame_id = 0
    violation_count = 0
    detections = []
    hand_inside = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model.predict(frame, conf=0.2, iou=0.5, verbose=False)

        hands, scoops = [], []
        for result in results:
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

        # State checks
        current_hands_in_roi = sum(in_roi(h) for h in hands)
        scooper_inside = any(in_roi(s) for s in scoops)

        # ---- Cycle-based violation logic ----
        if current_hands_in_roi > 0 and not hand_inside:
            # Hand just entered ROI
            hand_inside = True

            if not scooper_inside:  # scooper didn't move
                violation_count += 1
                msg = f"VIOLATION #{violation_count}: Hand entered ROI without scooper"
                detections.append({"frame_id": frame_id, "event": msg})
                cv2.putText(
                    frame, msg, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )
            else:
                msg = "Hand + scooper action (OK)"
                detections.append({"frame_id": frame_id, "event": msg})
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

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()
    return {"total_violations": violation_count, "detections": detections}


# ---- Example run ----
video_path = "Sahwb3dhaghalt(2).mp4"  # put video in Colab working dir
out_path = "output_cycle_violation1.mp4"
results = process_video(video_path, out_path)

print(f"Video saved: {out_path}")
print(f"Final violations detected: {results['total_violations']}")
print("Detection log:", results["detections"])
