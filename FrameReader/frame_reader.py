import cv2
import pika
import pickle
import time


def frame_reader(source, queue_name="frames", stop_event=None):
    # Works for both file paths and RTSP URLs [cite: 137, 199]
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Low latency buffering

    if not cap.isOpened():
        print(f"[FrameReader] Error opening source: {source}")
        return

    creds = pika.PlainCredentials("admin", "strongpassword")
    conn = pika.BlockingConnection(
        pika.ConnectionParameters("localhost", credentials=creds)
    )
    ch = conn.channel()
    ch.queue_declare(queue=queue_name, durable=True, auto_delete=True)

    frame_id = 0
    tiny_delay = 0.002  # Prevent CPU spin [cite: 138]

    print(f"[FrameReader] Streaming from: {source}")

    while cap.isOpened():
        # Check if stop signal was set by API [cite: 191]
        if stop_event and stop_event.is_set():
            break

        ret, frame = cap.read()
        if not ret:
            break

        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if not ok:
            time.sleep(tiny_delay)
            continue

        data = {"frame_id": frame_id, "frame_jpeg": jpeg.tobytes()}
        ch.basic_publish(exchange="", routing_key=queue_name, body=pickle.dumps(data))
        frame_id += 1
        time.sleep(tiny_delay)

    # [cite_start]Send End-of-Stream signal [cite: 140]
    if not stop_event or not stop_event.is_set():
        ch.basic_publish(
            exchange="", routing_key=queue_name, body=pickle.dumps({"end": True})
        )

    print(f"[FrameReader] Finished. Frames sent: {frame_id}")
    cap.release()
    conn.close()
