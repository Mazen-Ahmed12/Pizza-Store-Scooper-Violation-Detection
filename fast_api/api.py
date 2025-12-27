import os, uuid, threading, pickle, queue, time, asyncio, sys
from pathlib import Path
import cv2, requests, pika
from fastapi import FastAPI, UploadFile, File, WebSocket, Form, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the unified frame reader
from FrameReader.frame_reader import frame_reader
from detection.detection import detection_service
from postgresql.db import db_logger

app = FastAPI()
UPLOAD_DIR = PROJECT_ROOT / "fast_api" / "uploads"
VIOLATION_DIR = PROJECT_ROOT / "fast_api" / "violations"
for d in [UPLOAD_DIR, VIOLATION_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app.mount("/violations", StaticFiles(directory=str(VIOLATION_DIR)), name="violations")

JOBS_QUEUE = "jobs"
clients, violation_queue = set(), queue.Queue()
current_job = None
lock = threading.Lock()
rtsp_stop_event = threading.Event()


def rmq():
    return pika.BlockingConnection(
        pika.ConnectionParameters(
            "localhost", credentials=pika.PlainCredentials("admin", "strongpassword")
        )
    )


# --- Helpers ---
async def broadcast(msg):
    dead = []
    for ws in list(clients):
        try:
            await ws.send_json(msg)
        except:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


# --- Routes ---
@app.post("/save/{filename}")
async def save(filename: str, file: UploadFile = File(...)):
    (UPLOAD_DIR / filename).write_bytes(await file.read())
    return {"ok": True}


@app.post("/start")
async def start(path: str = Form(None), rtsp: str = Form(None)):
    global current_job
    with lock:
        if current_job:
            return Response(status_code=204)

    rtsp_stop_event.clear()
    job_id = str(uuid.uuid4())
    stream_name = "mystream" if rtsp and "mystream" in rtsp else f"camera_{job_id}"
    source = str(UPLOAD_DIR / path) if path else rtsp

    job = {
        "job_id": job_id,
        "stream_name": stream_name,
        "source": source,
        "type": "video" if path else "rtsp",
        "frames_q": f"frames_{job_id}",
        "db_q": f"db_{job_id}",
        "api_q": f"api_{job_id}",
    }

    if job["type"] == "rtsp" and stream_name != "mystream":
        requests.put(
            "http://localhost:1984/api/streams",
            json={"name": stream_name, "src": source},
        )

    with lock:
        current_job = job

    conn = rmq()
    conn.channel().queue_declare(queue=JOBS_QUEUE, durable=True)
    conn.channel().basic_publish(
        exchange="", routing_key=JOBS_QUEUE, body=pickle.dumps(job)
    )
    conn.close()

    await broadcast({"type": "reset"})
    live_url = (
        f"/api/annotated.mjpeg?src={stream_name}" if job["type"] == "rtsp" else None
    )
    await broadcast(
        {"type": "processing_started", "source_type": job["type"], "live_url": live_url}
    )
    return {"ok": True}


@app.post("/stop")
async def stop():
    global current_job
    with lock:
        if not current_job:
            raise HTTPException(status_code=400, detail="No active job")
        job = current_job
        current_job = None

    rtsp_stop_event.set()

    # NEW: Explicitly send a termination signal to the detection queue
    try:
        conn = rmq()
        ch = conn.channel()
        # Send 'end' to the frames queue so detection_service stops consuming
        ch.basic_publish(
            exchange="",
            routing_key=job["frames_queue"],
            body=pickle.dumps({"end": True, "force_stop": True}),
        )
        conn.close()
    except Exception as e:
        print(f"[Stop] Failed to send kill signal: {e}")

    await broadcast({"type": "processing_finished"})
    # ... rest of your go2rtc cleanup ...


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except:
        clients.discard(ws)


@app.get("/api/annotated.mjpeg")
async def annotated_stream(src: str):
    def gen():
        conn = rmq()
        qname = f"annotated_frames_{src}"
        conn.channel().queue_declare(queue=qname, durable=True, auto_delete=True)
        for _, _, body in conn.channel().consume(qname, auto_ack=True):
            frame = pickle.loads(body).get("frame_jpeg")
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

    return StreamingResponse(
        gen(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


# --- Background Workers ---
def worker():
    while True:
        try:
            conn = rmq()
            ch = conn.channel()
            ch.queue_declare(queue=JOBS_QUEUE, durable=True)

            def callback(chx, method, _, body):
                job = pickle.loads(body)
                print(f"[Worker] Starting job: {job['source']}")

                # Start Services
                threading.Thread(
                    target=detection_service,
                    args=(
                        job["frames_q"],
                        job["db_q"],
                        job["api_q"],
                        job["stream_name"],
                    ),
                    daemon=True,
                ).start()
                threading.Thread(
                    target=db_logger, args=(job["db_q"],), daemon=True
                ).start()
                threading.Thread(
                    target=broadcaster, args=(job["api_q"],), daemon=True
                ).start()

                # [cite_start]Start Frame Reader (Replaces manual loop!) [cite: 213]
                # Pass rtsp_stop_event so API can stop it
                threading.Thread(
                    target=frame_reader,
                    args=(job["source"], job["frames_q"], rtsp_stop_event),
                    daemon=True,
                ).start()

                chx.basic_ack(method.delivery_tag)

            ch.basic_consume(queue=JOBS_QUEUE, on_message_callback=callback)
            ch.start_consuming()
        except:
            time.sleep(3)


def broadcaster(queue_name):
    try:
        conn = rmq()
        ch = conn.channel()
        ch.queue_declare(queue=queue_name, durable=True, auto_delete=True)

        def cb(ch, method, _, body):
            d = pickle.loads(body)
            violation_queue.put(
                {
                    "type": "violation",
                    "count": d["violation_count"],
                    "time": d["created_at"],
                    "file_path": d["file_path"],
                }
            )
            ch.basic_ack(method.delivery_tag)

        ch.basic_consume(queue=queue_name, on_message_callback=cb)
        ch.start_consuming()
    except Exception as e:
        print(f"Broadcaster error: {e}")


def finish_listener():
    # Cleans up queues and files when detection finishes
    global current_job
    while True:
        try:
            conn = rmq()
            ch = conn.channel()
            ch.queue_declare(queue="finish_signals", auto_delete=True)

            def cb(ch, method, _, body):
                global current_job
                if pickle.loads(body).get("type") == "detection_finished":
                    violation_queue.put({"type": "processing_finished"})
                    with lock:
                        job = current_job
                        current_job = None
                    if job:
                        for q in [job["frames_q"], job["db_q"], job["api_q"]]:
                            try:
                                ch.queue_delete(queue=q)
                            except:
                                pass
                        if job["type"] == "video" and os.path.exists(job["source"]):
                            os.remove(job["source"])
                ch.basic_ack(method.delivery_tag)

            ch.basic_consume(queue="finish_signals", on_message_callback=cb)
            ch.start_consuming()
        except:
            time.sleep(3)


async def violation_distributor():
    while True:
        await broadcast(await asyncio.to_thread(violation_queue.get))


@app.on_event("startup")
async def startup():
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=finish_listener, daemon=True).start()
    asyncio.create_task(violation_distributor())
