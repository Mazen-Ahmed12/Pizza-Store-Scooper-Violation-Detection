import pika
import pickle
import psycopg
from datetime import datetime


def db_logger(queue_name):
    conn = psycopg.connect(
        "host=localhost port=5432 user=admin password=strongpassword dbname=db"
    )
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS detection_logs (
            id SERIAL PRIMARY KEY, job_id TEXT, frame_id INTEGER, message TEXT, 
            violation_count INTEGER, file_path TEXT, created_at TIMESTAMPTZ DEFAULT NOW()
        )"""
    )
    conn.commit()

    mq = pika.BlockingConnection(
        pika.ConnectionParameters(
            "localhost", credentials=pika.PlainCredentials("admin", "strongpassword")
        )
    )
    ch = mq.channel()
    ch.queue_declare(queue=queue_name, durable=True, auto_delete=True)

    def callback(chx, method, _, body):
        data = pickle.loads(body)
        if "end" in data:
            return chx.basic_ack(method.delivery_tag)

        try:
            cur.execute(
                """
                INSERT INTO detection_logs (job_id, frame_id, message, violation_count, file_path, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)""",
                (
                    data.get("job_id"),
                    data.get("frame_id"),
                    data.get("message"),
                    data.get("violation_count"),
                    data.get("file_path"),
                    datetime.utcnow(),
                ),
            )
            conn.commit()
            print(f"[DB] Saved violation #{data.get('violation_count')}")
        except Exception as e:
            conn.rollback()
            print(f"[DB] Error: {e}")
        chx.basic_ack(method.delivery_tag)

    print(f"[DB] Listening on {queue_name}")
    ch.basic_consume(queue=queue_name, on_message_callback=callback)
    ch.start_consuming()
