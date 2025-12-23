import pika
import pickle
import psycopg

# Psycopg 3 Connection
conn_string = "host=localhost port=5432 user=admin password=strongpassword dbname=db"
conn = psycopg.connect(conn_string)
cur = conn.cursor()

# RabbitMQ Setup
credentials = pika.PlainCredentials("admin", "strongpassword")
parameters = pika.ConnectionParameters(host="localhost", credentials=credentials)

connection = pika.BlockingConnection(parameters)
channel = connection.channel()
channel.queue_declare(queue="detection_logs", durable=True)


def callback(ch, method, props, body):
    data = pickle.loads(body)

    # AUTO-CLOSE when finish
    if "end" in data:
        print("End of stream received. Closing database logger...")
        ch.stop_consuming()
        return

    try:
        cur.execute(
            """INSERT INTO detection_logs (frame_id, message, frame_jpeg, violation_count, created_at)
               VALUES (%s, %s, %s, %s, %s)""",
            (
                data["frame_id"],
                data["message"],
                data["frame_jpeg"],
                data["violation_count"],
                data["created_at"],
            ),
        )
        conn.commit()
        print(
            f"Inserted frame {data['frame_id']} | Total Violations: {data['violation_count']}"
        )
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        conn.rollback()
        print(f"Database error: {e}")


channel.basic_consume(queue="detection_logs", on_message_callback=callback)
print("Log service active. Monitoring detections...")
channel.start_consuming()

# Clean up when everything  finishes
cur.close()
conn.close()
