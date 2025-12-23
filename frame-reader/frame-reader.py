import cv2
import pika
import pickle
import sys  # Import sys for better error handling


def frame_reader(video_path, queue_name="frames"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Could not open video file")

    # Use the username and password you set in your Docker command
    credentials = pika.PlainCredentials("admin", "strongpassword")
    parameters = pika.ConnectionParameters(host="localhost", credentials=credentials)

    # Connect to RabbitMQ
    try:
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        channel.queue_declare(queue=queue_name, durable=True)
    except pika.exceptions.ProbableAuthenticationError:
        print(
            "ERROR: Authentication failed. Check 'myuser' and 'mypassword' in the script."
        )
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during connection: {e}")
        sys.exit(1)

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        data = pickle.dumps({"frame_id": frame_id, "frame": frame})

        # Publish to queue
        channel.basic_publish(exchange="", routing_key=queue_name, body=data)
        print(f"Sent frame {frame_id}")

        frame_id += 1

    # Send end signal
    channel.basic_publish(
        exchange="", routing_key=queue_name, body=pickle.dumps({"end": True})
    )
    print("Video frames sent. End signal published.")

    cap.release()
    connection.close()


video_path = "Sahwb3dhaghalt(2).mp4"
frame_reader(video_path)
