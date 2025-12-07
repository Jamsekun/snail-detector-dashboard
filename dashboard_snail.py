from flask import Flask, Response, render_template
import cv2
import threading
import time
import torch
from ultralytics import YOLO # type: ignore

app = Flask(__name__)

# ------------------------------
# Global Shared Data
# ------------------------------
latest_frame = None          # Raw frame from camera
latest_inferred = None       # YOLO annotated frame
latest_count = 0
current_snail_count = 0
lock = threading.Lock()

running = True               # Thread control flag

# ------------------------------
# Load YOLO Model
# ------------------------------
model = YOLO("models/last.pt")

# ------------------------------
# Camera Capture Thread
# ------------------------------
def camera_thread():
    global latest_frame, running

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Improve FPS
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30) # you can change fps here

    if not cap.isOpened():
        raise IOError("❌ Cannot access webcam")

    print("Camera thread started")

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        with lock:
            latest_frame = frame

        time.sleep(0.005)   # Prevent 100% CPU spike

    cap.release()


# ------------------------------
# YOLO Inference Thread
# ------------------------------
def inference_thread():
    global latest_frame, latest_inferred, latest_count, running, current_snail_count

    print("YOLO inference thread started")

    while running:
        with lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(0.01)
            continue

        results = model.predict(frame, imgsz=640, conf=0.4, verbose=False)
        r = results[0]

        boxes = r.boxes
        preds = []

        if boxes is not None and boxes.xyxy is not None:
            xyxy = boxes.xyxy
            preds = xyxy.cpu().numpy() if isinstance(xyxy, torch.Tensor) else xyxy

        latest_count = len(preds)
        current_snail_count = latest_count


        # Draw results
        for i, box in enumerate(preds):
            x1, y1, x2, y2 = map(int, box)
            conf = float(boxes.conf[i])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.putText(frame, f"Count: {latest_count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        with lock:
            latest_inferred = frame

        time.sleep(0.001)


# ------------------------------
# MJPEG Stream Generator
# ------------------------------
def gen_frames():
    global latest_inferred

    while True:
        with lock:
            if latest_inferred is None:
                continue
            frame = latest_inferred.copy()

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            buffer.tobytes() +
            b'\r\n'
        )


# ------------------------------
# Flask Routes
# ------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
    
@app.route("/snail_count")
def snail_count():
    return {"count": current_snail_count}



# ------------------------------
# Start Threads + Flask
# ------------------------------
if __name__ == "__main__":
    print("Starting multithreaded Snail Detector Dashboard...")

    # Start camera capture thread
    t1 = threading.Thread(target=camera_thread, daemon=True)
    t1.start()

    # Start YOLO inference thread
    t2 = threading.Thread(target=inference_thread, daemon=True)
    t2.start()

    # Start Flask server
    app.run(host="0.0.0.0", port=5000, debug=False)

    running = False
    t1.join()
    t2.join()
