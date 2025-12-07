# dashboard_snail_fast.py
from flask import Flask, Response, render_template
import cv2
import threading
import time
import numpy as np
import torch
from ultralytics import YOLO  # type: ignore
import queue
import os
import sys

app = Flask(__name__)

# ------------------------------
# CONFIG - tune these for speed
# ------------------------------
MODEL_PATH = "models/last.pt"          # your model - try a small model (yolov8n / yolov8n-seg)
TARGET_WIDTH = 1280                     # inference input width (smaller -> faster) 480 or 1280
TARGET_HEIGHT = 720                    # inference input height (smaller -> faster) 640 or 720
TARGET_FPS = 60.0                      # target visual framerate
ANNOTATE_EVERY_N = 1                   # draw boxes on every Nth inference (increase to reduce drawing cost)
JPEG_QUALITY = 70                      # encode quality (0-100) smaller = faster & less bandwidth
FRAME_QUEUE_MAXSIZE = 2                # keep only most recent frames
WARMUP_ROUNDS = 2                      # run small warmups on model

# ------------------------------
# GLOBAL STATE
# ------------------------------
frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)   # camera -> inference
latest_inferred = None
latest_count = 0
current_snail_count = 0
inference_lock = threading.Lock()
running = True

# ------------------------------
# DEVICE / ACCELERATOR DETECTION
# ------------------------------
def pick_device():
    # Prefer torch.cuda if available. On Pi, this is often NOT available.
    if torch.cuda.is_available():
        return "cuda:0"
    # allow forcing a device via env var (e.g. "cpu", "cuda:0", "hpu", or custom)
    env = os.environ.get("INFERENCE_DEVICE", "").strip()
    if env:
        return env
    return "cpu"

DEVICE = pick_device()
print(f"[INFO] Selected inference device: {DEVICE}")

# ------------------------------
# Load model (Ultralytics YOLO wrapper)
# ------------------------------
print("[INFO] Loading model:", MODEL_PATH)
model = YOLO(MODEL_PATH)

# Try to move model to device (if supported)
try:
    if hasattr(model, "to"):
        model.to(DEVICE)
except Exception:
    # not fatal - model.predict will accept device arg below
    pass

# Optionally enable fuse/half if supported on device
USE_HALF = False
if "cuda" in DEVICE:
    USE_HALF = True

# Warmup
def warmup_model():
    print("[INFO] Warming up model...")
    dummy = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
    for _ in range(WARMUP_ROUNDS):
        try:
            _ = model.predict(dummy, imgsz=(TARGET_HEIGHT, TARGET_WIDTH),
                              device=DEVICE, conf=0.25, verbose=False)
        except TypeError:
            # some ultralytics versions expect imgsz=int not tuple
            _ = model.predict(dummy, imgsz=TARGET_WIDTH, device=DEVICE, conf=0.25, verbose=False)
    print("[INFO] Warmup done")
warmup_model()

# ------------------------------
# CAMERA THREAD (producer)
# ------------------------------
def camera_thread():
    global running
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    # suggest higher capture fps, hardware permitting
    cap.set(cv2.CAP_PROP_FPS, min(60, int(TARGET_FPS)))

    if not cap.isOpened():
        raise IOError("❌ Cannot access webcam")

    print("[INFO] Camera thread started")
    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.005)
            continue

        # keep only most-recent frames (drop oldest)
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                # drop one, then put
                _ = frame_queue.get_nowait()
            except Exception:
                pass
            try:
                frame_queue.put_nowait(frame)
            except Exception:
                pass

        # tiny sleep to let other threads run
        time.sleep(0.001)
    cap.release()
    print("[INFO] Camera thread stopped")

# ------------------------------
# INFERENCE THREAD (consumer -> produces annotated frames)
# ------------------------------
def inference_thread():
    global latest_inferred, latest_count, current_snail_count, running

    print("[INFO] Inference thread started (device: %s, half=%s)" % (DEVICE, USE_HALF))
    annotate_counter = 0

    target_interval = 1.0 / TARGET_FPS
    last_inference_ts = 0.0

    while running:
        # fetch the most recent frame from queue (drain to latest)
        frame = None
        try:
            # block briefly to await a frame
            frame = frame_queue.get(timeout=0.2)
            # drain to the newest frame if there are more
            while True:
                try:
                    frame = frame_queue.get_nowait()
                except queue.Empty:
                    break
        except queue.Empty:
            # no frame available
            time.sleep(0.005)
            continue

        # resize to inference size to reduce workload
        input_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        # convert BGR -> RGB if needed by model wrapper
        # ultralytics accepts BGR np array as well, so we pass as-is.

        # run inference and measure time
        t0 = time.time()
        try:
            # prefer passing device and imgsz explicitly; pass half if supported
            predict_kwargs = dict(imgsz=(TARGET_HEIGHT, TARGET_WIDTH), device=DEVICE, conf=0.35, verbose=False)
            if USE_HALF:
                predict_kwargs["half"] = True
            results = model.predict(input_frame, **predict_kwargs)
        except TypeError:
            # fallback if API doesn't accept tuple imgsz
            try:
                results = model.predict(input_frame, imgsz=TARGET_WIDTH, device=DEVICE, conf=0.35, verbose=False)
            except Exception as e:
                print("[WARN] model.predict failed:", e)
                time.sleep(0.01)
                continue
        except Exception as e:
            print("[WARN] model.predict failed:", e)
            time.sleep(0.01)
            continue

        t_infer = time.time() - t0

        r = results[0]
        boxes = getattr(r, "boxes", None)
        preds = []
        if boxes is not None and getattr(boxes, "xyxy", None) is not None:
            xyxy = boxes.xyxy
            preds = xyxy.cpu().numpy() if isinstance(xyxy, torch.Tensor) else xyxy

        latest_count = len(preds)
        current_snail_count = latest_count

        # Decide whether to annotate to reduce drawing cost (annotate every ANNOTATE_EVERY_N frames)
        annotate_counter = (annotate_counter + 1) % ANNOTATE_EVERY_N
        annotated = input_frame if annotate_counter == 0 else input_frame.copy()

        if annotate_counter == 0 and len(preds) > 0:
            # draw boxes (on resized frame)
            for i, box in enumerate(preds):
                x1, y1, x2, y2 = map(int, box)
                # ensure coordinates fit resized dims
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(TARGET_WIDTH - 1, x2), min(TARGET_HEIGHT - 1, y2)
                conf_val = float(boxes.conf[i])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{conf_val:.2f}", (x1, max(10, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # show count
        cv2.putText(annotated, f"Count: {latest_count}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # upscale annotated to a display size (optional) or keep small to speed encoding
        display_frame = cv2.resize(annotated, (TARGET_WIDTH, TARGET_HEIGHT))

        with inference_lock:
            latest_inferred = display_frame

        # throttle to not exceed target FPS too aggressively (account for inference time)
        elapsed = time.time() - t0
        sleep_time = max(0.0, target_interval - elapsed)
        if sleep_time > 0:
            time.sleep(min(sleep_time, 0.02))  # sleep max 20ms to stay responsive

    print("[INFO] Inference thread stopped")

# ------------------------------
# MJPEG Stream Generator
# ------------------------------
def gen_frames():
    global latest_inferred
    print("[INFO] Stream generator started")
    while True:
        with inference_lock:
            if latest_inferred is None:
                # produce a tiny black image while we wait
                blank = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
                ret, buffer = cv2.imencode('.jpg', blank, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.03)
                continue
            frame = latest_inferred.copy()

        # encode with lower quality to reduce CPU/network
        encode_start = time.time()
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ret:
            continue
        # small adaptive sleep to limit stream CPU usage (very small)
        encode_elapsed = time.time() - encode_start
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        # slight throttle (allow browser to fetch)
        time.sleep(0.001)

# ------------------------------
# Flask Routes
# ------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/snail_count")
def snail_count():
    return {"count": current_snail_count}

# ------------------------------
# MAIN: start threads + flask
# ------------------------------
if __name__ == "__main__":
    print("[INFO] Starting multithreaded fast Snail Detector Dashboard...")
    t_cam = threading.Thread(target=camera_thread, daemon=True)
    t_inf = threading.Thread(target=inference_thread, daemon=True)
    t_cam.start()
    t_inf.start()

    # run Flask (main thread)
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        running = False
        t_cam.join(timeout=1.0)
        t_inf.join(timeout=1.0)
        print("[INFO] Shutdown complete")
