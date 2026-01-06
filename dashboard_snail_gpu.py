# dashboard_snail_fast.py
from flask import Flask, Response, render_template, jsonify
import cv2
import threading
import time
import numpy as np
import torch
from ultralytics import YOLO  # type: ignore
import queue
import os
import sys
from collections import deque

app = Flask(__name__)


# ------------------------------
# COUNTING & FILTER CONFIG
# ------------------------------
MIN_BOX_AREA = 800        # ignore tiny noise (pixels²) // Ibahin ito at iresize if malayo ang snail.
MAX_BOX_AREA = 200000     # ignore unrealistically large blobs
MIN_ASPECT_RATIO = 0.4   # snail shape filtering
MAX_ASPECT_RATIO = 2.5
MIN_WIDTH = 20            # minimum bounding box width (pixels)
MIN_HEIGHT = 20           # minimum bounding box height (pixels)

STABILITY_FRAMES = 3     # N-frame rule
MAX_MOVE_DISTANCE = 35   # snail speed (pixels/frame)

TRACKER_MAX_DISTANCE = 50  # smaller = less ID switching

# ------------------------------
# SIMPLE CENTROID TRACKER
# ------------------------------
class CentroidTracker:
    def __init__(self, max_distance= TRACKER_MAX_DISTANCE):
        self.next_id = 0
        self.objects = {}          # id -> centroid
        self.last_positions = {}   # id -> previous centroid
        self.stable_frames = {}    # id -> consecutive frames
        self.counted_ids = set()
        self.max_distance = max_distance
        self.confidences = {}      # id -> confidence score
        self.boxes = {}            # id -> (x1, y1, x2, y2) for visualization

    def update(self, detections_with_conf):
        """
        Update tracker with detections.
        detections_with_conf: list of tuples ((x1, y1, x2, y2), confidence)
        """
        new_objects = {}
        new_confidences = {}
        new_boxes = {}

        for det, conf in detections_with_conf:
            (x1, y1, x2, y2) = det
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            centroid = (cx, cy)

            matched = False
            best_match_id = None
            best_match_dist = float('inf')
            
            # Find closest matching existing object
            for obj_id, prev_centroid in self.objects.items():
                dist = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
                if dist < self.max_distance and dist < best_match_dist:
                    best_match_dist = dist
                    best_match_id = obj_id
                    matched = True

            if matched and best_match_id is not None:
                # Update existing object
                new_objects[best_match_id] = centroid
                self.last_positions[best_match_id] = self.objects[best_match_id]
                self.stable_frames[best_match_id] = self.stable_frames.get(best_match_id, 0) + 1
                # Update confidence (use average or latest - using latest here)
                new_confidences[best_match_id] = conf
                new_boxes[best_match_id] = det
            else:
                # Create new object
                new_objects[self.next_id] = centroid
                self.last_positions[self.next_id] = centroid
                self.stable_frames[self.next_id] = 1
                new_confidences[self.next_id] = conf
                new_boxes[self.next_id] = det
                self.next_id += 1

        # Clean up tracking data for objects that disappeared
        # (only keep data for objects that still exist)
        disappeared_ids = set(self.objects.keys()) - set(new_objects.keys())
        for old_id in disappeared_ids:
            # Reset stable_frames for disappeared objects (they'll restart if they reappear)
            # Note: We don't remove from counted_ids to prevent double-counting
            if old_id not in self.counted_ids:
                # Only reset if not counted yet (allow reappearance to be tracked fresh)
                self.stable_frames.pop(old_id, None)
                self.last_positions.pop(old_id, None)

        # Update tracking dictionaries
        self.objects = new_objects
        self.confidences = new_confidences
        self.boxes = new_boxes
        return self.objects


# ------------------------------
# CONFIG - tune these for speed
# ------------------------------
MODEL_PATH = "models/last.pt"          # your model - try a small model (yolov8n / yolov8n-seg)
TARGET_WIDTH = 1280                     # inference input width (smaller -> faster) 480 or 1280
TARGET_HEIGHT = 720                    # inference input height (smaller -> faster) 640 or 720
TARGET_FPS = 30.0                      # target visual framerate (lower = smoother if hardware limited)
ANNOTATE_EVERY_N = 1                   # draw boxes on every Nth inference (increase to reduce drawing cost)
JPEG_QUALITY = 70                      # encode quality (0-100) smaller = faster & less bandwidth (lowered for speed)
FRAME_QUEUE_MAXSIZE = 3                # keep only most recent frames (increased for smoother playback)
WARMUP_ROUNDS = 2                      # run small warmups on model

# Performance optimization settings
SKIP_INFERENCE_FRAMES = 0              # skip inference every N frames (0 = no skip, 1 = every other frame, etc.)
STREAM_FPS_TARGET = 30.0               # target FPS for stream (can be different from inference FPS)
ENABLE_FPS_DISPLAY = True              # show FPS counter on screen
FPS_AVERAGE_WINDOW = 30                # number of frames to average for FPS calculation



# ------------------------------
# GLOBAL STATE
# ------------------------------
frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)   # camera -> inference
latest_inferred = None
latest_count = 0
current_snail_count = 0
inference_lock = threading.Lock()
running = True
tracker = CentroidTracker()
total_snail_count: int = 0

# FPS tracking
inference_fps = 0.0
stream_fps = 0.0
inference_times = []
stream_times = []
fps_lock = threading.Lock()

# Historical statistics (1-minute timeframe)
STATS_HISTORY_SIZE = 60  # 60 data points for 1 minute (1 per second)
stats_history = deque(maxlen=STATS_HISTORY_SIZE)
stats_lock = threading.Lock()
last_stats_update = 0.0



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

# helper function to validate snail box
def valid_snail_box(x1, y1, x2, y2):
    """
    Validate bounding box using spatial filtering:
    - Size constraints (area, width, height)
    - Shape constraints (aspect ratio)
    """
    w = x2 - x1
    h = y2 - y1
    area = w * h
    aspect_ratio = w / float(h + 1e-6)

    # Spatial filtering: area constraints
    if area < MIN_BOX_AREA or area > MAX_BOX_AREA:
        return False
    
    # Spatial filtering: minimum width/height
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return False
    
    # Shape filtering: aspect ratio (snails have consistent proportions)
    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        return False
    return True

def slow_enough(obj_id, centroid, tracker):
    prev = tracker.last_positions.get(obj_id)
    if prev is None:
        return True
    dist = np.linalg.norm(np.array(centroid) - np.array(prev))
    return dist <= MAX_MOVE_DISTANCE


def inference_thread():
    global latest_inferred, latest_count, current_snail_count, running
    global inference_fps, inference_times, last_stats_update, total_snail_count

    print("[INFO] Inference thread started (device: %s, half=%s)" % (DEVICE, USE_HALF))
    annotate_counter = 0
    frame_skip_counter = 0

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

        # Skip inference on some frames for performance (if configured)
        # Note: We still process every frame, but can skip expensive inference
        frame_skip_counter += 1
        should_skip_inference = (SKIP_INFERENCE_FRAMES > 0 and 
                                 (frame_skip_counter % (SKIP_INFERENCE_FRAMES + 1)) != 0)

        # run inference and measure time (skip if configured)
        t0 = time.time()
        
        if should_skip_inference:
            # Skip inference, use previous detections (tracker maintains state)
            preds = []
            confs = []
        else:
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
            confs = []
            
            # Extract bounding boxes and confidence scores
            if boxes is not None and getattr(boxes, "xyxy", None) is not None:
                xyxy = boxes.xyxy
                preds = xyxy.cpu().numpy() if isinstance(xyxy, torch.Tensor) else xyxy
                
                # Extract confidence scores
                if hasattr(boxes, "conf") and boxes.conf is not None:
                    confs_tensor = boxes.conf
                    confs = confs_tensor.cpu().numpy() if isinstance(confs_tensor, torch.Tensor) else confs_tensor
                else:
                    # Fallback: use default confidence if not available
                    confs = [0.5] * len(preds)

        # Apply spatial and shape filtering
        filtered_detections = []
        for i, box in enumerate(preds):
            x1, y1, x2, y2 = map(int, box)
            if valid_snail_box(x1, y1, x2, y2):
                conf = float(confs[i]) if i < len(confs) else 0.5
                filtered_detections.append(((x1, y1, x2, y2), conf))

        # Update tracker with filtered detections (includes confidence)
        tracked = tracker.update(filtered_detections)

        latest_count = len(tracked)

        for obj_id, centroid in tracked.items():
            global total_snail_count

            # motion validation
            if not slow_enough(obj_id, centroid, tracker):
                continue

            # stability check
            if tracker.stable_frames[obj_id] >= STABILITY_FRAMES:
                if obj_id not in tracker.counted_ids:
                    tracker.counted_ids.add(obj_id)
                    total_snail_count += 1

        current_snail_count = latest_count
        
        # Update statistics history (once per second)
        current_time = time.time()
        if current_time - last_stats_update >= 1.0:  # Update every second
            with fps_lock:
                inf_fps = inference_fps
                str_fps = stream_fps
            
            with stats_lock:
                stats_history.append({
                    'timestamp': current_time,
                    'total_count': total_snail_count,
                    'live_count': latest_count,
                    'inference_fps': inf_fps,
                    'stream_fps': str_fps
                })
            last_stats_update = current_time




        # Decide whether to annotate to reduce drawing cost (annotate every ANNOTATE_EVERY_N frames)
        annotate_counter = (annotate_counter + 1) % ANNOTATE_EVERY_N
        annotated = input_frame if annotate_counter == 0 else input_frame.copy()

        if annotate_counter == 0:
            # Draw bounding boxes with confidence scores and IDs
            for obj_id in tracked.keys():
                if obj_id in tracker.boxes:
                    x1, y1, x2, y2 = tracker.boxes[obj_id]
                    conf = tracker.confidences.get(obj_id, 0.0)
                    
                    # Draw bounding box (green for tracked snails)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw centroid
                    cx, cy = tracked[obj_id]
                    cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
                    
                    # Display ID and confidence score
                    label = f"ID:{obj_id} {conf:.2f}"
                    # Check if counted (stable)
                    if obj_id in tracker.counted_ids:
                        label += " ✓"
                    
                    # Calculate text size for background
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    
                    # Draw text background for better visibility
                    cv2.rectangle(annotated, 
                                (x1, y1 - text_height - 5), 
                                (x1 + text_width, y1), 
                                (0, 0, 0), -1)
                    
                    # Draw text
                    cv2.putText(annotated, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Display statistics
            cv2.putText(annotated, f"LIVE: {latest_count}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(annotated, f"TOTAL: {total_snail_count}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Display stability info
            stable_count = sum(1 for fid in tracked.keys() 
                             if tracker.stable_frames.get(fid, 0) >= STABILITY_FRAMES)
            cv2.putText(annotated, f"STABLE: {stable_count}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # Display FPS counter
            if ENABLE_FPS_DISPLAY:
                with fps_lock:
                    inf_fps = inference_fps
                    str_fps = stream_fps
                cv2.putText(annotated, f"INF FPS: {inf_fps:.1f}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(annotated, f"STR FPS: {str_fps:.1f}", (10, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)



        # upscale annotated to a display size (optional) or keep small to speed encoding
        display_frame = cv2.resize(annotated, (TARGET_WIDTH, TARGET_HEIGHT))

        # Calculate and update inference FPS
        elapsed = time.time() - t0
        with fps_lock:
            inference_times.append(time.time())
            # Keep only recent times for averaging
            if len(inference_times) > FPS_AVERAGE_WINDOW:
                inference_times.pop(0)
            # Calculate FPS from time differences
            if len(inference_times) > 1:
                time_span = inference_times[-1] - inference_times[0]
                if time_span > 0:
                    inference_fps = (len(inference_times) - 1) / time_span

        with inference_lock:
            latest_inferred = display_frame

        # throttle to not exceed target FPS too aggressively (account for inference time)
        sleep_time = max(0.0, target_interval - elapsed)
        if sleep_time > 0:
            time.sleep(min(sleep_time, 0.02))  # sleep max 20ms to stay responsive

    print("[INFO] Inference thread stopped")

# ------------------------------
# MJPEG Stream Generator
# ------------------------------
def gen_frames():
    global latest_inferred, stream_fps, stream_times
    print("[INFO] Stream generator started")
    
    stream_interval = 1.0 / STREAM_FPS_TARGET if STREAM_FPS_TARGET > 0 else 0.033
    last_stream_time = time.time()
    
    while True:
        current_time = time.time()
        
        with inference_lock:
            if latest_inferred is None:
                # produce a tiny black image while we wait
                blank = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
                ret, buffer = cv2.imencode('.jpg', blank, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(stream_interval)
                continue
            frame = latest_inferred.copy()

        # encode with lower quality to reduce CPU/network
        encode_start = time.time()
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ret:
            continue
        
        # Calculate and update stream FPS
        encode_elapsed = time.time() - encode_start
        with fps_lock:
            stream_times.append(time.time())
            # Keep only recent times for averaging
            if len(stream_times) > FPS_AVERAGE_WINDOW:
                stream_times.pop(0)
            # Calculate FPS from time differences
            if len(stream_times) > 1:
                time_span = stream_times[-1] - stream_times[0]
                if time_span > 0:
                    stream_fps = (len(stream_times) - 1) / time_span
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        # Throttle stream to target FPS
        elapsed_since_last = current_time - last_stream_time
        sleep_time = max(0.0, stream_interval - elapsed_since_last - encode_elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
        last_stream_time = time.time()

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
    return {
        "live": latest_count,
        "total": total_snail_count
    }

@app.route("/stats_history")
def stats_history_endpoint():
    """Return historical statistics for the last minute"""
    with stats_lock:
        # Convert deque to list and format timestamps
        history = list(stats_history)
        # Calculate relative time (seconds ago)
        current_time = time.time()
        formatted_history = []
        for stat in history:
            formatted_history.append({
                'time_ago': int(current_time - stat['timestamp']),  # seconds ago
                'total_count': stat['total_count'],
                'live_count': stat['live_count'],
                'inference_fps': round(stat['inference_fps'], 1),
                'stream_fps': round(stat['stream_fps'], 1)
            })
        return jsonify(formatted_history)


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
