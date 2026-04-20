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
from typing import Any

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

TRACKER_MAX_DISTANCE = 100  # smaller = less ID switching

# ------------------------------
# SIMPLE CENTROID TRACKER
# ------------------------------
class CentroidTracker:
    def __init__(self, max_distance=100):
        self.next_id = 0
        self.objects = {}           # id -> centroid
        self.last_positions = {}    # id -> last centroid
        self.stable_frames = {}     # id -> consecutive stable frames
        self.counted_ids = set()    # ids already counted
        self.confidences = {}       # id -> confidence
        self.boxes = {}             # id -> bbox
        self.max_distance = max_distance

        # NEW: id -> class_id
        self.classes = {}

    def update(self, detections_with_conf):
        """
        Update tracker with detections.
        detections_with_conf: list of tuples ((x1,y1,x2,y2), conf, cls_id)
        Returns: dict of current objects id -> centroid
        """
        new_objects = {}
        new_confidences = {}
        new_boxes = {}
        new_classes = {}

        # Iterate detections: unpack bbox, confidence, and class id
        for det, conf, cls_id in detections_with_conf:
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
                # Only match to previous objects that are the same class (if previous class known).
                prev_cls = self.classes.get(obj_id, None)
                if prev_cls is not None and prev_cls != cls_id:
                    # don't match cross-class (prevents switching IDs between classes)
                    continue
                if dist < self.max_distance and dist < best_match_dist:
                    best_match_dist = dist
                    best_match_id = obj_id
                    matched = True

            if matched and best_match_id is not None:
                # Update existing object
                new_objects[best_match_id] = centroid
                self.last_positions[best_match_id] = self.objects[best_match_id]
                self.stable_frames[best_match_id] = self.stable_frames.get(best_match_id, 0) + 1

                # Update confidence, box and class for matched id
                new_confidences[best_match_id] = conf
                new_boxes[best_match_id] = det
                new_classes[best_match_id] = cls_id

            else:
                # Create new object
                new_id = self.next_id
                new_objects[new_id] = centroid
                self.last_positions[new_id] = centroid
                self.stable_frames[new_id] = 1
                new_confidences[new_id] = conf
                new_boxes[new_id] = det
                new_classes[new_id] = cls_id
                self.next_id += 1

        # Clean up tracking data for objects that disappeared
        disappeared_ids = set(self.objects.keys()) - set(new_objects.keys())
        for old_id in disappeared_ids:
            # Reset stable_frames and last_positions for disappeared objects
            if old_id not in self.counted_ids:
                self.stable_frames.pop(old_id, None)
                self.last_positions.pop(old_id, None)
            # Also remove old class/conf/box entries if present (they won't be carried forward)
            self.confidences.pop(old_id, None)
            self.boxes.pop(old_id, None)
            self.classes.pop(old_id, None)

        # Update tracking dictionaries to the new state
        self.objects = new_objects
        self.confidences = new_confidences
        self.boxes = new_boxes
        self.classes = new_classes

        return self.objects



# ------------------------------
# CONFIG - tune these for speed
# ------------------------------
MODEL_PATH = r"C:/James_folder/embedded_projects/Thesis_Clients/Snail_machine/dashboard/models/snail_detector3_newclass.pt"          # your model - try a small model (yolov8n / yolov8n-seg)

TARGET_WIDTH = 1280                     # inference input width (smaller -> faster) 480 or 1280
TARGET_HEIGHT = 640                    # inference input height (smaller -> faster) 640 or 720
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
# Tracker selection: set TRACKER_IMPL = "fusion" or "centroid"
TRACKER_IMPL = "fusion"  # options: "fusion" (default) or "centroid"
tracker: Any = None
# NOTE: FusionTracker implemented below provides a Kalman+IoU+Appearance fusion tracker.
class FusionTracker:
    """
    FusionTracker: combines a simple Kalman motion model, IoU gating and appearance embeddings.

    Backwards-compatible fields maintained:
      - objects: dict[id] -> centroid (x,y)
      - boxes: dict[id] -> (x1,y1,x2,y2)
      - confidences: dict[id] -> last confidence
      - classes: dict[id] -> class_id
      - counted_ids: set(ids)

    Edit nyo lang tong parameters guys 
    Hyperparameters (can be tuned externally):
      - max_distance: motion gating (pixels)
      - iou_threshold: minimum IoU to allow assignment
      - motion_weight, iou_weight, appearance_weight: weights for cost fusion
      - embedding_alpha: EMA weight for embedding updates
      - max_age: frames to keep alive without update
    """
    def __init__(self,
                 max_distance=TRACKER_MAX_DISTANCE,
                 iou_threshold=0.3,  # require more overlap to accept matches
                 motion_weight=0.6,  # motion useful but not dominant
                 iou_weight=1.0,  # make IoU the strongest spatial cue
                 appearance_weight=0.4,  # weak appearance if used; prefer lightweight features
                 embedding_alpha=0.15,  # slow embedding updates for stability
                 max_age=15,  # keep tracks longer to tolerate missed detections
                 use_appearance=False):  # start False on CPU; enable lightweight appearance later
        # identity & storage
        self.next_id = 0
        self.objects = {}           # id -> centroid
        self.boxes = {}             # id -> bbox
        self.confidences = {}
        self.classes = {}
        self.counted_ids = set()

        # track state containers
        self.kalman_states = {}     # id -> dict(state, P)
        self.embeddings = {}        # id -> embedding vector (numpy)
        self.age = {}               # id -> age (frames since creation)
        self.time_since_update = {} # id -> frames since last update
        self.hit_streak = {}        # id -> consecutive hits

        # compatibility fields used elsewhere in code
        self.stable_frames = {}     # id -> consecutive stable frames (used by counting logic)
        self.last_positions = {}    # id -> last centroid

        # params
        self.max_distance = float(max_distance)
        self.iou_threshold = float(iou_threshold)
        self.motion_weight = float(motion_weight)
        self.iou_weight = float(iou_weight)
        self.appearance_weight = float(appearance_weight)
        self.embedding_alpha = float(embedding_alpha)
        self.max_age = int(max_age)
        self.use_appearance = bool(use_appearance)

        # logging
        import logging
        self.log = logging.getLogger("FusionTracker")

        # attempt to import Hungarian solver
        try:
            from scipy.optimize import linear_sum_assignment
            self._linear_sum_assignment = linear_sum_assignment
        except Exception:
            try:
                from munkres import Munkres
                self._munkres = Munkres()
                self._linear_sum_assignment = None
            except Exception:
                self._linear_sum_assignment = None
                self._munkres = None
                self.log.warning("No Hungarian solver (scipy/munkres) found: falling back to greedy assignment")

    # ----------------------
    # Simple Kalman: state [cx,cy,vx,vy]
    # ----------------------
    def _init_kalman(self, centroid):
        state = np.array([centroid[0], centroid[1], 0.0, 0.0], dtype=float)
        P = np.eye(4, dtype=float) * 50.0
        return {"x": state, "P": P}

    def _predict_kalman(self, kf, dt=1.0):
        # Constant velocity model
        F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=float)
        Q = np.eye(4, dtype=float) * 1.0
        x = kf["x"]
        P = kf["P"]
        x_pred = F.dot(x)
        P_pred = F.dot(P).dot(F.T) + Q
        return {"x": x_pred, "P": P_pred}

    def _update_kalman(self, kf_pred, centroid):
        H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        R = np.eye(2, dtype=float) * 10.0
        x_pred = kf_pred["x"]
        P_pred = kf_pred["P"]
        z = np.array([centroid[0], centroid[1]], dtype=float)
        y = z - H.dot(x_pred)
        S = H.dot(P_pred).dot(H.T) + R
        K = P_pred.dot(H.T).dot(np.linalg.inv(S))
        x_upd = x_pred + K.dot(y)
        P_upd = (np.eye(4) - K.dot(H)).dot(P_pred)
        return {"x": x_upd, "P": P_upd}

    # ----------------------
    # IoU and appearance helpers
    # ----------------------
    @staticmethod
    def _iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
        boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
        denom = float(boxAArea + boxBArea - interArea) + 1e-6
        return interArea / denom

    @staticmethod
    def _norm_dist(a, b, max_dist):
        return np.linalg.norm(np.array(a)-np.array(b)) / (max_dist + 1e-6)

    @staticmethod
    def _cosine_dist(a, b):
        if a is None or b is None:
            return 1.0
        a = np.asarray(a)
        b = np.asarray(b)
        na = np.linalg.norm(a) + 1e-6
        nb = np.linalg.norm(b) + 1e-6
        return 1.0 - float(np.dot(a, b) / (na * nb))

    def _extract_embedding(self, frame, box):
        """
        Extract a cheap appearance embedding. If a heavy embedder is not provided
        the function falls back to a small HSV histogram computed on CPU.
        """
        if frame is None:
            return None
        x1,y1,x2,y2 = box
        h,w = frame.shape[:2]
        x1c = max(0, min(w-1, x1)); y1c = max(0, min(h-1, y1))
        x2c = max(0, min(w, x2)); y2c = max(0, min(h, y2))
        if x2c <= x1c or y2c <= y1c:
            return None
        crop = frame[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0,1], None, [16,8], [0,180,0,256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    # ----------------------
    # Main update
    # detections_with_conf: list of ((x1,y1,x2,y2), conf, cls_id)
    # optional frame for embedding extraction
    # ----------------------
    def update(self, detections_with_conf, frame=None):
        # Predict all tracks
        preds = {}
        for tid, kf in list(self.kalman_states.items()):
            kf_pred = self._predict_kalman(kf)
            preds[tid] = kf_pred

        # Build detection data
        det_boxes = [d[0] for d in detections_with_conf]
        det_confs = [d[1] for d in detections_with_conf]
        det_clses = [d[2] for d in detections_with_conf]
        det_centroids = [((b[0]+b[2])//2, (b[1]+b[3])//2) for b in det_boxes]
        det_embs = []
        if self.use_appearance:
            for b in det_boxes:
                try:
                    emb = self._extract_embedding(frame, b)
                except Exception:
                    emb = None
                det_embs.append(emb)
        else:
            det_embs = [None]*len(det_boxes)

        # If no existing tracks, create new ones from detections
        if len(self.kalman_states) == 0:
            for i, (box, conf, cls_id, cent, emb) in enumerate(zip(det_boxes, det_confs, det_clses, det_centroids, det_embs)):
                tid = self.next_id
                self.next_id += 1
                self.objects[tid] = cent
                self.boxes[tid] = box
                self.confidences[tid] = conf
                self.classes[tid] = cls_id
                self.kalman_states[tid] = self._init_kalman(cent)
                self.embeddings[tid] = emb
                self.age[tid] = 1
                self.time_since_update[tid] = 0
                self.hit_streak[tid] = 1
            return self.objects

        # Build cost matrix between predicted tracks and detections
        track_ids = list(self.kalman_states.keys())
        M = len(track_ids)
        N = len(det_boxes)
        cost = np.ones((M, N), dtype=float)
        # Fill costs
        for i, tid in enumerate(track_ids):
            kf_pred = preds.get(tid, None)
            pred_cent = tuple(map(int, kf_pred["x"][0:2])) if kf_pred is not None else tuple(self.objects.get(tid, (0,0)))
            pred_box = self.boxes.get(tid, None)
            emb_t = self.embeddings.get(tid, None)
            for j, db in enumerate(det_boxes):
                det_cent = det_centroids[j]
                det_emb = det_embs[j]
                # normalized motion distance
                d_motion = self._norm_dist(pred_cent, det_cent, self.max_distance)
                # IoU cost
                d_iou = 1.0 - (self._iou(pred_box, db) if pred_box is not None else 0.0)
                # appearance cost (cosine)
                d_app = self._cosine_dist(emb_t, det_emb) if self.use_appearance else 0.0
                fused = (self.motion_weight * d_motion + self.iou_weight * d_iou + self.appearance_weight * d_app)
                # Gating: if IoU below threshold and motion large, penalize heavily
                iou_val = self._iou(pred_box, db) if pred_box is not None else 0.0
                if iou_val < self.iou_threshold and d_motion > 1.0:
                    fused += 10.0
                cost[i, j] = fused

        # Solve assignment
        assigned_tracks = set()
        assigned_dets = set()
        assignments = []
        if M > 0 and N > 0:
            if self._linear_sum_assignment is not None:
                row_ind, col_ind = self._linear_sum_assignment(cost)
                for r,c in zip(row_ind, col_ind):
                    # accept assignments with reasonable cost
                    if cost[r,c] < 5.0:  # configurable gating threshold
                        assignments.append((track_ids[r], c, cost[r,c]))
                    else:
                        self.log.debug(f"Rejection by cost: track {track_ids[r]} -> det {c} cost={cost[r,c]:.3f}")
            elif getattr(self, "_munkres", None) is not None:
                matrix = (cost * 1000).astype(int).tolist()
                # Munkres.compute exists on the instance but static analyzers may not know; ignore attribute/type here
                mres = self._munkres.compute(matrix)  # type: ignore[attr-defined]
                for r, c in mres:
                    if r < M and c < N and cost[r, c] < 5.0:
                        assignments.append((track_ids[r], c, cost[r, c]))
            else:
                # greedy fallback
                flat = []
                for i in range(M):
                    for j in range(N):
                        flat.append((cost[i,j], i, j))
                for _, i, j in sorted(flat):
                    if i in assigned_tracks or j in assigned_dets:
                        continue
                    if cost[i,j] < 5.0:
                        assignments.append((track_ids[i], j, cost[i,j]))
                        assigned_tracks.add(track_ids[i]); assigned_dets.add(j)

        # Logging assignment costs
        if len(assignments) > 0:
            self.log.debug(f"Assignments: {[ (a,b,round(c,3)) for a,b,c in assignments ]}")

        # Update assigned tracks
        updated_tracks = set()
        for tid, det_idx, cval in assignments:
            box = det_boxes[det_idx]
            conf = det_confs[det_idx]
            cls_id = det_clses[det_idx]
            cent = det_centroids[det_idx]
            emb = det_embs[det_idx]

            # Update Kalman with detection
            kf_pred = preds.get(tid)
            kf_upd = self._update_kalman(kf_pred, cent)
            self.kalman_states[tid] = kf_upd
            self.objects[tid] = tuple(map(int, kf_upd["x"][0:2]))
            # compatibility: update last_positions and stable_frames
            self.last_positions[tid] = self.objects[tid]
            self.stable_frames[tid] = self.stable_frames.get(tid, 0) + 1
            self.boxes[tid] = box
            self.confidences[tid] = conf
            self.classes[tid] = cls_id

            # embedding EMA
            if emb is not None:
                prev = self.embeddings.get(tid)
                if prev is None:
                    self.embeddings[tid] = emb
                else:
                    self.embeddings[tid] = (1.0 - self.embedding_alpha) * prev + self.embedding_alpha * emb

            # lifecycle
            self.age[tid] = self.age.get(tid, 0) + 1
            self.time_since_update[tid] = 0
            self.hit_streak[tid] = self.hit_streak.get(tid, 0) + 1
            updated_tracks.add(tid)

        # Mark unmatched tracks as missed
        for tid in list(self.kalman_states.keys()):
            if tid not in updated_tracks:
                self.time_since_update[tid] = self.time_since_update.get(tid, 0) + 1
                self.age[tid] = self.age.get(tid, 0) + 1
                self.hit_streak[tid] = 0
                # reduce stable_frames when missed
                self.stable_frames[tid] = 0

        # Create new tracks for unmatched detections
        for j in range(N):
            if any(det_idx == j for _, det_idx, _ in assignments):
                continue
            box = det_boxes[j]
            conf = det_confs[j]
            cls_id = det_clses[j]
            cent = det_centroids[j]
            emb = det_embs[j]
            tid = self.next_id
            self.next_id += 1
            self.kalman_states[tid] = self._init_kalman(cent)
            self.objects[tid] = cent
            self.last_positions[tid] = cent
            self.stable_frames[tid] = 1
            self.boxes[tid] = box
            self.confidences[tid] = conf
            self.classes[tid] = cls_id
            self.embeddings[tid] = emb
            self.age[tid] = 1
            self.time_since_update[tid] = 0
            self.hit_streak[tid] = 1

        # Remove old tracks beyond max_age
        for tid in list(self.kalman_states.keys()):
            if self.time_since_update.get(tid, 0) > self.max_age:
                # log ID switch/event
                self.log.debug(f"Removing track {tid} due to age (time_since_update={self.time_since_update.get(tid)})")
                for d in (self.objects, self.boxes, self.confidences, self.classes, self.kalman_states, self.embeddings, self.age, self.time_since_update, self.hit_streak, self.stable_frames, self.last_positions):
                    d.pop(tid, None)

        # Ensure compatibility: expose tracker.objects, boxes, confidences, classes, counted_ids
        return self.objects


# Instantiate tracker according to TRACKER_IMPL
if TRACKER_IMPL.lower().startswith("fusion"):
    tracker = FusionTracker()
else:
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
CLASS_NAMES = model.names  # dict or list: {id: "label"} or ["label", ...]
# Normalize CLASS_NAMES to a dict[int->str] so code works regardless of ultralytics version
if isinstance(CLASS_NAMES, (list, tuple)):
    CLASS_NAMES = {i: name for i, name in enumerate(CLASS_NAMES)}
elif isinstance(CLASS_NAMES, dict):
    # ensure keys are ints
    CLASS_NAMES = {int(k): v for k, v in CLASS_NAMES.items()}
else:
    # fallback
    CLASS_NAMES = dict()

print("[INFO] Model classes:", CLASS_NAMES)

# Precompute the class id(s) that correspond to "snail" (case-insensitive).
# This lets the dashboard count ONLY snails while still displaying/all classes.
SNAIL_CLASS_IDS = {cid for cid, name in CLASS_NAMES.items() if str(name).lower() == "snail"}
print(f"[INFO] Snail class ids: {SNAIL_CLASS_IDS}")

# Create a deterministic color for each class id for consistent visualization
def _make_color(cls_id: int):
    # simple hash -> color mapping (BGR)
    rng = (cls_id * 123457) & 0xFFFFFF
    b = (rng & 0xFF)
    g = ((rng >> 8) & 0xFF)
    r = ((rng >> 16) & 0xFF)
    return (int(b), int(g), int(r))

CLASS_COLORS = {cid: _make_color(cid) for cid in CLASS_NAMES.keys()}


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

        # before running inference, ensure these are defined
        preds = []
        confs = []
        clses = []

        if should_skip_inference:
            # Skip inference, use previous detections (tracker maintains state) # preds, confs, clses remain empty
            pass
        else:
            try:
                # prefer passing device and imgsz explicitly; pass half if supported
                predict_kwargs = dict(imgsz=(TARGET_HEIGHT, TARGET_WIDTH), device=DEVICE, conf=0.50, verbose=False)
                if USE_HALF:
                    predict_kwargs["half"] = True
                results = model.predict( input_frame, imgsz=(TARGET_HEIGHT, TARGET_WIDTH), device=DEVICE, conf=0.50, iou=.30, verbose=False, half=USE_HALF if USE_HALF else False )
            except TypeError:
                # fallback if API doesn't accept tuple imgsz
                try:
                    results = model.predict(input_frame, imgsz=TARGET_WIDTH, device=DEVICE, conf=0.50, iou=.30, verbose=False)
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
            
            # Extract bounding boxes and confidence scores
            if boxes is not None and getattr(boxes, "xyxy", None) is not None:
                preds = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clses = boxes.cls.cpu().numpy().astype(int)
                
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
            cls_id = clses[i]
            conf = float(confs[i])

            label = CLASS_NAMES.get(cls_id, "unknown")

            if label == "snail":
                if not valid_snail_box(x1, y1, x2, y2):
                    continue

            filtered_detections.append(((x1, y1, x2, y2), conf, cls_id))


        # Update tracker with filtered detections (includes confidence)
        try:
            tracked = tracker.update(filtered_detections, frame=input_frame)
        except Exception as e:
            # Don't let tracker errors kill the inference thread — log and continue with empty state
            try:
                tracker.log.exception("Tracker update failed: %s", e)
            except Exception:
                print("[ERROR] Tracker update failed:", e)
            tracked = {}

        # Count only objects that belong to the snail class(es)
        latest_count = sum(1 for obj_id in tracked.keys() if tracker.classes.get(obj_id) in SNAIL_CLASS_IDS)

        for obj_id, centroid in tracked.items():    
            global total_snail_count

            # Only consider snail-class objects for counting and stability
            if tracker.classes.get(obj_id) not in SNAIL_CLASS_IDS:
                continue

            # motion validation
            if not slow_enough(obj_id, centroid, tracker):
                continue

            # stability check
            if tracker.stable_frames.get(obj_id, 0) >= STABILITY_FRAMES:
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
            # Draw bounding boxes with class name, confidence scores and IDs
            for obj_id in tracked.keys():
                if obj_id in tracker.boxes:
                    x1, y1, x2, y2 = tracker.boxes[obj_id]
                    conf = tracker.confidences.get(obj_id, 0.0)

                    # find class id and name for this object (tracker.classes set in update)
                    cls_id = tracker.classes.get(obj_id, None)
                    cls_name = CLASS_NAMES.get(cls_id, "unknown") if cls_id is not None else "unknown"

                    # choose color by class (fallback green)
                    color = CLASS_COLORS.get(cls_id, (0, 255, 0))

                    # Draw bounding box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                    # Draw centroid
                    cx, cy = tracked[obj_id]
                    cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)

                    # Display Class, ID and confidence score
                    label = f"{cls_name} ID:{obj_id} {conf:.2f}"
                    # Check if counted (stable)
                    if obj_id in tracker.counted_ids:
                        label += " ✓"

                    # Calculate text size for background
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )

                    # Draw text background for better visibility
                    cv2.rectangle(annotated,
                                (x1, y1 - text_height - 6),
                                (x1 + text_width, y1),
                                (0, 0, 0), -1)

                    # Draw text
                    cv2.putText(annotated, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw legend of classes and colors on the top-right
            try:
                start_x = TARGET_WIDTH - 200
                start_y = 10
                box_h = 20
                padding = 6
                for cid, name in CLASS_NAMES.items():
                    color = CLASS_COLORS.get(cid, (0, 255, 0))
                    # small color box
                    cv2.rectangle(annotated, (start_x, start_y), (start_x + box_h, start_y + box_h), color, -1)
                    cv2.putText(annotated, f"{cid}: {name}", (start_x + box_h + 8, start_y + box_h - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    start_y += box_h + padding
            except Exception:
                # don't break annotation if legend drawing fails
                pass

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


@app.route("/classes")
def classes_endpoint():
    """Return model class id -> name mapping and color (hex) for UI consumption."""
    mapping = {}
    for cid, name in CLASS_NAMES.items():
        col = CLASS_COLORS.get(cid, (0, 255, 0))
        # convert BGR to hex RGB for web UI
        b, g, r = col
        hexcol = "#{:02x}{:02x}{:02x}".format(r, g, b)
        mapping[int(cid)] = {"name": name, "color": hexcol}
    return jsonify(mapping)


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
