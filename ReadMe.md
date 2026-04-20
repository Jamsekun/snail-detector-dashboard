## This dashboard is using Flask for rapid prototyping

### Github link:
https://github.com/Jamsekun/snail-detector-dashboard

### Files Drive link: 
https://drive.google.com/file/d/1Rbq3E7ODBcgu35EoYQKBFglkOH3NvcFh/view?usp=sharing

### Deployment Docs: 
https://docs.google.com/document/d/1Ord8GUadrCt9juEyxec3E2J4PcRICa8mgtLTRGmQCzY/edit?usp=sharing

### Diagram: 
https://drive.google.com/file/d/1h9UvJq3isVFq-7sfJFf6jpK9AfBp8Zjo/view?usp=drive_link

### IMPORTANT:
run it thru `dashboard\dashboard_snail_3classes.py`

### How to run After Cloning (make sure you are in the main outside folder)
1. `python -m venv venv`
2. `venv\Scripts\activate`
3. `pip install -r requirements.txt`
4. `python dashboard_snail_3classes.py`

# Changelog

## [2025-01-05]
### Added
- Statistics dropdown with 1-minute line chart
- Backend statistics data collection
- Frontend statistics UI (Chart.js)

### Changed
- Auto-update charts every 2s when expanded
- Current statistics refresh every 1s

### Fixed
- Added missing `last_stats_update` global in `inference_thread()`
- Added `total_snail_count` to global declarations
- Resolved statistics collection runtime error

---

## [2025-12-31]
### Changed
- Reduced JPEG quality from 80 → 70 for faster encoding
- Increased frame queue size from 2 → 3
- Lowered default `TARGET_FPS` from 60 → 30
- Optimized stream generator FPS throttling
- Reduced CPU usage during streaming

### Added
- Centralized performance configuration variables:
  - `TARGET_FPS`
  - `STREAM_FPS_TARGET`
  - `JPEG_QUALITY`
  - `SKIP_INFERENCE_FRAMES`
  - `ENABLE_FPS_DISPLAY`
  - `FPS_AVERAGE_WINDOW`

---

## [2025-12-27]
### Added
- On-screen FPS counters:
  - **INF FPS** (inference rate)
  - **STR FPS** (stream rate)
- Rolling FPS average over last 30 frames

---

## [2026-02-06]
### Added
- FusionTracker: new Kalman+IoU+Appearance tracker (keeps backward-compatible fields used by the dashboard)
- `/classes` API endpoint exposing model class id→name and color for the UI
- Unit tests for `FusionTracker` (dashboard/tests/test_fusion_tracker.py)
- `dashboard/README_tracker.md` with switching and tuning instructions

### Changed
- Default tracker selection moved to top-level flag `TRACKER_IMPL` (set to `"fusion"` or `"centroid"`) for easy switching
- Dashboard now displays per-class labels and a legend with deterministic colors
- Frontend: main dashboard counter (`static/script.js`) now prefers the live count (`live`) over total

### Fixed
- Prevent inference thread crashes from tracker KeyError by guarding `stable_frames` access and catching tracker exceptions (inference thread now logs errors and continues)
- Improved robustness of tracker assignment code (Hungarian / greedy fallback with gating)

### Notes
- Appearance embeddings use a cheap HSV-histogram fallback when a heavy embedder is not provided (CPU-friendly). Disable appearance matching by constructing `FusionTracker(use_appearance=False)` or set `TRACKER_IMPL = "centroid"` to use the simpler tracker.
- If you want me to run the unit tests in your venv or flip the default to `centroid` for lower CPU load, tell me and I'll update the repo and run them.


---

## [2026-04-20]
### Updated
- best.pt in \models
- added iou parameter in dashboard
- 300 Epoch training because maangas tayo 

## NEW Training March

ultralytics train model="yolo11n.pt" data="data.yaml" epochs=300 imgsz=640 batch=16 device=0 optimizer=AdamW save_period=10 name=snail_detector_IMPROVED plots=true val=true


## after training March, run to test tests 
yolo val model=runs/train/snail_detector_IMPROVED/weights/best.pt data=data.yaml split=test
