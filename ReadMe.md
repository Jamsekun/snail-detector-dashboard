## This dashboard is using Flask for rapid prototyping

### Github link:
https://github.com/Jamsekun/snail-detector-dashboard

### Files Drive link: 
https://drive.google.com/file/d/1Rbq3E7ODBcgu35EoYQKBFglkOH3NvcFh/view?usp=sharing

### Deployment Docs: 
https://docs.google.com/document/d/1Ord8GUadrCt9juEyxec3E2J4PcRICa8mgtLTRGmQCzY/edit?usp=sharing

### Diagram: 
https://drive.google.com/file/d/1h9UvJq3isVFq-7sfJFf6jpK9AfBp8Zjo/view?usp=drive_link

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

