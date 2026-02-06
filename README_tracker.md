FusionTracker (short)
======================

This project ships a `FusionTracker` implementation in `dashboard_snail_3classes.py`.

How to switch from the simple centroid tracker to the FusionTracker

- In the same file the `tracker` variable is set to `FusionTracker()` by default.
- To fall back to the older `CentroidTracker`, change the assignment near the top of the file:

    tracker = CentroidTracker()  # old
    tracker = FusionTracker()    # new

Main hyperparameters (tune these at top of file or when constructing):
- max_distance (pixels): motion gating / normalization
- iou_threshold: minimum IoU to allow a cheap association
- motion_weight, iou_weight, appearance_weight: weights used to fuse cost components
- embedding_alpha: EMA coefficient for updating track embeddings (0..1)
- max_age: frames to keep tracks alive without updates

Notes:
- Appearance extraction uses a cheap HSV histogram if no heavy embedder is provided. It runs on CPU.
- Hungarian assignment (SciPy) is used when available. There is a greedy fallback if not.
- Unit tests live in `dashboard/tests/test_fusion_tracker.py`.

Logging:
- The tracker uses the `FusionTracker` logger. Set logging level to DEBUG to see assignments and ID removal logs.
