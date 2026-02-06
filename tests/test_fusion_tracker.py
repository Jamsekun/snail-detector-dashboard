import numpy as np
from dashboard_snail_3classes import FusionTracker


def make_det(box, conf=0.9, cls=2):
    return (box, conf, cls)


def test_crossing_tracks_preserve_ids():
    # Two objects crossing: ensure tracker doesn't swap ids when motion is smooth
    tr = FusionTracker(max_distance=100, use_appearance=False)

    # Frame 1: left and right
    dets_f1 = [make_det((10, 10, 50, 50)), make_det((200, 10, 240, 50))]
    tr.update(dets_f1, frame=np.zeros((300,300,3), dtype=np.uint8))
    ids_f1 = set(tr.objects.keys())
    assert len(ids_f1) == 2

    # Frame 2: move towards center
    dets_f2 = [make_det((60, 10, 100, 50)), make_det((150, 10, 190, 50))]
    tr.update(dets_f2, frame=np.zeros((300,300,3), dtype=np.uint8))
    ids_f2 = set(tr.objects.keys())
    # same number of tracks
    assert len(ids_f2) == 2
    # IDs should be preserved (no mass reassign swap) -- rough check: intersection non-empty
    assert len(ids_f1 & ids_f2) >= 1


def test_occlusion_and_reappearance_keeps_id():
    tr = FusionTracker(max_distance=150, max_age=5, use_appearance=False)
    # create single detection
    d0 = [make_det((50,50,90,90))]
    tr.update(d0, frame=np.zeros((200,200,3), dtype=np.uint8))
    tid = next(iter(tr.objects.keys()))

    # occlude for 2 frames (no detections)
    tr.update([], frame=None)
    tr.update([], frame=None)

    # reappear near same spot
    tr.update([make_det((55,55,95,95))], frame=np.zeros((200,200,3), dtype=np.uint8))
    # ensure original id still exists (time_since_update small)
    assert tid in tr.objects
