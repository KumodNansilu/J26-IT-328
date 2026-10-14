"""MediaPipe Face Mesh wrapper (468 landmarks).

Landmarks support facial analysis ONLY. They are NOT fed into the CNN —
the CNN reads facial image crops. Returns None when MediaPipe is missing
so validation can still run without it.
"""
from __future__ import annotations

_mesh = None


def get_mesh():
    """Lazily create the FaceMesh object, or None if mediapipe missing."""
    global _mesh
    if _mesh is not None:
        return _mesh
    try:
        import mediapipe as mp
    except ImportError:
        return None
    _mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=False, min_detection_confidence=0.5)
    return _mesh


def extract_landmarks(bgr_crop):
    """Return list of (x, y) normalized landmarks, or None if no face/none."""
    mesh = get_mesh()
    if mesh is None or bgr_crop is None or getattr(bgr_crop, "size", 0) == 0:
        return None
    import cv2

    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    res = mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    return [(lm.x, lm.y) for lm in res.multi_face_landmarks[0].landmark]


def close():
    global _mesh
    try:
        if _mesh is not None:
            _mesh.close()
    finally:
        _mesh = None
