"""MediaPipe Face Mesh wrapper (468 landmarks).

Landmarks support facial analysis ONLY. They are NOT fed into the CNN —
the CNN reads facial image crops. Returns None when MediaPipe is missing
so validation can still run without it.

Supports BOTH APIs (installed version wins, no outdated code copied):
- mediapipe.tasks (newer: FaceLandmarker + .task bundle download)
- mp.solutions.face_mesh (classic, ships in most wheels)
Use extract_landmarks_with_backend() to learn which backend ran.
"""
from __future__ import annotations

_mesh = None
_mesh_backend = "none"


def _make_solutions_mesh():
    import mediapipe as mp

    holder = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=False, min_detection_confidence=0.5)
    return holder, "solutions"


def _make_tasks_landmarker():
    from pathlib import Path

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    root = Path(__file__).resolve().parents[3]
    bundle = root / "models" / "face_landmarker.task"
    if not bundle.exists():
        raise FileNotFoundError(
            f"FaceLandmarker bundle missing: {bundle}. Download it from "
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task "
            "into models/face_landmarker.task, or rely on the classic "
            "mp.solutions.face_mesh API instead.")
    base = mp_python.BaseOptions(model_asset_path=str(bundle))
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base, num_faces=1,
        min_face_detection_confidence=0.5)
    return mp_vision.FaceLandmarker.create_from_options(opts), "tasks"


def get_mesh():
    """Lazily create the mesh object, or None if mediapipe missing."""
    global _mesh, _mesh_backend
    if _mesh is not None:
        return _mesh
    try:
        import mediapipe as mp  # noqa: F401
    except ImportError:
        _mesh_backend = "none"
        return None
    # Prefer the classic solutions API when present (no bundle download).
    try:
        import mediapipe as _mp

        if hasattr(_mp, "solutions") and hasattr(_mp.solutions, "face_mesh"):
            _mesh, _mesh_backend = _make_solutions_mesh()
            return _mesh
    except Exception:
        pass
    try:
        _mesh, _mesh_backend = _make_tasks_landmarker()
        return _mesh
    except Exception:
        _mesh, _mesh_backend = None, "none"
        return None


def extract_landmarks(bgr_crop):
    """Return list of (x, y) normalized landmarks, or None if no face/none."""
    landmarks, _backend = extract_landmarks_with_backend(bgr_crop)
    return landmarks


def extract_landmarks_with_backend(bgr_crop):
    """Return (landmarks | None, backend: 'tasks' | 'solutions' | 'none').

    Reports the real landmark count from the installed config; the caller
    compares against EXPECTED_LANDMARK_COUNT instead of us asserting.
    """
    mesh = get_mesh()
    if mesh is None or bgr_crop is None or getattr(bgr_crop, "size", 0) == 0:
        return None, "none"
    import cv2

    import mediapipe as mp

    backend = _mesh_backend
    try:
        from mediapipe.tasks.python import vision as mp_vision

        if backend == "tasks" or isinstance(mesh, mp_vision.FaceLandmarker):
            rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = mesh.detect(mp_image)
            if not res.face_landmarks:
                return None, "tasks"
            return [(lm.x, lm.y) for lm in res.face_landmarks[0]], "tasks"
    except Exception:
        pass
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    res = mesh.process(rgb)
    if not getattr(res, "multi_face_landmarks", None):
        return None, "none" if backend == "none" else backend
    pts = [(lm.x, lm.y) for lm in res.multi_face_landmarks[0].landmark]
    return pts, (backend if backend != "none" else "solutions")


def close():
    global _mesh, _mesh_backend
    try:
        if _mesh is not None:
            try:
                _mesh.close()
            except Exception:
                pass
    finally:
        _mesh = None
        _mesh_backend = "none"
