"""Face detection with OpenCV Haar cascade (ships with opencv-python).

Missing faces are reported, NEVER silently mapped to 'Neutral'.
"""
from __future__ import annotations

from typing import Iterable

_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        import os
        import cv2

        # opencv-python 5.x wheels no longer bundle the Haar XML next to
        # cv2.data. Try in order: bundled path -> local project copy
        # (MindSync-Enterprise/models/haarcascade). First that loads wins.
        # NOTE: skip empty/missing candidates — CascadeClassifier("") logs
        # scary C++ errors and returns empty, which is a success-path trap.
        candidates = []
        try:
            data_dir = getattr(getattr(cv2, "data", None),
                               "haarcascades", "")
            if data_dir:
                candidates.append(os.path.join(
                    data_dir, "haarcascade_frontalface_default.xml"))
        except Exception:
            pass
        try:
            from pathlib import Path

            # faces.py -> face_detection -> video_facial_emotion_behavior
            # -> ai_services -> MindSync-Enterprise (3 levels up).
            root = Path(__file__).resolve().parents[3]
            candidates.append(
                str(root / "models" / "haarcascade_frontalface_default.xml"))
        except Exception:
            pass
        # De-dup + drop empties / missing files before touching OpenCV.
        seen = set()
        usable = []
        for c in candidates:
            if not c or c in seen:
                continue
            seen.add(c)
            if os.path.exists(c):
                usable.append(c)
        last = ""
        for cascade_path in usable:
            try:
                det = cv2.CascadeClassifier(cascade_path)
                if not det.empty():
                    _detector = det
                    return _detector
                last = cascade_path
            except Exception:
                last = cascade_path
                continue
        raise RuntimeError(
            "Could not load Haar cascade. Tried: "
            + ("; ".join(usable) if usable else "(no candidate files found)")
            + ". Fix: place 'haarcascade_frontalface_default.xml' in "
              "MindSync-Enterprise/models/ (already done for this project)."
        )
    return _detector


def detect_faces(frame, *, min_size: int = 60) -> list:
    """Detect faces in one BGR frame. Returns [(x, y, w, h), ...]."""
    import cv2

    if frame is None or getattr(frame, "size", 0) == 0:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _get_detector().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def crop_largest_face(frame, box):
    """Crop + clamp a face box. Returns BGR crop or None."""
    if box is None:
        return None
    h, w = frame.shape[:2]
    x, y, bw, bh = box
    x, y = max(0, x), max(0, y)
    crop = frame[y:min(h, y + bh), x:min(w, x + bw)]
    return crop if crop.size else None


def face_coverage(frames: Iterable, *, min_size: int = 60):
    """Return (frames_with_face, total_frames, ratio)."""
    total, usable = 0, 0
    for f in frames:
        total += 1
        if detect_faces(f, min_size=min_size):
            usable += 1
    return usable, total, (usable / total if total else 0.0)


def check_face_sufficiency(usable, total, ratio, *, min_usable=10, min_coverage=0.3):
    """Decide if enough face frames exist. Returns (ok, message)."""
    if total == 0:
        return False, "No readable frames — video may be corrupt."
    if usable < min_usable:
        return False, (f"Too few usable frames: {usable}/{total} contain a face "
                       f"(need >= {min_usable}). Try better lighting / face the camera.")
    if ratio < min_coverage:
        return False, (f"Face in only {ratio:.0%} of sampled frames "
                       f"(need >= {min_coverage:.0%}).")
    return True, f"Face found in {usable}/{total} sampled frames ({ratio:.0%})."
