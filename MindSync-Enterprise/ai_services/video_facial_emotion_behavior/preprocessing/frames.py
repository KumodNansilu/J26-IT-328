"""In-memory frame sampling (no frame folders, no DB writes)."""
from __future__ import annotations


def sample_frames(path: str, *, sample_fps: float = 2.0, max_frames: int = 120):
    """Yield sampled BGR frames from a video file (presentation order)."""
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        step = max(1, int(round(src_fps / max(sample_fps, 0.1))))
        idx, kept = 0, 0
        while kept < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                break
            if idx % step == 0:
                kept += 1
                yield frame
            idx += 1
    finally:
        cap.release()


def count_readable_frames(path: str, *, probe_limit: int = 10) -> int:
    """Quick readability probe: count up to probe_limit decodable frames."""
    import cv2

    cap = cv2.VideoCapture(path)
    n = 0
    try:
        if not cap.isOpened():
            return 0
        for _ in range(probe_limit):
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                break
            n += 1
        return n
    finally:
        cap.release()
