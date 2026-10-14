"""Video probing + validation (Qt-free, testable).

Reads the ORIGINAL file directly (never copies uploads into the DB).
Uses OpenCV only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class VideoProperties:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    readable: bool


@dataclass
class ValidationResult:
    ok: bool
    errors: list = field(default_factory=list)
    props: VideoProperties | None = None

    def message(self) -> str:
        return "; ".join(self.errors) if self.errors else "Video is valid."


def probe_video(path: str) -> VideoProperties:
    """Open with OpenCV and read basic properties."""
    import cv2

    if not path or not os.path.exists(path):
        return VideoProperties(path or "", 0, 0, 0.0, 0, 0.0, False)
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return VideoProperties(path, 0, 0, 0.0, 0, 0.0, False)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = (count / fps) if fps > 0 and count > 0 else 0.0
        ok, frame = cap.read()
        readable = bool(ok) and frame is not None
        return VideoProperties(path, width, height, fps, count, duration, readable)
    finally:
        cap.release()


def validate_video_file(
    path: str,
    *,
    allowed_extensions=(".mp4", ".avi", ".mov"),
    min_duration_s: float = 25.0,
    max_duration_s: float = 35.0,
    duration_override_s: float | None = None,
) -> ValidationResult:
    """Validate an upload/recording WITHOUT starting AI analysis.

    duration_override_s: trusted wall-clock seconds for a fresh webcam take.
    Webcam mp4 headers can under-report (e.g. stamps 22s for a 30s capture
    when capture ran slower than the header FPS), so the recorder passes its
    measured wall time. Uploads pass None -> header duration is used.
    """
    errors: list[str] = []
    if not path:
        return ValidationResult(False, ["No video selected."], None)
    if not os.path.exists(path):
        return ValidationResult(False, [f"File not found: {path}"], None)
    ext = os.path.splitext(path)[1].lower()
    if ext not in tuple(e.lower() for e in allowed_extensions):
        errors.append(
            f"Unsupported format '{ext}'. Allowed: {', '.join(allowed_extensions)} "
            "(actual playback also depends on installed decoders)."
        )
    props = probe_video(path)
    if not props.readable:
        errors.append("OpenCV could not open/read this video (unsupported codec or corrupt file).")
        return ValidationResult(False, errors, props)
    if props.width <= 0 or props.height <= 0:
        errors.append("Could not read video dimensions (corrupt header?).")
    hard_errors = list(errors)
    # Prefer recorder wall-clock when provided (fresh webcam take only).
    eff_duration = (
        float(duration_override_s)
        if duration_override_s is not None and duration_override_s > 0
        else props.duration_s
    )
    if eff_duration > 0:
        if eff_duration < min_duration_s:
            hard_errors.append(
                f"Video too short: {eff_duration:.1f}s < minimum {min_duration_s:.0f}s "
                "(target 30s ± tolerance)."
            )
        elif eff_duration > max_duration_s:
            hard_errors.append(
                f"Video too long: {eff_duration:.1f}s > maximum {max_duration_s:.0f}s "
                "(target 30s ± tolerance)."
            )
    else:
        errors.append("Warning: duration unknown (missing FPS/metadata); continuing anyway.")
        return ValidationResult(len(hard_errors) == 0, errors, props)
    return ValidationResult(len(hard_errors) == 0, hard_errors if hard_errors else errors, props)


