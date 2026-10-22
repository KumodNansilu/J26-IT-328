"""Video probing + validation (Qt-free, testable).

Reads the ORIGINAL file directly (never copies uploads into the DB).
Uses OpenCV only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from app.config import VIDEO_MAX_DURATION_S, VIDEO_MIN_DURATION_S


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
    min_duration_s: float = VIDEO_MIN_DURATION_S,
    max_duration_s: float = VIDEO_MAX_DURATION_S,
    duration_override_s: float | None = None,
) -> ValidationResult:
    """Validate an upload/recording WITHOUT starting AI analysis.

    duration_override_s: measured wall-clock seconds for a fresh webcam take.
    The file must still be readable and contain frames; the override only
    compensates for unreliable webcam FPS metadata. Uploads pass None.
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
    if props.frame_count <= 0:
        errors.append("Video contains no frames.")
    if props.fps <= 0:
        errors.append("Video FPS metadata is missing or invalid.")
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
                f"(target {min_duration_s + (max_duration_s - min_duration_s) / 2:.0f}s ± "
                f"{(max_duration_s - min_duration_s) / 2:.0f}s)."
            )
        elif eff_duration > max_duration_s:
            hard_errors.append(
                f"Video too long: {eff_duration:.1f}s > maximum {max_duration_s:.0f}s "
                f"(target {min_duration_s + (max_duration_s - min_duration_s) / 2:.0f}s ± "
                f"{(max_duration_s - min_duration_s) / 2:.0f}s)."
            )
    else:
        hard_errors.append("Video duration could not be measured from its metadata.")
        return ValidationResult(False, hard_errors, props)
    return ValidationResult(len(hard_errors) == 0, hard_errors if hard_errors else errors, props)
