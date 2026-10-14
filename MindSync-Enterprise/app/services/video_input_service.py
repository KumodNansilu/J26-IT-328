"""Shared VFEBA input-stage validation service (Qt-free).

Used by BOTH upload and webcam flows so they enter the SAME pipeline:
validate file -> sample frames -> face coverage. Full CNN/VBS stages come
later; this module honestly reports what it checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.config import (FRAME_SAMPLE_FPS, MIN_FACE_COVERAGE, MIN_USABLE_FRAMES,
                        SUPPORTED_VIDEO_EXTENSIONS, VIDEO_MAX_DURATION_S,
                        VIDEO_MIN_DURATION_S)


@dataclass
class CheckResult:
    ok: bool
    stage: str
    message: str
    details: dict = field(default_factory=dict)


def validate_video_input(path: str, duration_override_s: float | None = None) -> CheckResult:
    """Full input validation: file -> duration -> frames -> face coverage."""
    from ai_services.video_facial_emotion_behavior.preprocessing import \
        video_loader
    from ai_services.video_facial_emotion_behavior.preprocessing.frames import \
        sample_frames
    from ai_services.video_facial_emotion_behavior.face_detection.faces import (
        check_face_sufficiency, face_coverage)

    vr = video_loader.validate_video_file(
        path, allowed_extensions=SUPPORTED_VIDEO_EXTENSIONS,
        min_duration_s=VIDEO_MIN_DURATION_S, max_duration_s=VIDEO_MAX_DURATION_S,
        duration_override_s=duration_override_s)
    if not vr.ok:
        return CheckResult(False, "file", vr.message(),
                           {"props": vars(vr.props) if vr.props else {}})
    frames = list(sample_frames(path, sample_fps=FRAME_SAMPLE_FPS))
    if not frames:
        return CheckResult(False, "frames", "No decodable frames found.", {})
    usable, total, ratio = face_coverage(frames)
    ok, msg = check_face_sufficiency(
        usable, total, ratio,
        min_usable=MIN_USABLE_FRAMES, min_coverage=MIN_FACE_COVERAGE)
    return CheckResult(ok, "face" if ok else "face", msg,
                       {"usable": usable, "total": total, "ratio": ratio})
