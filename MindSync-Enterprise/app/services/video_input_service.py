"""Shared VFEBA input-stage validation service (Qt-free).

Used by BOTH upload and webcam flows so they enter the SAME pipeline:
validate file -> sample frames -> face coverage. Full CNN/VBS stages come
later; this module honestly reports what it checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.config import (FRAME_SAMPLE_FPS, FRAME_SAMPLE_MAX_FRAMES,
                        MIN_FACE_COVERAGE, MIN_USABLE_FRAMES, MIN_VALID_FRAMES,
                        SUPPORTED_VIDEO_EXTENSIONS, VIDEO_MAX_DURATION_S,
                        VIDEO_MIN_DURATION_S)


@dataclass
class CheckResult:
    ok: bool
    stage: str
    message: str
    details: dict = field(default_factory=dict)


def validate_video_input(path: str, duration_override_s: float | None = None,
                         is_cancelled=None,
                         on_progress=None) -> CheckResult:
    """Full input validation: file -> duration -> frames -> face coverage.

    Step-2 shared pipeline: BOTH upload and webcam recordings call this
    same function. Frames come from preprocessing.preprocess_video()
    (in-memory, capped, never on disk/DB). Face check reuses existing
    faces.face_coverage (no duplication). Numpy refs are cleared before
    return so temp memory is freed promptly.
    """
    from ai_services.video_facial_emotion_behavior.preprocessing import \
        video_loader
    from ai_services.video_facial_emotion_behavior.preprocessing.frames import \
        preprocess_video

    vr = video_loader.validate_video_file(
        path, allowed_extensions=SUPPORTED_VIDEO_EXTENSIONS,
        min_duration_s=VIDEO_MIN_DURATION_S, max_duration_s=VIDEO_MAX_DURATION_S,
        duration_override_s=duration_override_s)
    if not vr.ok:
        return CheckResult(False, "file", vr.message(),
                           {"props": vars(vr.props) if vr.props else {}})
    pre = preprocess_video(
        path, sample_fps=FRAME_SAMPLE_FPS,
        max_frames=FRAME_SAMPLE_MAX_FRAMES,
        min_valid_frames=MIN_VALID_FRAMES,
        is_cancelled=is_cancelled, on_progress=on_progress)
    if not pre.ok:
        pre.frames.clear()  # release any partial numpy refs
        stage = "cancelled" if "cancelled" in pre.message().lower() else "frames"
        return CheckResult(False, stage, pre.message(),
                           {"sampled": pre.sampled_count,
                            "scanned": pre.scanned_count,
                            "dropped": pre.dropped_count})
    # Step-3: shared face + landmark stage (same for upload + webcam).
    # Memory only: crops/landmarks are cleared before return.
    from app.config import (EXPECTED_LANDMARK_COUNT, FACE_CROP_PADDING,
                            FACE_MIN_SIZE_PX)
    from ai_services.video_facial_emotion_behavior.landmarks import (
        run_face_stage)
    face_res = run_face_stage(
        pre.frames, min_size=FACE_MIN_SIZE_PX, padding=FACE_CROP_PADDING,
        min_usable=MIN_USABLE_FRAMES, min_coverage=MIN_FACE_COVERAGE,
        expected_landmarks=EXPECTED_LANDMARK_COUNT,
        is_cancelled=is_cancelled)
    details = {"usable": face_res.with_face, "total": face_res.total,
               "ratio": face_res.coverage,
               "with_landmarks": sum(1 for f in face_res.usable
                                     if f.landmark_count > 0)}
    msg = face_res.message()
    try:
        face_res.clear()  # privacy: drop crops/landmarks right after check
        pre.frames.clear()
    finally:
        pass
    if not face_res.ok:
        stage = "cancelled" if "cancel" in msg.lower() else "face"
        return CheckResult(False, stage, msg, details)
    return CheckResult(True, "face", msg, details)
