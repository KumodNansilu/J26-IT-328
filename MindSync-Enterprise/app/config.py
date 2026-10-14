"""App-wide configuration (no Qt / no OpenCV imports here).

Simple constants so thresholds are documented in one place.
"""
from __future__ import annotations

# ---- Video check-in (VFEBA input stage) ----
VIDEO_TARGET_DURATION_S: float = 30.0
# Allowed tolerance around the 30 s target. Documented choice:
# ±5 s keeps check-ins comparable while allowing webcam timing jitter
# and slightly short/long uploads without arbitrary rejection.
VIDEO_DURATION_TOLERANCE_S: float = 5.0
VIDEO_MIN_DURATION_S: float = VIDEO_TARGET_DURATION_S - VIDEO_DURATION_TOLERANCE_S
VIDEO_MAX_DURATION_S: float = VIDEO_TARGET_DURATION_S + VIDEO_DURATION_TOLERANCE_S

# File picker filter + validation allow-list (decoder support still
# depends on the local OpenCV/FFmpeg build; we probe with cv2 anyway).
SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")
VIDEO_PICKER_FILTER = "Videos (*.mp4 *.avi *.mov)"

# Frame sampling: process ~2 fps to keep a 30 s clip cheap
# (30 s * 2 = ~60 frames) while handling any source FPS.
FRAME_SAMPLE_FPS: float = 2.0
# Minimum usable (face-containing) frames needed for a trustworthy result.
MIN_USABLE_FRAMES: int = 10
# Minimum share of sampled frames that must contain a face (0..1).
# We do NOT reject a video because of one faceless frame.
MIN_FACE_COVERAGE: float = 0.3

# Webcam recording
WEBCAM_DEVICE_INDEX: int = 0
WEBCAM_RECORD_FPS: float = 20.0
WEBCAM_FRAME_WIDTH: int = 640
WEBCAM_FRAME_HEIGHT: int = 480
# mp4v is the most widely available mp4 codec on Windows OpenCV builds.
WEBCAM_FOURCC: str = "mp4v"

