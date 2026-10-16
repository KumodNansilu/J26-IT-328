"""Step-3 face + landmark stage (Qt-free, reusable, testable).

Consumes Step-2 SampledFrames, produces FaceFrame records for the future CNN:
frame -> Haar detect -> largest box -> padded BGR crop (in-memory) ->
MediaPipe mesh on the crop -> landmark count + landmarks (in-memory).

Rules:
- One faceless frame never fails the video; only the configurable minimum
  usable count / coverage can fail it.
- Crops are clamped at image borders (faces at the edge are safe).
- Landmark failure never fails the frame: the crop is still usable by CNN.
- Landmarks are NEVER fed into the CNN here; they ride alongside for
  separate analysis. No CNN/VBS/DSI in this module.
- Nothing is written to disk or DB. MediaPipe handle is closed in `finally`,
  also on cancel/failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional


@dataclass
class FaceFrame:
    """One usable frame's outputs for the future CNN stage (memory only)."""

    index: int                 # source frame index from Step-2
    timestamp_s: float         # seconds into the video
    box: tuple                 # (x, y, w, h) largest face, clamped ints
    crop: object               # padded BGR numpy crop (memory only)
    landmarks: object = None   # list[(x, y)] normalized or None (mesh failed)
    landmark_count: int = 0    # len(landmarks) or 0
    mesh_used: str = "none"    # "tasks" | "solutions" | "none"


@dataclass
class FaceStageResult:
    """Documented output for the CNN stage (Step-4 reads this)."""

    ok: bool
    total: int = 0             # sampled frames processed
    with_face: int = 0         # frames with a detected face
    without_face: int = 0      # total - with_face
    usable: list = field(default_factory=list)  # FaceFrame list (memory only)
    coverage: float = 0.0      # with_face / total
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def message(self) -> str:
        if self.errors:
            return "; ".join(self.errors)
        base = (f"Face found in {self.with_face}/{self.total} frames "
                f"({self.coverage:.0%}).")
        if self.warnings:
            return base + " Warnings: " + "; ".join(self.warnings)
        return base

    def clear(self):
        """Release numpy/landmark refs (privacy: call after CNN consumes)."""
        try:
            self.usable.clear()
        except Exception:
            pass


def _largest_box(boxes) -> tuple | None:
    """Pick the largest face box (closest person). None if empty."""
    if not boxes:
        return None
    try:
        return max(boxes, key=lambda b: int(b[2]) * int(b[3]))
    except Exception:
        return boxes[0]


def crop_face(frame, box, *, padding: float = 0.1):
    """Crop padded face region, clamped to borders. Memory only.

    Returns None for invalid input/degenerate crop (never raises).
    """
    try:
        if frame is None or getattr(frame, "size", 0) == 0 or box is None:
            return None
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return None
        x, y, bw, bh = (int(v) for v in box)
        if bw <= 0 or bh <= 0:
            return None
        pad_w, pad_h = int(bw * padding), int(bh * padding)
        x0, y0 = max(0, x - pad_w), max(0, y - pad_h)
        x1, y1 = min(w, x + bw + pad_w), min(h, y + bh + pad_h)
        if x1 <= x0 or y1 <= y0:
            return None
        crop = frame[y0:y1, x0:x1]
        return crop if getattr(crop, "size", 0) > 0 else None
    except Exception:
        return None


def run_face_stage(
    sampled_frames: Iterable,
    *,
    min_size: int = 60,
    padding: float = 0.1,
    min_usable: int = 10,
    min_coverage: float = 0.3,
    expected_landmarks: int = 468,
    is_cancelled: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> FaceStageResult:
    """Run Step-3 over Step-2 frames. See module docstring for contract."""
    from ai_services.video_facial_emotion_behavior.face_detection.faces import (
        detect_faces)
    from ai_services.video_facial_emotion_behavior.landmarks.mesh import (
        close as mesh_close)
    from ai_services.video_facial_emotion_behavior.landmarks.mesh import (
        extract_landmarks_with_backend)

    frames = list(sampled_frames or [])
    total = len(frames)
    if total == 0:
        return FaceStageResult(ok=False, errors=["No sampled frames."])
    usable: list = []
    with_face = 0
    counts: list = []
    mesh_failed = False
    try:
        for i, sf in enumerate(frames):
            if is_cancelled is not None and is_cancelled():
                return FaceStageResult(
                    ok=False, total=total, with_face=with_face,
                    without_face=total - with_face, usable=usable,
                    coverage=(with_face / total if total else 0.0),
                    errors=["Face stage cancelled by user."])
            frame = getattr(sf, "frame", sf)
            try:
                boxes = detect_faces(frame, min_size=min_size)
            except Exception:
                boxes = []
            if not boxes:
                if on_progress is not None:
                    try:
                        on_progress(i + 1, total)
                    except Exception:
                        pass
                continue
            with_face += 1
            box = _largest_box(boxes)
            crop = crop_face(frame, box, padding=padding)
            if crop is None:
                with_face -= 1
                if on_progress is not None:
                    try:
                        on_progress(i + 1, total)
                    except Exception:
                        pass
                continue
            try:
                landmarks, backend = extract_landmarks_with_backend(crop)
            except Exception:
                landmarks, backend = None, "none"
            count = len(landmarks) if landmarks else 0
            if backend == "none":
                mesh_failed = True
            if count:
                counts.append(count)
            usable.append(FaceFrame(
                index=int(getattr(sf, "index", i)),
                timestamp_s=float(getattr(sf, "timestamp_s", 0.0)),
                box=tuple(int(v) for v in box),
                crop=crop, landmarks=landmarks,
                landmark_count=count, mesh_used=backend))
            if on_progress is not None:
                try:
                    on_progress(i + 1, total)
                except Exception:
                    pass
    finally:
        try:
            mesh_close()
        except Exception:
            pass

    coverage = (with_face / total) if total else 0.0
    warnings: list = []
    if mesh_failed and usable:
        warnings.append("MediaPipe unavailable/failed on some frames — "
                        "crops kept for CNN anyway.")
    if counts and any(c != expected_landmarks for c in counts):
        warnings.append(
            f"Landmark count differs from expected {expected_landmarks} "
            f"(saw {sorted(set(counts))}) — check MediaPipe config.")
    if with_face < min_usable:
        return FaceStageResult(
            ok=False, total=total, with_face=with_face,
            without_face=total - with_face, usable=usable, coverage=coverage,
            errors=[f"Too few usable frames: {with_face}/{total} with a face "
                    f"(need >= {min_usable})."],
            warnings=warnings)
    if coverage < min_coverage:
        return FaceStageResult(
            ok=False, total=total, with_face=with_face,
            without_face=total - with_face, usable=usable, coverage=coverage,
            errors=[f"Face in only {coverage:.0%} of frames "
                    f"(need >= {min_coverage:.0%})."],
            warnings=warnings)
    return FaceStageResult(
        ok=True, total=total, with_face=with_face,
        without_face=total - with_face, usable=usable, coverage=coverage,
        errors=[], warnings=warnings)

