"""In-memory frame sampling (no frame folders, no DB writes).

Sampling strategy (documented choice):
- Target ~2 fps (FRAME_SAMPLE_FPS) so a 30 s clip yields ~60 frames.
  Why 2 fps: facial emotion changes slowly vs 30 fps video; 60 crops keep
  Haar + CNN cheap on a laptop CPU while still covering the whole clip.
  Adjust via app.config.FRAME_SAMPLE_FPS after testing.
- Step = round(src_fps / sample_fps), min 1. If source fps < target
  (e.g. 10 fps webcam), step=1 takes every frame — never duplicates.
- Invalid fps (0 / NaN / Inf / negative) falls back to 30.0.
- Only valid (non-None, non-empty) frames are yielded.
- Generator yields one frame at a time; never loads whole video into RAM.
- Privacy: reads the ORIGINAL file path directly, never copies/deletes it.
  VideoCapture is released in `finally`, also on cancel/failure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional


@dataclass
class SampledFrame:
    """One kept frame + its position in the source video."""

    index: int            # source frame index (0-based, presentation order)
    timestamp_s: float    # index / effective source fps
    frame: object         # BGR numpy array (kept in memory only)


@dataclass
class PreprocessResult:
    """Interface for the NEXT stage (face -> MediaPipe -> CNN)."""

    ok: bool
    props_path: str
    width: int = 0
    height: int = 0
    src_fps: float = 0.0
    duration_s: float = 0.0
    sampled_count: int = 0      # frames yielded (valid only)
    dropped_count: int = 0      # unreadable/empty frames skipped
    scanned_count: int = 0      # total read attempts
    frames: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def message(self) -> str:
        return "; ".join(self.errors) if self.errors else (
            f"Preprocessed {self.sampled_count} valid frames "
            f"({self.scanned_count} scanned, {self.dropped_count} dropped).")


def _effective_fps(raw_fps: float) -> float:
    """Sanitise header FPS; fall back to 30.0 when missing/invalid."""
    try:
        f = float(raw_fps)
    except (TypeError, ValueError):
        return 30.0
    if not math.isfinite(f) or f <= 1e-6 or f > 240.0:
        return 30.0
    return f


def _compute_step(src_fps: float, sample_fps: float) -> int:
    """Frames to skip between kept frames. Min 1 (never duplicate)."""
    target = sample_fps if sample_fps and sample_fps > 0 else 2.0
    return max(1, int(round(src_fps / max(target, 0.1))))


def iter_sampled_frames(
    path: str,
    *,
    sample_fps: float = 2.0,
    max_frames: int = 120,
    is_cancelled: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Iterator[SampledFrame]:
    """Yield valid sampled frames in presentation order (memory-efficient)."""
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return
        raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        src_fps = _effective_fps(raw_fps)
        step = _compute_step(src_fps, sample_fps)
        idx, kept = 0, 0
        while kept < max_frames:
            if is_cancelled is not None and is_cancelled():
                break
            ok, frame = cap.read()
            if not ok or frame is None or getattr(frame, "size", 0) == 0:
                break  # EOF; corrupt-frame counting lives in preprocess_video
            if idx % step == 0:
                ts = idx / src_fps if src_fps > 0 else 0.0
                kept += 1
                yield SampledFrame(index=idx, timestamp_s=ts, frame=frame)
                if on_progress is not None:
                    try:
                        on_progress(kept, max_frames)
                    except Exception:
                        pass
            idx += 1
    finally:
        try:
            cap.release()
        except Exception:
            pass


def sample_frames(path: str, *, sample_fps: float = 2.0, max_frames: int = 120):
    """Back-compat generator yielding raw BGR arrays (used by validators)."""
    for sf in iter_sampled_frames(
            path, sample_fps=sample_fps, max_frames=max_frames):
        yield sf.frame


def preprocess_video(
    path: str,
    *,
    sample_fps: float = 2.0,
    max_frames: int = 120,
    min_valid_frames: int = 10,
    is_cancelled: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> PreprocessResult:
    """Shared Step-2 pipeline: probe -> sample -> validate counts.

    Used by BOTH upload and webcam recordings (same function, same rules).
    Never writes frames to disk/DB; keeps a capped list in memory only.
    Never modifies/deletes the original file.
    """
    import os

    import cv2

    if not path or not os.path.exists(path):
        return PreprocessResult(
            ok=False, props_path=path or "",
            errors=[f"File not found: {path}"])
    cap = cv2.VideoCapture(path)
    width = height = 0
    try:
        if not cap.isOpened():
            return PreprocessResult(
                ok=False, props_path=path,
                errors=["OpenCV could not open/read this video "
                        "(unsupported codec or corrupt file)."])
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        src_fps = _effective_fps(raw_fps)
        duration = (count / src_fps) if count > 0 and src_fps > 0 else 0.0
    finally:
        try:
            cap.release()
        except Exception:
            pass

    step = _compute_step(src_fps, sample_fps)
    frames: list = []
    dropped = 0
    scanned = 0
    cap2 = cv2.VideoCapture(path)
    try:
        if not cap2.isOpened():
            return PreprocessResult(
                ok=False, props_path=path, width=width, height=height,
                src_fps=src_fps, duration_s=duration,
                errors=["OpenCV could not open/read this video on rescan."])
        idx = 0
        while len(frames) < max_frames:
            if is_cancelled is not None and is_cancelled():
                return PreprocessResult(
                    ok=False, props_path=path, width=width, height=height,
                    src_fps=src_fps, duration_s=duration,
                    sampled_count=len(frames), dropped_count=dropped,
                    scanned_count=scanned, frames=frames,
                    errors=["Preprocessing cancelled by user."])
            ok, frame = cap2.read()
            if not ok or frame is None or getattr(frame, "size", 0) == 0:
                break  # EOF (mid-stream corrupt frames surface as short scan)
            scanned += 1
            if idx % step == 0:
                frames.append(SampledFrame(
                    index=idx, timestamp_s=idx / src_fps, frame=frame))
                if on_progress is not None:
                    try:
                        on_progress(len(frames), max_frames)
                    except Exception:
                        pass
            idx += 1
    finally:
        try:
            cap2.release()
        except Exception:
            pass

    if not frames:
        return PreprocessResult(
            ok=False, props_path=path, width=width, height=height,
            src_fps=src_fps, duration_s=duration,
            sampled_count=0, dropped_count=dropped, scanned_count=scanned,
            frames=[],
            errors=["No decodable frames found — video may be corrupt "
                    "or use an unsupported codec."])
    if len(frames) < min_valid_frames:
        return PreprocessResult(
            ok=False, props_path=path, width=width, height=height,
            src_fps=src_fps, duration_s=duration,
            sampled_count=len(frames), dropped_count=dropped,
            scanned_count=scanned, frames=frames,
            errors=[f"Insufficient frames: {len(frames)} valid "
                    f"(need >= {min_valid_frames}). Try a longer clip."])
    return PreprocessResult(
        ok=True, props_path=path, width=width, height=height,
        src_fps=src_fps, duration_s=duration,
        sampled_count=len(frames), dropped_count=dropped,
        scanned_count=scanned, frames=frames, errors=[])


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
