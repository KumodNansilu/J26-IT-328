"""Step-4 glue: FaceFrame list -> per-frame 7-prob lists (memory only).

No VBS / trends / DSI here. Invalid frames skipped (never zero-filled);
empty result raises ValueError so callers cannot use fake zeros.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EmotionFrameResult:
    index: int
    timestamp_s: float
    probs: list  # 7 floats in EMOTIONS order


@dataclass
class EmotionStageResult:
    ok: bool
    frames: list = field(default_factory=list)
    skipped: int = 0
    errors: list = field(default_factory=list)

    def message(self) -> str:
        if self.errors:
            return "; ".join(self.errors)
        return (f"Emotion inference on {len(self.frames)} frames "
                f"({self.skipped} skipped).")


def run_emotion_stage(face_frames, *, is_cancelled=None) -> EmotionStageResult:
    """Run CNN over Step-3 FaceFrames. Memory only, nothing persisted."""
    from ..emotion_recognition.classifier import validate_probs
    from ..emotion_recognition.predict import predict_probs

    out: list = []
    skipped = 0
    frames = list(face_frames or [])
    if not frames:
        return EmotionStageResult(ok=False, errors=["No face frames given."])
    for ff in frames:
        if is_cancelled is not None and is_cancelled():
            return EmotionStageResult(ok=False, frames=out, skipped=skipped,
                                      errors=["Emotion stage cancelled."])
        try:
            probs = predict_probs(ff.crop)
        except Exception:
            skipped += 1
            continue
        if not validate_probs(probs)[0]:
            skipped += 1
            continue
        out.append(EmotionFrameResult(
            index=int(getattr(ff, "index", 0)),
            timestamp_s=float(getattr(ff, "timestamp_s", 0.0)),
            probs=[float(v) for v in probs]))
    if not out:
        return EmotionStageResult(ok=False, skipped=skipped,
                                  errors=["No valid frame predictions."])
    return EmotionStageResult(ok=True, frames=out, skipped=skipped,
                              errors=[])
