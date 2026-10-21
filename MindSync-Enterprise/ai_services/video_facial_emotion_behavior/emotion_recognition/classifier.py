"""Emotion classifier interface (honest stub — no fake model).

Real CNN goes through `predict_probs(face_crop_bgr)` -> 7 probabilities in
EMOTIONS order. Until a trained FER2013 model file is wired in, calling it
raises ModelNotAvailableError instead of inventing numbers.
"""
from __future__ import annotations

import math

from .labels import EMOTIONS, NUM_EMOTIONS


class ModelNotAvailableError(RuntimeError):
    pass


_model = None
_model_info: dict = {"loaded": False, "path": None}


def model_status() -> dict:
    """Report whether a real CNN is loaded (never claim otherwise)."""
    return dict(_model_info)


def load_model(path: str | None = None) -> dict:
    """Load the trained FER2013 CNN via predict.py. Stub reports missing."""
    try:
        from .predict import load_model as _real_load
        return _real_load(path)
    except Exception:
        _model_info.update(loaded=False, path=path)
        return model_status()


def validate_probs(probs) -> tuple[bool, str]:
    """Check 7 finite values in [0,1] summing to ~1. Returns (ok, msg)."""
    try:
        vals = [float(v) for v in probs]
    except Exception:
        return False, "Probabilities must be 7 numbers."
    if len(vals) != NUM_EMOTIONS:
        return False, f"Expected {NUM_EMOTIONS} probabilities, got {len(vals)}."
    if any(not math.isfinite(v) or v < -1e-6 or v > 1 + 1e-6 for v in vals):
        return False, "Probabilities must be finite values in [0, 1]."
    if abs(sum(vals) - 1.0) > 0.02:
        return False, f"Probabilities sum to {sum(vals):.3f}, expected ~1.0."
    return True, "OK"


def predict_probs(face_crop_bgr):
    """Frame-level inference via predict.py. Raises until a model is wired."""
    from .predict import predict_probs as _real_predict
    return _real_predict(face_crop_bgr)


def aggregate_frame_probs(list_of_probs):
    """Mean of valid-frame distributions. Skips invalid rows.

    Raises ValueError when no valid frames (caller must NOT use zeros).
    """
    valid = [list(map(float, p)) for p in (list_of_probs or [])
             if validate_probs(p)[0]]
    if not valid:
        raise ValueError("No valid frame predictions to aggregate.")
    n = len(valid)
    mean = [sum(col) / n for col in zip(*valid)]
    s = sum(mean)
    return [v / s for v in mean] if s > 0 else mean
