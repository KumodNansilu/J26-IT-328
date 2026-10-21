# -*- coding: utf-8 -*-
"""Frame-level emotion inference: FaceFrame.crop -> 7 probs, EMOTIONS order.

Loads models/fer2013_cnn_best.pt (+ meta) when present; otherwise raises
ModelNotAvailableError (never invents numbers). Softmax applied here so
the model file stores raw logits. Memory only: no disk/DB writes, no
employee imagery in logs.
"""
from __future__ import annotations

import json
import os

from .classifier import (
    ModelNotAvailableError, _model, _model_info, model_status,
    validate_probs)
from .cnn_model import MODEL_INPUT_SIZE, build_model, preprocess_crop
from .labels import DEFAULT_CKPT, DEFAULT_META, EMOTIONS, NUM_EMOTIONS


DEFAULT_CKPT = "models/fer2013_cnn_best.pt"
DEFAULT_META = "models/fer2013_cnn_meta.json"


def _read_meta(meta_path: str):
    # Fail-closed: missing / malformed / incompatible metadata -> clear error.
    if not os.path.exists(meta_path):
        raise ModelNotAvailableError(
            "Metadata file missing for %s: %s. Run training first."
            % (DEFAULT_CKPT, meta_path))
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception as exc:
        raise ModelNotAvailableError(
            "Metadata file is not valid JSON (%s): %s." % (meta_path, exc)
        ) from exc
    if not isinstance(meta, dict):
        raise ModelNotAvailableError(
            "Metadata file is not a JSON object (%s)." % meta_path)

    order = meta.get("label_order")
    expected_input = [1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]
    if order is None:
        raise ModelNotAvailableError(
            "Metadata file has no 'label_order' key (%s)." % meta_path)
    if list(order) != list(EMOTIONS):
        raise ModelNotAvailableError(
            "Checkpoint label order %s != labels.py %s. "
            "Retrain so metadata and labels.py agree."
            % (order, list(EMOTIONS)))
    if meta.get("input") != expected_input:
        raise ModelNotAvailableError(
            "Checkpoint input %s != expected %s. "
            "Retrain so metadata and model architecture agree."
            % (meta.get("input"), expected_input))
    return meta


def load_model(path: str | None = None) -> dict:
    """Load checkpoint + verify metadata. Raises ModelNotAvailableError if
    the weights are absent or the metadata is missing / malformed /
    incompatible with this project's labels and input shape."""
    global _model
    ckpt = path or DEFAULT_CKPT
    if not os.path.exists(ckpt):
        _model_info.update(loaded=False, path=ckpt)
        raise ModelNotAvailableError(
            "No trained CNN at %s - run smoke/full training first "
            "(stages 5+ need your approval)." % ckpt)

    meta_path = os.path.join(
        os.path.dirname(ckpt) or ".", os.path.basename(DEFAULT_META))
    # Validate each piece; an unexpected failure is surfaced clearly
    # rather than silently ignored.
    try:
        _read_meta(meta_path)
    except ModelNotAvailableError:
        raise
    except Exception as exc:
        raise ModelNotAvailableError(
                "Metadata validation failed for %s: %s."
                % (meta_path, exc)) from exc

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(num_classes=NUM_EMOTIONS)
    state = torch.load(ckpt, map_location=device)
    # Accept {"state_dict": ...} or raw state dict.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    _model = model
    _model_info.update(loaded=True, path=ckpt, device=device)
    return model_status()


def predict_probs(face_crop_bgr):
    """Return 7 finite probabilities in EMOTIONS order for one BGR crop."""
    import torch

    if _model is None:
        raise ModelNotAvailableError(
            "No trained CNN loaded - call load_model() after training.")
    x = preprocess_crop(face_crop_bgr)
    if not isinstance(x, torch.Tensor):
        import numpy as np

        x = torch.from_numpy(np.asarray(x, dtype="float32"))
    x = x.unsqueeze(0).to(next(_model.parameters()).device)  # (1,1,48,48)
    with torch.no_grad():
        logits = _model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
    ok, msg = validate_probs(probs)
    if not ok:
        raise RuntimeError("CNN produced invalid probabilities: %s" % msg)
    return [float(v) for v in probs]
