
"""XLSR-Wav2Vec2 model loading and speech embedding extraction."""

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

XLSR_MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
SAMPLE_RATE = 16000
DEVICE = torch.device("cpu")

_feature_extractor = None
_xlsr_model = None


def load_xlsr_model():
    """Load XLSR once and reuse it for subsequent predictions."""

    global _feature_extractor, _xlsr_model

    if _feature_extractor is None or _xlsr_model is None:
        _feature_extractor = (
            Wav2Vec2FeatureExtractor.from_pretrained(
                XLSR_MODEL_NAME
            )
        )

        _xlsr_model = Wav2Vec2Model.from_pretrained(
            XLSR_MODEL_NAME
        ).to(DEVICE)

        _xlsr_model.eval()

    return _feature_extractor, _xlsr_model


def extract_xlsr_embedding(audio: np.ndarray) -> np.ndarray:
    """Extract a 1024-dimensional mean-pooled XLSR embedding."""

    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Audio must be a non-empty mono signal.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid numeric values.")

    feature_extractor, model = load_xlsr_model()

    inputs = feature_extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=False,
    )

    input_values = inputs.input_values.to(DEVICE)

    with torch.no_grad():
        outputs = model(input_values)

        embedding = (
            outputs.last_hidden_state
            .mean(dim=1)
            .squeeze(0)
            .cpu()
            .numpy()
        )

    embedding = np.asarray(embedding, dtype=np.float32)

    if embedding.shape != (1024,):
        raise ValueError(
            f"Expected embedding shape (1024,), got {embedding.shape}"
        )

    if not np.all(np.isfinite(embedding)):
        raise ValueError("XLSR embedding contains invalid values.")

    return embedding

