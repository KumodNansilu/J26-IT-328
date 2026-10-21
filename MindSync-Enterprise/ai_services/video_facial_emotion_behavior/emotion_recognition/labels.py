"""Canonical 7 emotion labels (FER2013 order).

Order matters: the CNN output index i corresponds to EMOTIONS[i].
Keep this single source of truth; model + aggregation must import it.
"""
from __future__ import annotations

EMOTIONS: tuple[str, ...] = (
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
)

NUM_EMOTIONS: int = len(EMOTIONS)

# --- Model metadata ---------------------------------------------------
#
# Single shared location for the checkpoint + its metadata filename.
# train_fer.py writes these; predict.py reads them. Both must reference
# these constants so training/inference can never drift.
DEFAULT_CKPT = "models/fer2013_cnn_best.pt"
DEFAULT_META = "models/fer2013_cnn_meta.json"

