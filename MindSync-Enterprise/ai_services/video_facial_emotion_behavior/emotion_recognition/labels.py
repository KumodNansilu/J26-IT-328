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

