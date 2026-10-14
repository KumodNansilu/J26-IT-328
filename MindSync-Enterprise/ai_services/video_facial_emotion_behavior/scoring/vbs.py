"""VBS calculator — PLACEHOLDER, no approved formula yet.

Do NOT invent emotion weights. `compute_vbs` raises NotImplementedError until
the research team approves and documents the scoring method (range, weights,
missing-data handling). Only then implement + enforce the range here.
"""
from __future__ import annotations


class VBSSpecMissingError(NotImplementedError):
    pass


def compute_vbs(video_emotion_probs, *, spec=None):
    """Compute VBS once an approved spec exists. Currently always raises."""
    _ = (video_emotion_probs, spec)
    raise VBSSpecMissingError(
        "No approved VBS formula. Propose a transparent, configurable scoring "
        "method (range, weights, interpretation, limitations) for approval "
        "before implementing.")
