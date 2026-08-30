"""
Discordance Delta (Δ) Logic.

The Discordance Delta quantifies the mismatch between the emotional state
expressed in the employee's text (TBS) and the emotional state inferred
from their non-verbal cues (VBS and ABS).

    Δ = TBS - (VBS + ABS) / 2

A large positive Δ indicates the employee is masking negative emotions
with positive text (Traditional Emotional Masking).
A large negative Δ indicates the employee is forcing composure while
their text reveals distress (Forced Composure).
"""

from typing import Dict, Union

from .config import MASKING_THRESHOLD, MASKING_ALERTS


def compute_discordance_delta(
    tbs: float,
    vbs: float,
    abs_: float,
) -> float:
    """
    Compute the Discordance Delta (Δ) between text and non-verbal scores.

    Parameters
    ----------
    tbs : float
        Text-Based Sentiment score in [0, 1].
    vbs : float
        Visual-Based Sentiment score in [0, 1].
    abs_ : float
        Audio-Based Sentiment score in [0, 1].

    Returns
    -------
    float
        Discordance Delta (Δ) in [-1, 1].
    """
    non_verbal_mean = (vbs + abs_) / 2.0
    return tbs - non_verbal_mean


def classify_masking(delta: float) -> Dict[str, Union[str, float, bool]]:
    """
    Classify whether the discordance delta indicates emotional masking.

    Parameters
    ----------
    delta : float
        Discordance Delta (Δ) computed by :func:`compute_discordance_delta`.

    Returns
    -------
    dict
        A dictionary containing:
            - "delta"          : the raw discordance delta
            - "is_masking"     : True if |Δ| exceeds the masking threshold
            - "masking_type"   : "positive" / "negative" / None
            - "alert_message"  : human-readable alert string
    """
    abs_delta = abs(delta)
    is_masking = abs_delta > MASKING_THRESHOLD

    if not is_masking:
        return {
            "delta": delta,
            "is_masking": False,
            "masking_type": None,
            "alert_message": "No significant emotional masking detected.",
        }

    masking_type = "positive" if delta > 0 else "negative"
    alert_message = MASKING_ALERTS[masking_type]

    return {
        "delta": delta,
        "is_masking": True,
        "masking_type": masking_type,
        "alert_message": alert_message,
    }