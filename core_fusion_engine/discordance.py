"""
Discordance Delta (Δ) Logic.

The Discordance Delta quantifies the mismatch between the emotional state
expressed in the employee's text (TBS) and the emotional state inferred
from their non-verbal cues (VBS and ABS).

    Δ = max(VBS, ABS) - TBS

Interpretation:
    - High positive Δ (> +0.50): Traditional Emotional Masking
      (Fake Happy Text - text is more positive than non-verbal cues).
    - High negative Δ (< -0.50): Forced Composure
      (Distressed Text, Calm Exterior - text is more negative than
      non-verbal cues).
"""

from typing import Dict, Union

from .config import MASKING_THRESHOLD, MASKING_ALERTS


def calculate_discordance_delta(
    tbs: float,
    vbs: float,
    abs_score: float,
) -> float:
    """
    Calculate the Discordance Delta (Δ) between text and non-verbal scores.

    Δ = max(VBS, ABS) - TBS

    Parameters
    ----------
    tbs : float
        Text-Based Sentiment score in [0, 1].
    vbs : float
        Visual-Based Sentiment score in [0, 1].
    abs_score : float
        Audio-Based Sentiment score in [0, 1].

    Returns
    -------
    float
        Discordance Delta (Δ) in [-1, 1].
    """
    non_verbal_max = max(vbs, abs_score)
    return non_verbal_max - tbs


def classify_masking(delta: float) -> Dict[str, Union[str, float, bool]]:
    """
    Classify whether the discordance delta indicates emotional masking.

    Parameters
    ----------
    delta : float
        Discordance Delta (Δ) computed by :func:`calculate_discordance_delta`.

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