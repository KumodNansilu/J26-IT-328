"""
Severity classification module.

Maps the fused DSI (Depression Severity Index) value in [0, 1] to a
categorical risk tier based on the configured thresholds.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .config import RED_TIER_THRESHOLD, YELLOW_TIER_THRESHOLD, RISK_TIERS


# ---------------------------------------------------------------------------
# Risk tiers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskTier:
    """Represents a single risk tier definition."""
    name: str
    min_score: float
    max_score: float
    color: str
    description: str


RISK_TIERS_DEFINITIONS: List[RiskTier] = [
    RiskTier(
        name="GREEN",
        min_score=0.0,
        max_score=YELLOW_TIER_THRESHOLD,
        color="#4CAF50",  # Green
        description="Low Risk - No significant depressive symptoms detected.",
    ),
    RiskTier(
        name="YELLOW",
        min_score=YELLOW_TIER_THRESHOLD,
        max_score=RED_TIER_THRESHOLD,
        color="#FFC107",  # Yellow/Amber
        description="Moderate Risk - Depressive symptoms may be present. Monitor closely.",
    ),
    RiskTier(
        name="RED",
        min_score=RED_TIER_THRESHOLD,
        max_score=1.0,
        color="#F44336",  # Red
        description="High Risk - Significant depressive symptoms detected. Immediate attention required.",
    ),
]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_risk_tier(dsi: float) -> RiskTier:
    """
    Classify a DSI score into a risk tier.

    Parameters
    ----------
    dsi : float
        The fused Depression Severity Index score in [0, 1].

    Returns
    -------
    RiskTier
        The matching risk tier definition.

    Raises
    ------
    ValueError
        If the DSI score is outside the valid range [0, 1].
    """
    if dsi < 0.0 or dsi > 1.0:
        raise ValueError(
            f"DSI score {dsi:.2f} is outside the valid range [0, 1]."
        )

    for tier in RISK_TIERS_DEFINITIONS:
        if tier.min_score <= dsi < tier.max_score:
            return tier

    # Handle the exact upper boundary (1.0)
    return RISK_TIERS_DEFINITIONS[-1]


def get_risk_summary(dsi: float) -> Dict[str, object]:
    """
    Build a summary dictionary for a given DSI score.

    Parameters
    ----------
    dsi : float
        The fused DSI score in [0, 1].

    Returns
    -------
    dict
        Dictionary containing the DSI, risk tier name, color,
        and description.
    """
    tier = classify_risk_tier(dsi)

    return {
        "dsi": round(float(dsi), 3),
        "risk_tier": tier.name,
        "color": tier.color,
        "description": tier.description,
    }


def get_risk_thresholds() -> List[Tuple[str, float, float]]:
    """
    Return the risk tier thresholds as a list of tuples.

    Useful for UI gauge configuration.

    Returns
    -------
    list of tuple
        Each tuple is (name, min_score, max_score).
    """
    return [
        (tier.name, tier.min_score, tier.max_score)
        for tier in RISK_TIERS_DEFINITIONS
    ]