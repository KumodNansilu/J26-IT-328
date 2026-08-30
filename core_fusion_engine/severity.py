"""
Severity classification module.

Maps the fused DSI (Depression Severity Index) value to a categorical
severity level based on standard PHQ-9 thresholds.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeverityLevel:
    """Represents a single severity level definition."""
    name: str
    min_score: float
    max_score: float
    color: str
    description: str


# Standard PHQ-9 severity thresholds
SEVERITY_LEVELS: List[SeverityLevel] = [
    SeverityLevel(
        name="Minimal",
        min_score=0.0,
        max_score=4.0,
        color="#4CAF50",  # Green
        description="Minimal or no depressive symptoms",
    ),
    SeverityLevel(
        name="Mild",
        min_score=4.0,
        max_score=9.0,
        color="#8BC34A",  # Light green
        description="Mild depressive symptoms",
    ),
    SeverityLevel(
        name="Moderate",
        min_score=9.0,
        max_score=14.0,
        color="#FFC107",  # Amber
        description="Moderate depressive symptoms",
    ),
    SeverityLevel(
        name="Moderately Severe",
        min_score=14.0,
        max_score=19.0,
        color="#FF9800",  # Orange
        description="Moderately severe depressive symptoms",
    ),
    SeverityLevel(
        name="Severe",
        min_score=19.0,
        max_score=27.0,
        color="#F44336",  # Red
        description="Severe depressive symptoms",
    ),
]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_severity(dsi: float) -> SeverityLevel:
    """
    Classify a DSI score into a severity level.

    Parameters
    ----------
    dsi : float
        The fused Depression Severity Index score (0-27).

    Returns
    -------
    SeverityLevel
        The matching severity level definition.

    Raises
    ------
    ValueError
        If the DSI score is outside the valid range [0, 27].
    """
    if dsi < 0.0 or dsi > 27.0:
        raise ValueError(
            f"DSI score {dsi:.2f} is outside the valid range [0, 27]."
        )

    for level in SEVERITY_LEVELS:
        if level.min_score <= dsi < level.max_score:
            return level

    # Handle the exact upper boundary (27.0)
    return SEVERITY_LEVELS[-1]


def get_severity_summary(dsi: float) -> Dict[str, object]:
    """
    Build a summary dictionary for a given DSI score.

    Parameters
    ----------
    dsi : float
        The fused DSI score.

    Returns
    -------
    dict
        Dictionary containing the DSI, severity level name, color,
        and description.
    """
    level = classify_severity(dsi)

    return {
        "dsi": round(float(dsi), 2),
        "severity": level.name,
        "color": level.color,
        "description": level.description,
    }


def get_severity_thresholds() -> List[Tuple[str, float, float]]:
    """
    Return the severity thresholds as a list of tuples.

    Useful for UI gauge configuration.

    Returns
    -------
    list of tuple
        Each tuple is (name, min_score, max_score).
    """
    return [
        (level.name, level.min_score, level.max_score)
        for level in SEVERITY_LEVELS
    ]