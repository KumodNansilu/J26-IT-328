"""
Core Fusion Engine - Gated Multimodal Fusion (GMU) for Depression Detection.

This module implements the central fusion architecture that combines
unimodal scores (TBS, VBS, ABS) into a unified Depression Severity Index (DSI).
"""

from .config import (
    MASKING_THRESHOLD,
    RED_TIER_THRESHOLD,
    YELLOW_TIER_THRESHOLD,
    RISK_TIERS,
    MASKING_ALERTS,
    DB_PATH,
    MODEL_WEIGHTS_PATH,
)
from .discordance import calculate_discordance_delta, classify_masking
from .fusion_model import GatedMultimodalFusionEngine

__all__ = [
    "MASKING_THRESHOLD",
    "RED_TIER_THRESHOLD",
    "YELLOW_TIER_THRESHOLD",
    "RISK_TIERS",
    "MASKING_ALERTS",
    "DB_PATH",
    "MODEL_WEIGHTS_PATH",
    "calculate_discordance_delta",
    "classify_masking",
    "GatedMultimodalFusionEngine",
]