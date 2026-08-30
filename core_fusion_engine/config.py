"""
Configuration settings for the Central Decision Engine & HR Analytics Dashboard.

This module centralizes all tunable thresholds, risk tier boundaries, and
application-wide constants used across the fusion engine, dashboard UI,
and database manager.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "core_fusion_engine"

# ---------------------------------------------------------------------------
# Database Configuration
# ---------------------------------------------------------------------------
DB_PATH = DATA_DIR / "local_audit_logs.db"
DB_TABLE_NAME = "checkins"

# ---------------------------------------------------------------------------
# Discordance Delta (Δ) Thresholds
# ---------------------------------------------------------------------------
# |Δ| > MASKING_THRESHOLD indicates emotional masking / forced composure.
MASKING_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Risk Tier Boundaries (based on Final DSI)
# ---------------------------------------------------------------------------
RED_TIER_THRESHOLD = 0.70      # DSI >= 0.70  -> RED TIER (High Risk)
YELLOW_TIER_THRESHOLD = 0.45   # 0.45 <= DSI < 0.70 -> YELLOW TIER (Moderate Risk)
                               # DSI < 0.45  -> GREEN TIER (Low Risk)

# ---------------------------------------------------------------------------
# Model Training Hyperparameters
# ---------------------------------------------------------------------------
TRAIN_EPOCHS = 40
TRAIN_LR = 0.01
TRAIN_SAMPLES = 1000
TRAIN_BATCH_SIZE = 32
MODEL_WEIGHTS_PATH = MODEL_DIR / "gmu_fusion_model.pth"

# ---------------------------------------------------------------------------
# Risk Tier Labels
# ---------------------------------------------------------------------------
RISK_TIERS = {
    "RED": "RED TIER (High Risk)",
    "YELLOW": "YELLOW TIER (Moderate Risk)",
    "GREEN": "GREEN TIER (Low Risk)",
}

# ---------------------------------------------------------------------------
# Masking Alert Labels
# ---------------------------------------------------------------------------
MASKING_ALERTS = {
    "positive": "Traditional Emotional Masking (Fake Happy Text)",
    "negative": "Forced Composure (Distressed Text, Calm Exterior)",
}