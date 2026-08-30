"""
UI Components for the HR Dashboard Application.

Contains custom PySide6 widgets for displaying risk gauges and session tables.
"""

from .risk_gauge import RiskGauge
from .session_table import SessionTable

__all__ = ["RiskGauge", "SessionTable"]