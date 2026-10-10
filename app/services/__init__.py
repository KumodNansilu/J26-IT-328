"""
Controllers for the HR Dashboard Application.

Contains logic controllers for alert management and session processing.
"""

from .dsi_service import AlertController, Alert

__all__ = ["AlertController", "Alert"]