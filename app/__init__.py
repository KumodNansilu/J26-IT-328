"""
HR Dashboard Application - PySide6 desktop UI for the multimodal depression detection system.

Provides HR personnel with a real-time dashboard showing Depression Severity Index (DSI),
risk gauges, session tables, and alert notifications.
"""

from app.database.connection import DatabaseManager
from app.services.dsi_service import AlertController

__all__ = [
    "DatabaseManager",
    "AlertController",
]