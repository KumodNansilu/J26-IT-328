"""
HR Dashboard Application - PySide6 desktop UI for the multimodal depression detection system.

Provides HR personnel with a real-time dashboard showing Depression Severity Index (DSI),
risk gauges, session tables, and alert notifications.
"""

from .database_manager import DatabaseManager
from .controllers.alert_controller import AlertController

__all__ = [
    "DatabaseManager",
    "AlertController",
]