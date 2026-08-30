"""
Session table widget for displaying check-in audit logs.

A sortable table view showing session history with color-coded risk tiers.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView


class SessionTable(QTableWidget):
    """
    Table widget for displaying check-in sessions.

    Columns:
        - Timestamp
        - Employee ID
        - TBS
        - VBS
        - ABS
        - DSI
        - Δ (Delta)
        - Risk Tier
        - Masking Alert
    """

    COLUMNS = [
        "Timestamp",
        "Employee ID",
        "TBS",
        "VBS",
        "ABS",
        "DSI",
        "Δ (Delta)",
        "Risk Tier",
        "Masking Alert",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(len(self.COLUMNS))
        self.setHorizontalHeaderLabels(self.COLUMNS)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setAlternatingRowColors(True)

    def set_sessions(self, sessions: list):
        """
        Populate the table with session data.

        Args:
            sessions: List of session dicts from DatabaseManager.
        """
        self.setRowCount(len(sessions))

        for row, session in enumerate(sessions):
            # Timestamp
            self._set_item(row, 0, session.get("timestamp", ""))

            # Employee ID
            self._set_item(row, 1, session.get("employee_id", ""))

            # TBS, VBS, ABS
            self._set_item(row, 2, f"{session.get('tbs', 0):.3f}")
            self._set_item(row, 3, f"{session.get('vbs', 0):.3f}")
            self._set_item(row, 4, f"{session.get('abs', 0):.3f}")

            # DSI
            dsi = session.get("dsi", 0)
            dsi_item = self._set_item(row, 5, f"{dsi:.3f}")
            dsi_item.setForeground(QBrush(self._risk_color(dsi)))

            # Delta
            delta = session.get("delta", 0)
            delta_item = self._set_item(row, 6, f"{delta:+.3f}")
            if abs(delta) > 0.50:
                delta_item.setForeground(QBrush(QColor(244, 67, 54)))  # Red

            # Risk Tier
            risk_tier = session.get("risk_tier", "")
            tier_item = self._set_item(row, 7, risk_tier)
            tier_item.setForeground(QBrush(self._tier_color(risk_tier)))

            # Masking Alert
            masking = session.get("masking_alert", "")
            self._set_item(row, 8, masking or "—")

    def _set_item(self, row: int, col: int, text: str) -> QTableWidgetItem:
        """Create and set a table item."""
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        self.setItem(row, col, item)
        return item

    def _risk_color(self, dsi: float) -> QColor:
        """Return color based on DSI risk level."""
        if dsi < 0.45:
            return QColor(76, 175, 80)      # Green
        elif dsi < 0.70:
            return QColor(255, 152, 0)      # Orange/Yellow
        else:
            return QColor(244, 67, 54)      # Red

    def _tier_color(self, tier: str) -> QColor:
        """Return color based on risk tier."""
        if tier == "GREEN":
            return QColor(76, 175, 80)
        elif tier == "YELLOW":
            return QColor(255, 152, 0)
        elif tier == "RED":
            return QColor(244, 67, 54)
        return QColor(50, 50, 50)