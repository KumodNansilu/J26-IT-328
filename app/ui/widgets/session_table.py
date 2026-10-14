"""
Session table widget for displaying check-in audit logs.

Dark card table with pill-style risk badges and readable timestamps.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView

from app.ui import get as theme_get


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
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(8, QHeaderView.Stretch)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(30)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        self.setSortingEnabled(True)

    def set_sessions(self, sessions: list):
        """
        Populate the table with session data.

        Args:
            sessions: List of session dicts from DatabaseManager.
        """
        self.setSortingEnabled(False)
        self.setRowCount(len(sessions))

        for row, session in enumerate(sessions):
            # Timestamp (short, readable)
            ts = str(session.get("timestamp", ""))[:16].replace("T", " ")
            self._set_item(row, 0, ts, dim=True)

            # Employee ID
            self._set_item(row, 1, session.get("employee_id", ""), bold=True)

            # TBS, VBS, ABS
            self._set_item(row, 2, f"{session.get('tbs', 0):.3f}")
            self._set_item(row, 3, f"{session.get('vbs', 0):.3f}")
            self._set_item(row, 4, f"{session.get('abs', 0):.3f}")

            # DSI (pill badge)
            dsi = session.get("dsi", 0)
            self._set_pill(row, 5, f"{dsi:.3f}", self._risk_color(dsi))

            # Delta
            delta = session.get("delta", 0)
            delta_item = self._set_item(row, 6, f"{delta:+.3f}")
            if abs(delta) > 0.50:
                delta_item.setForeground(QBrush(QColor(theme_get("RED"))))
                font = delta_item.font()
                font.setBold(True)
                delta_item.setFont(font)
            else:
                delta_item.setForeground(QBrush(QColor(theme_get("TEXT_DIM"))))

            # Risk Tier (pill badge)
            risk_tier = session.get("risk_tier", "")
            self._set_pill(row, 7, risk_tier, self._tier_color(risk_tier))

            # Masking Alert
            masking = session.get("masking_alert", "")
            if masking:
                item = self._set_item(row, 8, "⚠ " + str(masking)[:34])
                item.setForeground(QBrush(QColor(theme_get("YELLOW"))))
                item.setToolTip(str(masking))
            else:
                self._set_item(row, 8, "—", dim=True)

        self.setSortingEnabled(True)

    def _set_item(self, row: int, col: int, text: str, dim=False, bold=False) -> QTableWidgetItem:
        """Create and set a table item."""
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        if dim:
            item.setForeground(QBrush(QColor(theme_get("TEXT_DIM"))))
        if bold:
            font = item.font()
            font.setBold(True)
            item.setFont(font)
        self.setItem(row, col, item)
        return item

    def _set_pill(self, row: int, col: int, text: str, color: QColor) -> QTableWidgetItem:
        """Pill-style badge cell with tinted background."""
        item = QTableWidgetItem(f"  ● {text}  ")
        item.setTextAlignment(Qt.AlignCenter)
        bg = QColor(color)
        bg.setAlpha(38)
        item.setBackground(QBrush(bg))
        item.setForeground(QBrush(color))
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        self.setItem(row, col, item)
        return item

    def _risk_color(self, dsi: float) -> QColor:
        """Return color based on DSI risk level."""
        if dsi < 0.45:
            return QColor(theme_get("GREEN"))
        elif dsi < 0.70:
            return QColor(theme_get("YELLOW"))
        else:
            return QColor(theme_get("RED"))

    def _tier_color(self, tier: str) -> QColor:
        """Return color based on risk tier."""
        if tier == "GREEN":
            return QColor(theme_get("GREEN"))
        elif tier == "YELLOW":
            return QColor(theme_get("YELLOW"))
        elif tier == "RED":
            return QColor(theme_get("RED"))
        return QColor(theme_get("TEXT_DIM"))
