"""
Risk gauge widget for displaying the Depression Severity Index (DSI).

A circular gauge with Green/Yellow/Red zones that visually indicates
the current depression risk level.
"""

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget


class RiskGauge(QWidget):
    """
    Circular gauge widget showing DSI risk level.

    Color zones:
        - Green:  DSI < 0.45 (Low Risk)
        - Yellow: 0.45 <= DSI < 0.70 (Moderate Risk)
        - Red:    DSI >= 0.70 (High Risk)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dsi = 0.0
        self.setMinimumSize(220, 220)

    def set_dsi(self, value: float):
        """Update the displayed DSI value (0-1)."""
        self._dsi = max(0.0, min(1.0, value))
        self.update()

    def _risk_color(self) -> QColor:
        """Return the color for the current DSI value."""
        if self._dsi < 0.45:
            return QColor(76, 175, 80)      # Green
        elif self._dsi < 0.70:
            return QColor(255, 193, 7)      # Yellow
        else:
            return QColor(244, 67, 54)      # Red

    def _risk_label(self) -> str:
        """Return the risk label for the current DSI value."""
        if self._dsi < 0.45:
            return "GREEN TIER\nLow Risk"
        elif self._dsi < 0.70:
            return "YELLOW TIER\nModerate Risk"
        else:
            return "RED TIER\nHigh Risk"

    def paintEvent(self, event):
        """Custom paint for the gauge."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Gauge dimensions
        side = min(self.width(), self.height())
        margin = 20
        gauge_rect = QRectF(
            (self.width() - side) / 2 + margin,
            (self.height() - side) / 2 + margin,
            side - 2 * margin,
            side - 2 * margin,
        )

        # Draw background arc (full circle)
        painter.setPen(QPen(QColor(230, 230, 230), 20))
        painter.drawArc(gauge_rect, 0, 360 * 16)

        # Draw colored arc based on DSI (270 degrees starting from bottom)
        start_angle = 135 * 16  # Start at bottom-left
        span_angle = int(-270 * 16 * self._dsi)  # Clockwise up to 270 degrees

        color = self._risk_color()
        painter.setPen(QPen(color, 20, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(gauge_rect, start_angle, span_angle)

        # Draw DSI value in center
        painter.setPen(QColor(50, 50, 50))
        font = QFont("Arial", 28, QFont.Bold)
        painter.setFont(font)
        dsi_text = f"{self._dsi:.2f}"
        text_rect = gauge_rect.adjusted(0, -20, 0, -20)
        painter.drawText(text_rect, Qt.AlignCenter, dsi_text)

        # Draw risk label below value
        painter.setPen(color)
        font = QFont("Arial", 11, QFont.Bold)
        painter.setFont(font)
        label_rect = gauge_rect.adjusted(0, 40, 0, 40)
        painter.drawText(label_rect, Qt.AlignCenter, self._risk_label())

        painter.end()