"""
Risk gauge widget for displaying the Depression Severity Index (DSI).

Modern segmented dial: light track, three risk arcs (green/yellow/red),
needle marker, glow dot and high-contrast center readout.
"""

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from app.ui import get as theme_get

GREEN_DSI_MAX = 0.45
YELLOW_DSI_MAX = 0.70


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
        self.setMinimumSize(240, 240)

    def set_dsi(self, value: float):
        """Update the displayed DSI value (0-1)."""
        self._dsi = max(0.0, min(1.0, value))
        self.update()

    def _risk_color(self) -> QColor:
        """Return the color for the current DSI value."""
        if self._dsi < GREEN_DSI_MAX:
            return QColor(theme_get("GREEN"))
        elif self._dsi < YELLOW_DSI_MAX:
            return QColor(theme_get("YELLOW"))
        else:
            return QColor(theme_get("RED"))

    def _risk_label(self) -> str:
        """Return the risk label for the current DSI value."""
        if self._dsi < GREEN_DSI_MAX:
            return "GREEN TIER\nLow Risk"
        elif self._dsi < YELLOW_DSI_MAX:
            return "YELLOW TIER\nModerate Risk"
        else:
            return "RED TIER\nHigh Risk"

    def paintEvent(self, event):
        """Custom paint for the gauge."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Gauge geometry: 270-degree dial with a gap at the bottom
        side = min(self.width(), self.height())
        margin = 22
        gauge_rect = QRectF(
            (self.width() - side) / 2 + margin,
            (self.height() - side) / 2 + margin,
            side - 2 * margin,
            side - 2 * margin,
        )

        start_angle = 135 * 16  # bottom-left, clockwise dial
        full_span = -270 * 16

        GREEN = QColor(theme_get("GREEN"))
        YELLOW = QColor(theme_get("YELLOW"))
        RED = QColor(theme_get("RED"))
        TRACK = QColor(theme_get("TRACK"))
        TEXT_MAIN = QColor(theme_get("TEXT"))
        TEXT_DIM = QColor(theme_get("TEXT_DIM"))

        # Track
        painter.setPen(QPen(TRACK, 18, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(gauge_rect, start_angle, full_span)

        # Faint zone markers (green / yellow / red thirds of the dial)
        zone_pen_width = 18
        for zone_start_frac, zone_frac, color in (
            (0.0, GREEN_DSI_MAX, GREEN),
            (GREEN_DSI_MAX, YELLOW_DSI_MAX - GREEN_DSI_MAX, YELLOW),
            (YELLOW_DSI_MAX, 1.0 - YELLOW_DSI_MAX, RED),
        ):
            faint = QColor(color)
            faint.setAlpha(52)
            painter.setPen(QPen(faint, zone_pen_width, Qt.SolidLine, Qt.FlatCap))
            painter.drawArc(
                gauge_rect,
                int(start_angle + full_span * zone_start_frac + 40),
                int(full_span * zone_frac - 80),
            )

        # Active value arc
        color = self._risk_color()
        painter.setPen(QPen(color, 18, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(gauge_rect, start_angle, int(full_span * self._dsi))

        # Needle dot at the tip of the active arc
        import math

        angle_deg = 135 - 270 * self._dsi
        angle_rad = math.radians(angle_deg)
        cx = gauge_rect.center().x()
        cy = gauge_rect.center().y()
        radius = gauge_rect.width() / 2
        tip_x = cx + radius * math.cos(angle_rad)
        tip_y = cy - radius * math.sin(angle_rad)
        glow = QColor(color)
        glow.setAlpha(70)
        painter.setPen(Qt.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(int(tip_x) - 11, int(tip_y) - 11, 22, 22)
        painter.setBrush(color)
        painter.drawEllipse(int(tip_x) - 6, int(tip_y) - 6, 12, 12)

        # Center DSI value (bright, always readable on dark bg)
        painter.setPen(TEXT_MAIN)
        font = QFont("Segoe UI", 34, QFont.Bold)
        painter.setFont(font)
        dsi_text = f"{self._dsi:.2f}"
        text_rect = gauge_rect.adjusted(0, -26, 0, -26)
        painter.drawText(text_rect, Qt.AlignCenter, dsi_text)

        # Small caption + risk pill text below value
        painter.setPen(TEXT_DIM)
        font = QFont("Segoe UI", 10)
        painter.setFont(font)
        painter.drawText(
            gauge_rect.adjusted(0, 12, 0, 12), Qt.AlignCenter, "DEPRESSION SEVERITY"
        )
        painter.setPen(color)
        font = QFont("Segoe UI", 11, QFont.Bold)
        painter.setFont(font)
        label_rect = gauge_rect.adjusted(0, 44, 0, 44)
        painter.drawText(label_rect, Qt.AlignCenter, self._risk_label())

        painter.end()
