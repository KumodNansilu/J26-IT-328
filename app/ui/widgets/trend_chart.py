"""14-day company depression trend line chart (theme-aware, matplotlib)."""

from typing import List, Dict

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt

from app.ui import get as theme_get

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


class TrendChart(QWidget):
    """Line chart: company avg DSI per day for the past 14 days."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if not _HAS_MPL:
            msg = QLabel("Install matplotlib to see the trend chart:\n pip install matplotlib")
            msg.setAlignment(Qt.AlignCenter)
            msg.setWordWrap(True)
            layout.addWidget(msg)
            self.canvas = None
            self.fig = None
            self.ax = None
            return
        self.fig = Figure(figsize=(7, 3.2), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        layout.addWidget(self.canvas)
        self.setMinimumHeight(260)

    def set_data(self, daily: List[Dict]):
        """Render daily avg DSI. daily: [{label, avg_dsi|None, count}]."""
        if not _HAS_MPL or self.ax is None:
            return
        t = {
            "TEXT": theme_get("TEXT"),
            "TEXT_DIM": theme_get("TEXT_DIM"),
            "BORDER": theme_get("BORDER"),
            "BG_CARD": theme_get("BG_CARD"),
            "GREEN": theme_get("GREEN"),
            "YELLOW": theme_get("YELLOW"),
            "RED": theme_get("RED"),
            "ACCENT": theme_get("ACCENT"),
        }
        ax = self.ax
        ax.clear()
        labels = [d["label"] for d in daily]
        vals = [d["avg_dsi"] for d in daily]

        self.fig.patch.set_facecolor(t["BG_CARD"])
        ax.set_facecolor(t["BG_CARD"])

        # Risk zone bands
        ax.axhspan(0.0, 0.45, color=t["GREEN"], alpha=0.10, linewidth=0)
        ax.axhspan(0.45, 0.70, color=t["YELLOW"], alpha=0.10, linewidth=0)
        ax.axhspan(0.70, 1.0, color=t["RED"], alpha=0.10, linewidth=0)
        ax.axhline(0.45, color=t["YELLOW"], alpha=0.35, linewidth=1, linestyle="--")
        ax.axhline(0.70, color=t["RED"], alpha=0.35, linewidth=1, linestyle="--")

        # Line with gaps for empty days
        xs = [i for i, v in enumerate(vals) if v is not None]
        ys = [v for v in vals if v is not None]
        if xs:
            ax.plot(xs, ys, color=t["ACCENT"], linewidth=2.5, zorder=3)
            ax.scatter(xs, ys, color=t["ACCENT"], s=34, zorder=4, edgecolors="white",
                       linewidths=1.0)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=8,
                            color=t["TEXT_DIM"])
            latest = ys[-1]
            color = t["GREEN"] if latest < 0.45 else (t["YELLOW"] if latest < 0.70 else t["RED"])
            ax.annotate(f"Today: {latest:.3f}", xy=(xs[-1], latest),
                        xytext=(12, -18), textcoords="offset points", fontsize=9,
                        fontweight="bold", color=color,
                        bbox=dict(boxstyle="round,pad=0.4", fc=t["BG_CARD"],
                                  ec=color, alpha=0.9))
        else:
            ax.text(0.5, 0.5, "No check-ins in the past 14 days\nEvaluate a session to begin",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=11, color=t["TEXT_DIM"])

        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_ylim(0, 1.02)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8,
                           color=t["TEXT_DIM"])
        ax.tick_params(axis="y", labelsize=8, colors=t["TEXT_DIM"])
        ax.set_ylabel("Avg DSI", fontsize=9, color=t["TEXT_DIM"])
        for spine in ax.spines.values():
            spine.set_color(t["BORDER"])
        ax.grid(True, axis="y", alpha=0.25, color=t["BORDER"])
        self.fig.tight_layout(pad=1.2)
        self.canvas.draw()

