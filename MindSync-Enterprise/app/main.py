"""MindSync Enterprise entry point — launches Video Check-in (VFEBA input)."""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QMainWindow

from app.ui.pages.video_checkin_page import VideoCheckinPage


def main() -> int:
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle("MindSync Enterprise — Video Check-in")
    win.setCentralWidget(VideoCheckinPage())
    win.resize(640, 800)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

