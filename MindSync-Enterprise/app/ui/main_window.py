"""Main window with stacked navigation between check-in and analysis pages."""
from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QStackedWidget

from app.ui.pages.video_analysis_page import VideoAnalysisPage
from app.ui.pages.video_checkin_page import VideoCheckinPage


class MainWindow(QMainWindow):
    """Stacked-container main window: check-in -> analysis navigation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MindSync Enterprise — Video Check-in")
        self.resize(640, 900)

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self.checkin_page = VideoCheckinPage()
        self.analysis_page = VideoAnalysisPage()

        self._stack.addWidget(self.checkin_page)
        self._stack.addWidget(self.analysis_page)

        # Wire check-in -> analysis navigation.
        self.checkin_page.analysis_requested.connect(self._show_analysis)

        self._stack.setCurrentWidget(self.checkin_page)

    def _show_analysis(self, video_path: str):
        """Switch to the analysis page and feed it the validated video path."""
        self.analysis_page.set_video_path(video_path)
        self._stack.setCurrentWidget(self.analysis_page)
