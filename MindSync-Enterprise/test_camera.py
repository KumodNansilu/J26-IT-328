import sys
from PySide6.QtWidgets import QApplication
app = QApplication(sys.argv)
from app.ui.pages.video_checkin_page import VideoCheckinPage
page = VideoCheckinPage()
page.show()
sys.exit(app.exec())