import sys
sys.path.insert(0, r'c:/Users/User/Desktop/J26-IT-328/MindSync-Enterprise')

from app.ui.main_window import MainWindow
from PySide6.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)
win = MainWindow()
win.show()
sys.exit(app.exec())