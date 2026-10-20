"""Add-new-user dialog: User ID + Username + Password -> MongoDB."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QMessageBox,
)

from app.database.mongo_connection import MongoUserManager


class AddUserDialog(QDialog):
    """Form dialog that inserts an HR user into MindSync_DB.users."""

    def __init__(self, mongo: MongoUserManager, parent=None):
        super().__init__(parent)
        self.mongo = mongo
        self.setWindowTitle("Add New User — MindSync_DB")
        self.setFixedSize(430, 340)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 22, 28, 22)
        layout.setSpacing(10)

        title = QLabel("＋  Add New User")
        title.setObjectName("loginTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        sub = QLabel("Saved to MongoDB · MindSync_DB · users")
        sub.setObjectName("loginSub")
        sub.setAlignment(Qt.AlignCenter)
        layout.addWidget(sub)

        form = QFormLayout()
        form.setSpacing(8)
        self.user_id_input = QLineEdit()
        self.user_id_input.setPlaceholderText("e.g. HR002")
        form.addRow("User ID:", self.user_id_input)

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("e.g. nurse@company.com")
        form.addRow("Username:", self.username_input)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("min 4 characters")
        self.password_input.setEchoMode(QLineEdit.Password)
        form.addRow("Password:", self.password_input)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Full name (optional)")
        form.addRow("Full name:", self.name_input)
        layout.addLayout(form)

        self.status = QLabel("")
        self.status.setObjectName("loginHint")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save User")
        save_btn.setCursor(Qt.PointingHandCursor)
        save_btn.clicked.connect(self._handle_save)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("ghost")
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def _handle_save(self):
        try:
            doc = self.mongo.add_user(
                user_id=self.user_id_input.text(),
                username=self.username_input.text(),
                password=self.password_input.text(),
                full_name=self.name_input.text(),
            )
        except ValueError as e:
            self.status.setText(str(e))
            return
        except Exception as e:
            QMessageBox.critical(self, "MongoDB Error", f"Could not save user:\n{e}")
            return
        QMessageBox.information(
            self, "Saved", f"User '{doc['username']}' saved to MindSync_DB.users"
        )
        self.accept()
