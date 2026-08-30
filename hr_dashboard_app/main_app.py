"""
PySide6 Desktop UI for the HR Dashboard.

Features:
    a. Login Window validating against hr_users table
    b. Simulator Panel to manually test TBS, VBS, ABS inputs
    c. Visual DSI Gauge displaying Risk Tiers (Green/Yellow/Red)
    d. Dynamic Weight Breakdown Panel showing % weights
    e. Warning Banner triggered when |Δ| > 0.50
    f. Audit Log Table displaying historical DB records
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QDoubleSpinBox,
    QGroupBox,
    QMessageBox,
    QSplitter,
    QProgressBar,
    QFrame,
    QDialog,
    QFormLayout,
)

from core_fusion_engine.config import (
    MASKING_THRESHOLD,
    RED_TIER_THRESHOLD,
    YELLOW_TIER_THRESHOLD,
    MODEL_WEIGHTS_PATH,
)
from core_fusion_engine.discordance import calculate_discordance_delta, classify_masking
from core_fusion_engine.fusion_model import GatedMultimodalFusionEngine
from core_fusion_engine.severity import classify_risk_tier
from hr_dashboard_app.database_manager import DatabaseManager
from hr_dashboard_app.controllers.alert_controller import AlertController
from hr_dashboard_app.ui_components.risk_gauge import RiskGauge
from hr_dashboard_app.ui_components.session_table import SessionTable


# ---------------------------------------------------------------------------
# Login Dialog
# ---------------------------------------------------------------------------

class LoginDialog(QDialog):
    """Login window validating against the hr_users table."""

    def __init__(self, db: DatabaseManager, parent=None):
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("HR Dashboard - Login")
        self.setFixedSize(400, 250)

        layout = QVBoxLayout(self)

        # Title
        title = QLabel("🧠 HR Dashboard Login")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        # Form
        form = QFormLayout()

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("admin@company.com")
        form.addRow("Username:", self.username_input)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("admin123")
        self.password_input.setEchoMode(QLineEdit.Password)
        form.addRow("Password:", self.password_input)

        layout.addLayout(form)

        # Buttons
        btn_row = QHBoxLayout()
        login_btn = QPushButton("Login")
        login_btn.clicked.connect(self._handle_login)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(login_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        # Hint
        hint = QLabel("Default: admin@company.com / admin123")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(hint)

        # Enter key triggers login
        self.password_input.returnPressed.connect(self._handle_login)

    def _handle_login(self):
        """Validate credentials and accept if valid."""
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()

        user = self.db.validate_login(username, password)
        if user:
            self.accept()
        else:
            QMessageBox.warning(
                self,
                "Login Failed",
                "Invalid username or password. Please try again.",
            )


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """Main HR Dashboard window."""

    def __init__(self, db: DatabaseManager):
        super().__init__()
        self.db = db
        self.alert_controller = AlertController(db=db)
        self.model = self._load_model()

        self.setWindowTitle("HR Dashboard - Multimodal Depression Detection")
        self.setMinimumSize(1100, 750)

        self._build_ui()

        # Refresh timer
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_sessions)
        self.refresh_timer.start(5000)  # Refresh every 5 seconds

        # Initial data load
        self.refresh_sessions()

    def _load_model(self) -> GatedMultimodalFusionEngine | None:
        """Load the trained GMU model if weights exist."""
        if MODEL_WEIGHTS_PATH.exists():
            try:
                import torch

                model = GatedMultimodalFusionEngine()
                model.load_state_dict(
                    torch.load(MODEL_WEIGHTS_PATH, map_location="cpu")
                )
                model.eval()
                return model
            except Exception as e:
                print(f"Warning: Could not load model: {e}")
        return None

    def _build_ui(self):
        """Build the main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # Header
        header = QLabel("Multimodal Depression Detection - HR Dashboard")
        header.setStyleSheet("font-size: 20px; font-weight: bold; padding: 10px;")
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)

        # Splitter for left (simulator) and right (audit log)
        splitter = QSplitter(Qt.Horizontal)

        # Left panel: Simulator
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Risk gauge
        gauge_group = QGroupBox("Depression Severity Index (DSI)")
        gauge_layout = QVBoxLayout(gauge_group)
        self.risk_gauge = RiskGauge()
        gauge_layout.addWidget(self.risk_gauge)
        left_layout.addWidget(gauge_group)

        # Simulator inputs
        input_group = QGroupBox("Simulator Panel")
        input_layout = QVBoxLayout(input_group)

        # Employee ID
        emp_row = QHBoxLayout()
        emp_row.addWidget(QLabel("Employee ID:"))
        self.employee_input = QLineEdit()
        self.employee_input.setPlaceholderText("e.g., EMP001")
        emp_row.addWidget(self.employee_input)
        input_layout.addLayout(emp_row)

        # TBS
        tbs_row = QHBoxLayout()
        tbs_row.addWidget(QLabel("TBS:"))
        self.tbs_input = QDoubleSpinBox()
        self.tbs_input.setRange(0.0, 1.0)
        self.tbs_input.setSingleStep(0.05)
        self.tbs_input.setValue(0.30)
        tbs_row.addWidget(self.tbs_input)
        input_layout.addLayout(tbs_row)

        # VBS
        vbs_row = QHBoxLayout()
        vbs_row.addWidget(QLabel("VBS:"))
        self.vbs_input = QDoubleSpinBox()
        self.vbs_input.setRange(0.0, 1.0)
        self.vbs_input.setSingleStep(0.05)
        self.vbs_input.setValue(0.35)
        vbs_row.addWidget(self.vbs_input)
        input_layout.addLayout(vbs_row)

        # ABS
        abs_row = QHBoxLayout()
        abs_row.addWidget(QLabel("ABS:"))
        self.abs_input = QDoubleSpinBox()
        self.abs_input.setRange(0.0, 1.0)
        self.abs_input.setSingleStep(0.05)
        self.abs_input.setValue(0.28)
        abs_row.addWidget(self.abs_input)
        input_layout.addLayout(abs_row)

        # Evaluate button
        self.evaluate_btn = QPushButton("Evaluate Session")
        self.evaluate_btn.clicked.connect(self.evaluate_session)
        input_layout.addWidget(self.evaluate_btn)

        left_layout.addWidget(input_group)

        # Weight breakdown panel
        weight_group = QGroupBox("Dynamic Weight Breakdown")
        weight_layout = QVBoxLayout(weight_group)

        self.text_weight_bar = self._create_weight_bar("Text (w_text)", QColor(66, 133, 244))
        self.video_weight_bar = self._create_weight_bar("Video (w_video)", QColor(52, 168, 83))
        self.audio_weight_bar = self._create_weight_bar("Audio (w_audio)", QColor(251, 188, 5))

        weight_layout.addWidget(self.text_weight_bar)
        weight_layout.addWidget(self.video_weight_bar)
        weight_layout.addWidget(self.audio_weight_bar)

        left_layout.addWidget(weight_group)

        # Warning banner
        self.warning_banner = QLabel("")
        self.warning_banner.setStyleSheet(
            "background-color: #FFEBEE; color: #C62828; "
            "font-weight: bold; padding: 10px; border-radius: 5px;"
        )
        self.warning_banner.setWordWrap(True)
        self.warning_banner.hide()
        left_layout.addWidget(self.warning_banner)

        left_layout.addStretch()

        # Right panel: Audit log table
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        table_group = QGroupBox("Audit Log")
        table_layout = QVBoxLayout(table_group)
        self.session_table = SessionTable()
        table_layout.addWidget(self.session_table)
        right_layout.addWidget(table_group)

        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_sessions)
        right_layout.addWidget(refresh_btn)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([450, 650])

        main_layout.addWidget(splitter)

    def _create_weight_bar(self, label: str, color: QColor) -> QWidget:
        """Create a labeled progress bar for weight display."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 2, 0, 2)

        label_widget = QLabel(label)
        label_widget.setFixedWidth(110)
        layout.addWidget(label_widget)

        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(True)
        bar.setFormat("%v%")
        bar.setStyleSheet(
            f"QProgressBar {{ border: 1px solid #ccc; border-radius: 3px; }}"
            f"QProgressBar::chunk {{ background-color: {color.name()}; }}"
        )
        layout.addWidget(bar)

        # Store reference to the bar
        widget.bar = bar
        return widget

    def evaluate_session(self):
        """Evaluate a new check-in session from the input fields."""
        try:
            employee_id = self.employee_input.text().strip()
            if not employee_id:
                QMessageBox.warning(self, "Input Error", "Please enter an Employee ID.")
                return

            tbs = self.tbs_input.value()
            vbs = self.vbs_input.value()
            abs_ = self.abs_input.value()

            # Compute DSI using model or fallback to weighted average
            if self.model is not None:
                dsi, weights, delta = self.model.predict(tbs, vbs, abs_)
            else:
                # Fallback: simple average
                dsi = (tbs + vbs + abs_) / 3.0
                weights = [1/3, 1/3, 1/3]
                delta = calculate_discordance_delta(tbs, vbs, abs_)

            # Classify risk tier
            risk_tier = classify_risk_tier(dsi)

            # Check masking
            masking = classify_masking(delta)

            # Save to database
            self.db.save_checkin(
                employee_id=employee_id,
                tbs=tbs,
                vbs=vbs,
                abs_=abs_,
                dsi=dsi,
                delta=delta,
                risk_tier=risk_tier.name,
                masking_alert=masking["alert_message"] if masking["is_masking"] else None,
            )

            # Update gauge
            self.risk_gauge.set_dsi(dsi)

            # Update weight bars
            self.text_weight_bar.bar.setValue(int(weights[0] * 100))
            self.video_weight_bar.bar.setValue(int(weights[1] * 100))
            self.audio_weight_bar.bar.setValue(int(weights[2] * 100))

            # Show/hide warning banner
            if masking["is_masking"]:
                self.warning_banner.setText(
                    f"⚠️ {masking['alert_message']} (Δ={delta:+.2f})"
                )
                self.warning_banner.show()
            else:
                self.warning_banner.hide()

            # Refresh table
            self.refresh_sessions()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def refresh_sessions(self):
        """Refresh the session table with latest data."""
        sessions = self.db.get_checkins(limit=50)
        self.session_table.set_sessions(sessions)

        # Update gauge with latest DSI if available
        if sessions:
            self.risk_gauge.set_dsi(sessions[0]["dsi"])


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

def main():
    """Application entry point."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Initialize database
    db = DatabaseManager()

    # Show login dialog
    login = LoginDialog(db)
    if login.exec() != QDialog.Accepted:
        sys.exit(0)

    # Show main window
    window = MainWindow(db)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()