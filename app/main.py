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

from ai_services.dsi_fusion_member4.config import (
    MASKING_THRESHOLD,
    RED_TIER_THRESHOLD,
    YELLOW_TIER_THRESHOLD,
    MODEL_WEIGHTS_PATH,
)
from ai_services.dsi_fusion_member4.discordance_delta import calculate_discordance_delta, classify_masking
from ai_services.dsi_fusion_member4.softmax_gmu import GatedMultimodalFusionEngine
from ai_services.dsi_fusion_member4.severity import classify_risk_tier
from app.database.connection import DatabaseManager
from app.services.dsi_service import AlertController
from app.ui import get as theme_get, mode as theme_mode
from app.ui.widgets.risk_gauge import RiskGauge
from app.ui.widgets.session_table import SessionTable
from app.ui.widgets.trend_chart import TrendChart
from app.utils.theme import apply_theme, saved_mode


# ---------------------------------------------------------------------------
# Login Dialog
# ---------------------------------------------------------------------------

class LoginDialog(QDialog):
    """Login window: MongoDB MindSync_DB.users first, SQLite fallback."""

    def __init__(self, db: DatabaseManager, parent=None):
        super().__init__(parent)
        self.db = db
        try:
            from app.database.mongo_connection import MongoUserManager

            self.mongo = MongoUserManager()
        except Exception as e:
            print(f"Mongo unavailable for login ({e}); using SQLite only.")
            self.mongo = None
        self.setWindowTitle("MindSync — HR Login")
        self.setFixedSize(420, 330)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 22, 28, 22)
        layout.setSpacing(10)

        # Title
        title = QLabel("●  MindSync")
        title.setObjectName("loginTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        sub = QLabel("HR Dashboard Login")
        sub.setObjectName("loginSub")
        sub.setAlignment(Qt.AlignCenter)
        layout.addWidget(sub)

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
        login_btn.setCursor(Qt.PointingHandCursor)
        login_btn.clicked.connect(self._handle_login)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("ghost")
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(login_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        # Hint
        hint = QLabel("Default: admin@company.com / admin123")
        hint.setObjectName("loginHint")
        hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint)

        # Enter key triggers login
        self.password_input.returnPressed.connect(self._handle_login)

    def _handle_login(self):
        """Validate credentials (Mongo first, SQLite fallback) and accept if valid."""
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()

        if self.mongo is not None:
            try:
                if self.mongo.validate_login(username, password):
                    self.accept()
                    return
            except Exception as e:
                print(f"Mongo login failed, trying SQLite: {e}")

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

        self.setWindowTitle("MindSync — HR Dashboard")
        self.setMinimumSize(1200, 780)

        self._build_ui()
        self._refresh_theme_chrome()

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
        main_layout.setContentsMargins(18, 14, 18, 14)
        main_layout.setSpacing(12)

        # Header bar: logo dot + title + live stats pills
        header_bar = QHBoxLayout()
        header_bar.setSpacing(12)

        logo = QLabel("●")
        logo.setObjectName("logoDot")
        logo.setFixedWidth(24)
        header_bar.addWidget(logo)

        title_box = QVBoxLayout()
        title = QLabel("MindSync · HR Dashboard")
        title.setObjectName("appTitle")
        subtitle = QLabel("Multimodal Depression Detection — TBS · VBS · ABS → DSI")
        subtitle.setObjectName("appSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        title_box.setSpacing(0)
        header_bar.addLayout(title_box)
        header_bar.addStretch()

        self.stat_total = self._stat_pill("Sessions", "0")
        self.stat_avg = self._stat_pill("Avg DSI", "--")
        self.stat_red = self._stat_pill("High risk", "0", "RED")
        self.stat_mask = self._stat_pill("Masking", "0", "YELLOW")
        for pill in (self.stat_total, self.stat_avg, self.stat_red, self.stat_mask):
            header_bar.addWidget(pill)

        # Add New User button (MongoDB form)
        self.add_user_btn = QPushButton("＋ Add New User")
        self.add_user_btn.setCursor(Qt.PointingHandCursor)
        self.add_user_btn.setToolTip("Add HR user to MongoDB MindSync_DB.users")
        self.add_user_btn.clicked.connect(self.open_add_user)
        header_bar.addWidget(self.add_user_btn)

        # Theme toggle (sun/moon)
        self.theme_btn = QPushButton()
        self.theme_btn.setObjectName("ghost")
        self.theme_btn.setCursor(Qt.PointingHandCursor)
        self.theme_btn.setFixedSize(44, 36)
        self.theme_btn.setToolTip("Toggle light / dark theme")
        self.theme_btn.clicked.connect(self.toggle_theme)
        header_bar.addWidget(self.theme_btn)

        main_layout.addLayout(header_bar)

        # Splitter for left (gauge+simulator) and right (audit log)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(10)

        # Left panel
        left_panel = QWidget()
        left_panel.setStyleSheet("background: transparent;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        # Risk gauge
        gauge_group = QGroupBox("Depression Severity Index (DSI)")
        gauge_layout = QVBoxLayout(gauge_group)
        self.risk_gauge = RiskGauge()
        gauge_layout.addWidget(self.risk_gauge)
        left_layout.addWidget(gauge_group)

        # Simulator inputs
        input_group = QGroupBox("New Check-in Simulator")
        input_layout = QVBoxLayout(input_group)
        input_layout.setSpacing(8)

        # Employee ID
        emp_row = QHBoxLayout()
        emp_label = QLabel("Employee ID")
        emp_label.setObjectName("fieldLabel")
        emp_label.setFixedWidth(92)
        emp_row.addWidget(emp_label)
        self.employee_input = QLineEdit()
        self.employee_input.setPlaceholderText("e.g., EMP001")
        emp_row.addWidget(self.employee_input)
        input_layout.addLayout(emp_row)

        # TBS / VBS / ABS with sliders + spinboxes
        self.tbs_input = self._score_row(input_layout, "Text (TBS)", 0.30)
        self.vbs_input = self._score_row(input_layout, "Video (VBS)", 0.35)
        self.abs_input = self._score_row(input_layout, "Audio (ABS)", 0.28)

        # Evaluate button
        self.evaluate_btn = QPushButton("Evaluate Session")
        self.evaluate_btn.setCursor(Qt.PointingHandCursor)
        self.evaluate_btn.clicked.connect(self.evaluate_session)
        input_layout.addWidget(self.evaluate_btn)

        left_layout.addWidget(input_group)

        # Weight breakdown panel
        weight_group = QGroupBox("Dynamic Weight Breakdown")
        weight_layout = QVBoxLayout(weight_group)

        self.text_weight_bar = self._create_weight_bar("Text", "ACCENT")
        self.video_weight_bar = self._create_weight_bar("Video", "GREEN")
        self.audio_weight_bar = self._create_weight_bar("Audio", "YELLOW")

        weight_layout.addWidget(self.text_weight_bar)
        weight_layout.addWidget(self.video_weight_bar)
        weight_layout.addWidget(self.audio_weight_bar)

        left_layout.addWidget(weight_group)

        # Warning banner
        self.warning_banner = QLabel("")
        self.warning_banner.setObjectName("warningBanner")
        self.warning_banner.setWordWrap(True)
        self.warning_banner.hide()
        left_layout.addWidget(self.warning_banner)

        left_layout.addStretch()

        # Right panel: Audit log table
        right_panel = QWidget()
        right_panel.setStyleSheet("background: transparent;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        table_group = QGroupBox("Audit Log  —  latest sessions")
        table_layout = QVBoxLayout(table_group)

        # Employee search bar (filters table by employee ID)
        search_row = QHBoxLayout()
        search_row.setSpacing(8)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍  Search employee  (e.g. EMP001) …")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.textChanged.connect(self._on_search_changed)
        search_row.addWidget(self.search_input, 1)

        # Masking-type filter: All / Traditional / Forced / No masking
        from PySide6.QtWidgets import QComboBox

        self.mask_filter = QComboBox()
        self.mask_filter.addItems(
            ["All sessions", "🎭 Traditional masking", "😐 Forced composure", "✅ No masking"]
        )
        self.mask_filter.setFixedWidth(210)
        self.mask_filter.setCursor(Qt.PointingHandCursor)
        self.mask_filter.currentIndexChanged.connect(lambda _i: self.refresh_sessions())
        search_row.addWidget(self.mask_filter)

        self.search_count = QLabel("")
        self.search_count.setObjectName("appSubtitle")
        search_row.addWidget(self.search_count)
        clear_btn = QPushButton("✕")
        clear_btn.setObjectName("ghost")
        clear_btn.setFixedSize(36, 32)
        clear_btn.setCursor(Qt.PointingHandCursor)
        clear_btn.setToolTip("Clear search + filter")
        clear_btn.clicked.connect(self._clear_filters)
        search_row.addWidget(clear_btn)
        table_layout.addLayout(search_row)

        self.session_table = SessionTable()
        table_layout.addWidget(self.session_table)
        right_layout.addWidget(table_group, 1)

        # 14-day trend chart (title flips company <-> employee)
        trend_group = QGroupBox("Company Trend  —  avg DSI, past 14 days")
        trend_layout = QVBoxLayout(trend_group)
        self.trend_group = trend_group
        self.trend_sub = QLabel("Company-wide daily average DSI")
        self.trend_sub.setObjectName("appSubtitle")
        trend_layout.addWidget(self.trend_sub)
        self.trend_chart = TrendChart()
        trend_layout.addWidget(self.trend_chart)
        right_layout.addWidget(trend_group, 1)

        # Refresh + demo-data buttons
        btn_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setObjectName("ghost")
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.clicked.connect(self.refresh_sessions)
        btn_row.addWidget(refresh_btn)
        self.demo_btn = QPushButton("Load 14-day demo")
        self.demo_btn.setObjectName("ghost")
        self.demo_btn.setCursor(Qt.PointingHandCursor)
        self.demo_btn.setToolTip("Insert demo check-ins across the past 14 days")
        self.demo_btn.clicked.connect(self.load_demo_trend)
        btn_row.addWidget(self.demo_btn)
        right_layout.addLayout(btn_row)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([430, 670])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter, 1)

    def _stat_pill(self, caption: str, value: str, accent: str = "ACCENT") -> QWidget:
        """Small header stat card (QSS-driven, accent = token name)."""
        card = QFrame()
        card.setObjectName("statCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(0)
        cap = QLabel(caption.upper())
        cap.setObjectName("statCap")
        val = QLabel(value)
        val.setObjectName("statVal")
        val.setProperty("accent", accent)
        layout.addWidget(cap)
        layout.addWidget(val)
        card.setFixedWidth(118)
        card._value_label = val  # type: ignore[attr-defined]
        card._accent = accent  # type: ignore[attr-defined]
        return card

    def _score_row(self, parent_layout, label: str, default: float) -> QDoubleSpinBox:
        """Labeled TBS/VBS/ABS row: caption + slider + spinbox synced."""
        from PySide6.QtWidgets import QSlider

        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setObjectName("fieldLabel")
        lbl.setFixedWidth(92)
        row.addWidget(lbl)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(int(default * 100))
        row.addWidget(slider, 1)

        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        spin.setValue(default)
        spin.setFixedWidth(82)
        row.addWidget(spin)

        slider.valueChanged.connect(lambda v: spin.setValue(v / 100.0))
        spin.valueChanged.connect(lambda v: slider.setValue(int(v * 100)))

        parent_layout.addLayout(row)
        return spin

    def _create_weight_bar(self, label: str, accent: str) -> QWidget:
        """Create a labeled progress bar for weight display (QSS-driven)."""
        widget = QWidget()
        widget.setObjectName("weightRow")
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 2, 0, 2)

        label_widget = QLabel(label)
        label_widget.setObjectName("fieldLabelDim")
        label_widget.setFixedWidth(64)
        layout.addWidget(label_widget)

        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(True)
        bar.setFormat("%v%")
        bar.setProperty("accent", accent)
        layout.addWidget(bar)

        # Store reference to the bar
        widget.bar = bar
        widget._accent = accent  # type: ignore[attr-defined]
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
                    f"⚠  {masking['alert_message']}  (Δ={delta:+.2f})"
                )
                self.warning_banner.show()
            else:
                self.warning_banner.hide()

            # Refresh table
            self.refresh_sessions()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def refresh_sessions(self):
        """Refresh the session table (respects search + masking filter)."""
        query = self.search_input.text().strip() if hasattr(self, "search_input") else ""
        mode = self.mask_filter.currentIndex() if hasattr(self, "mask_filter") else 0

        if query:
            sessions = self.db.get_checkins(employee_id=query, limit=500)
        else:
            sessions = self.db.get_checkins(limit=200)

        # Masking filter: 0=All, 1=Traditional, 2=Forced, 3=No masking
        if mode == 1:
            sessions = [s for s in sessions if (s.get("masking_alert") or "").startswith("Traditional")]
        elif mode == 2:
            sessions = [s for s in sessions if (s.get("masking_alert") or "").startswith("Forced")]
        elif mode == 3:
            sessions = [s for s in sessions if not s.get("masking_alert")]

        total = self.db.get_checkins(limit=500)
        filt_names = ["", "🎭 Traditional", "😐 Forced", "✅ Clean"]
        tag = f" • {filt_names[mode]}" if mode else ""
        self.search_count.setText(f"{len(sessions)} / {len(total)} shown{tag}")
        self.session_table.set_sessions(sessions[:200])

        # Update gauge with latest DSI if available
        if sessions:
            self.risk_gauge.set_dsi(sessions[0]["dsi"])

        # Trend chart: employee view when searching, company view otherwise
        try:
            emp = query if query else None
            if emp:
                self.trend_group.setTitle(f"{emp} — personal DSI, past 14 days")
            else:
                self.trend_group.setTitle("Company Trend  —  avg DSI, past 14 days")
            daily = self.db.get_daily_avg_dsi(14, employee_id=emp)
            self.trend_chart.set_data(daily)
            vals = [d["avg_dsi"] for d in daily if d["avg_dsi"] is not None]
            scope = f"{emp} — personal daily DSI" if emp else "Company-wide daily average DSI"
            days_hit = sum(1 for d in daily if d["avg_dsi"] is not None)
            if vals:
                latest = vals[-1]
                lo, hi = min(vals), max(vals)
                self.trend_sub.setText(
                    f"{scope}   •   latest {latest:.3f}   •   "
                    f"{days_hit}/14 days   •   range {lo:.2f}–{hi:.2f}"
                )
            else:
                self.trend_sub.setText(f"{scope}   •   no data yet")
        except Exception:
            pass

        # Update header stat pills
        try:
            stats = self.db.get_summary_stats()
            dist = stats.get("risk_distribution", {})
            self.stat_total._value_label.setText(str(stats.get("total_checkins", 0)))
            self.stat_avg._value_label.setText(f"{stats.get('avg_dsi', 0):.3f}")
            self.stat_red._value_label.setText(str(dist.get("RED", 0)))
            self.stat_mask._value_label.setText(str(stats.get("masking_alerts", 0)))
        except Exception:
            pass

    def _clear_filters(self):
        """Reset search text + masking filter."""
        self.search_input.clear()
        self.mask_filter.setCurrentIndex(0)
        self.refresh_sessions()

    def _on_search_changed(self, text: str):
        """Live-filter the audit table as the user types."""
        self.refresh_sessions()
        # Keep gauge on the filtered employee's latest session
        query = text.strip()
        if query and self.session_table.rowCount() > 0:
            try:
                dsi_text = self.session_table.item(0, 5).text()
                dsi_val = float("".join(c for c in dsi_text if c.isdigit() or c == "."))
                self.risk_gauge.set_dsi(dsi_val)
            except Exception:
                pass

    def load_demo_trend(self):
        """Seed demo check-ins across 14 days, then refresh."""
        try:
            n = self.db.seed_demo_trend_data(14)
            self.demo_btn.setText(f"Demo loaded ({n})")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not seed demo data: {e}")
            return
        self.refresh_sessions()

    def open_add_user(self):
        """Open the Add-New-User form (MongoDB MindSync_DB.users)."""
        try:
            from app.database.mongo_connection import MongoUserManager
            from app.ui.dialogs.add_user_dialog import AddUserDialog

            mongo = MongoUserManager()
        except Exception as e:
            QMessageBox.critical(
                self, "MongoDB Error",
                f"Cannot connect to MongoDB.\nCheck MONGODB_URI in .env\n\n{e}",
            )
            return
        dlg = AddUserDialog(mongo, self)
        dlg.exec()

    def toggle_theme(self):
        """Flip light/dark theme and repaint theme-aware widgets."""
        new_mode = "light" if theme_mode() == "dark" else "dark"
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, new_mode)
        self._refresh_theme_chrome(new_mode)
        self.risk_gauge.update()
        self.refresh_sessions()  # repaints table pills + trend chart

    def _refresh_theme_chrome(self, mode: str | None = None):
        """Refresh dynamic-accent widgets (QSS cannot read runtime props)."""
        mode = mode or theme_mode()

        # Toggle icon: show sun in dark mode, moon in light mode
        self.theme_btn.setText("☀" if mode == "dark" else "☾")

        # Stat values + weight bars use accent token colors
        for pill in (self.stat_total, self.stat_avg, self.stat_red, self.stat_mask):
            val = pill._value_label
            val.setStyleSheet(
                f"color: {theme_get(pill._accent)}; font-size: 18px; "
                "font-weight: 800; background: transparent; border: none;"
            )
        for wrow in (
            self.text_weight_bar,
            self.video_weight_bar,
            self.audio_weight_bar,
        ):
            bar = wrow.bar
            bar.setStyleSheet(
                f"QProgressBar {{ background-color: {theme_get('BG_INPUT')}; "
                f"border: 1px solid {theme_get('BORDER')}; border-radius: 7px; "
                f"text-align: center; color: {theme_get('TEXT')}; }}"
                f"QProgressBar::chunk {{ background-color: {theme_get(wrow._accent)}; "
                "border-radius: 6px; }"
            )


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

def load_theme(app: QApplication) -> str:
    """Apply the persisted theme (dark default). Returns active mode."""
    mode = saved_mode()
    apply_theme(app, mode)
    return mode


def main():
    """Application entry point."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    load_theme(app)

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