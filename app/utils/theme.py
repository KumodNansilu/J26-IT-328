"""Theme manager: apply/persist light & dark QSS themes (Qt has no CSS vars)."""

from pathlib import Path

from PySide6.QtCore import QSettings

from app.ui import set_mode, tokens

ORG = "MindSync"
APP = "HRDashboard"
KEY = "theme/mode"


def saved_mode() -> str:
    """Return the persisted theme mode ('dark' default)."""
    try:
        return QSettings(ORG, APP).value(KEY, "dark")
    except Exception:
        return "dark"


def save_mode(mode: str) -> None:
    """Persist the theme mode."""
    try:
        QSettings(ORG, APP).setValue(KEY, mode)
    except Exception:
        pass


def build_stylesheet(mode: str) -> str:
    """Render the full QSS for the given mode from tokens."""
    t = set_mode(mode)
    return f"""# MindSync theme - {mode} mode (generated, do not hand-edit)

QWidget {{
    background-color: {t['BG_APP']};
    color: {t['TEXT']};
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}}

QMainWindow::separator {{ background: {t['BG_APP']}; }}

/* ---------- Cards ---------- */
QGroupBox {{
    background-color: {t['BG_CARD']};
    border: 1px solid {t['BORDER']};
    border-radius: 14px;
    margin-top: 14px;
    padding: 14px;
    font-weight: 600;
    color: {t['TEXT']};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    top: 2px;
    padding: 0 6px;
    color: {t['TEXT_DIM']};
    font-size: 12px;
    font-weight: 600;
    background: transparent;
}}

/* ---------- Inputs ---------- */
QLineEdit, QDoubleSpinBox {{
    background-color: {t['BG_INPUT']};
    border: 1px solid {t['BORDER']};
    border-radius: 8px;
    padding: 7px 10px;
    color: {t['TEXT']};
    selection-background-color: {t['ACCENT']};
}}
QLineEdit:focus, QDoubleSpinBox:focus {{ border: 1px solid {t['ACCENT']}; }}
QLineEdit::placeholder {{ color: {t['PLACEHOLDER']}; }}

/* ---------- Buttons ---------- */
QPushButton {{
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {t['ACCENT']}, stop:1 {t['ACCENT_2']});
    border: none;
    border-radius: 9px;
    padding: 9px 14px;
    color: white;
    font-weight: 700;
}}
QPushButton:hover {{ background-color: {t['ACCENT']}; }}
QPushButton:pressed {{ background-color: {t['ACCENT']}; }}
QPushButton#ghost {{
    background-color: {t['BG_CARD_2']};
    border: 1px solid {t['BORDER']};
    color: {t['TEXT']};
    font-weight: 600;
}}
QPushButton#ghost:hover {{ background-color: {t['BG_INPUT']}; }}

/* ---------- Table ---------- */
QTableWidget {{
    background-color: {t['BG_CARD']};
    alternate-background-color: {t['BG_ALT_ROW']};
    gridline-color: {t['GRID']};
    border: 1px solid {t['BORDER']};
    border-radius: 12px;
    selection-background-color: {t['ACCENT']};
    selection-color: #FFFFFF;
    outline: 0;
}}
QTableWidget::item {{ padding: 6px; border: none; }}
QHeaderView::section {{
    background-color: {t['BG_CARD_2']};
    color: {t['TEXT_DIM']};
    border: none;
    border-bottom: 1px solid {t['BORDER']};
    padding: 8px 6px;
    font-weight: 700;
    font-size: 12px;
}}
QTableCornerButton::section {{ background: {t['BG_CARD_2']}; border: none; }}

/* ---------- Progress bars ---------- */
QProgressBar {{
    background-color: {t['BG_INPUT']};
    border: 1px solid {t['BORDER']};
    border-radius: 7px;
    text-align: center;
    color: {t['TEXT']};
    height: 18px;
}}
QProgressBar::chunk {{ border-radius: 6px; }}

/* ---------- Scrollbars ---------- */
QScrollBar:vertical {{
    background: transparent; width: 10px; margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {t['BORDER']}; border-radius: 5px; min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: transparent; height: 10px; margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: {t['BORDER']}; border-radius: 5px; min-width: 30px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ---------- Splitter ---------- */
QSplitter::handle {{ background: {t['BG_APP']}; width: 8px; }}

/* ---------- Message boxes ---------- */
QMessageBox {{ background-color: {t['BG_CARD']}; }}

/* ---------- Header ---------- */
QLabel#logoDot {{
    color: {t['ACCENT']}; font-size: 26px; background: transparent;
}}
QLabel#appTitle {{
    font-size: 21px; font-weight: 800; background: transparent; padding: 0;
    color: {t['TEXT']};
}}
QLabel#appSubtitle {{
    font-size: 12px; color: {t['TEXT_DIM']}; background: transparent; padding: 0;
}}
QLabel#fieldLabel {{ color: {t['TEXT']}; background: transparent; }}
QLabel#fieldLabelDim {{ color: {t['TEXT_DIM']}; background: transparent; }}
QWidget#weightRow {{ background: transparent; }}
QSlider {{ background: transparent; }}

/* ---------- Stat cards ---------- */
QFrame#statCard {{
    background-color: {t['BG_CARD']};
    border: 1px solid {t['BORDER']};
    border-radius: 12px;
}}
QLabel#statCap {{
    color: {t['TEXT_DIM']}; font-size: 10px; font-weight: 700;
    background: transparent; border: none;
}}
QLabel#statVal {{
    font-size: 18px; font-weight: 800;
    background: transparent; border: none;
}}

/* ---------- Warning banner ---------- */
QLabel#warningBanner {{
    background-color: {t['RED']}24;
    color: {t['WARN_TEXT']};
    font-weight: bold;
    padding: 10px 12px;
    border-radius: 10px;
    border: 1px solid {t['RED']}73;
}}

/* ---------- Login dialog ---------- */
QDialog {{
    background-color: {t['BG_APP']};
}}
QLabel#loginTitle {{
    font-size: 22px; font-weight: 800; color: {t['TEXT']}; padding: 4px;
    background: transparent;
}}
QLabel#loginSub {{
    color: {t['TEXT_DIM']}; font-size: 12px; background: transparent;
}}
QLabel#loginHint {{
    color: {t['PLACEHOLDER']}; font-size: 11px; background: transparent;
}}
QDialog QLabel {{ color: {t['TEXT']}; background: transparent; }}
QDialog QLineEdit {{
    background-color: {t['BG_INPUT']}; border: 1px solid {t['BORDER']};
    border-radius: 8px; padding: 8px 10px; color: {t['TEXT']};
}}
"""


def apply_theme(app, mode: str) -> dict:
    """Apply + persist theme. Returns active tokens for widget repaints."""
    t = set_mode(mode)
    save_mode(mode)
    try:
        app.setStyleSheet(build_stylesheet(mode))
    except Exception as e:
        print(f"Warning: could not apply theme: {e}")
    return tokens()
