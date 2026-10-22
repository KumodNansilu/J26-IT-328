
"""Main window for MindSync Enterprise."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QFrame,
)

from app.ui.pages.audio_analysis_page import AudioAnalysisPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("MindSync Enterprise")
        self.resize(1280, 800)
        self.setMinimumSize(950, 600)

        self._build_ui()
        self._apply_styles()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Sidebar
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(220)

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(16, 24, 16, 20)
        sidebar_layout.setSpacing(12)

        brand = QLabel("MindSync")
        brand.setObjectName("BrandTitle")

        enterprise = QLabel("ENTERPRISE")
        enterprise.setObjectName("BrandSubtitle")

        nav_heading = QLabel("WORKSPACE")
        nav_heading.setObjectName("NavHeading")

        sidebar_layout.addWidget(brand)
        sidebar_layout.addWidget(enterprise)
        sidebar_layout.addSpacing(20)
        sidebar_layout.addWidget(nav_heading)

        # Page container
        self.pages = QStackedWidget()
        self.page_widgets = {}

        menu_items = [
            "Overview",
            "Employees",
            "Check-in",
            "Text Analysis",
            "Video Analysis",
            "Audio Analysis",
            "DSI Analytics",
            "Behaviour Trends",
            "Reports",
            "Settings",
        ]

        # Create a page for every sidebar item.
        for name in menu_items:
            if name == "Audio Analysis":
                page = AudioAnalysisPage()
            else:
                page = self._create_placeholder_page(name)

            self.pages.addWidget(page)
            self.page_widgets[name] = page

        # Sidebar buttons
        self.nav_buttons = {}

        for name in menu_items:
            button = QPushButton(name)
            button.setObjectName("NavButton")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setMinimumHeight(38)

            button.clicked.connect(
                lambda checked=False, page_name=name:
                    self._navigate(page_name)
            )

            sidebar_layout.addWidget(button)
            self.nav_buttons[name] = button

        sidebar_layout.addStretch()

        footer = QLabel("Workplace Wellbeing Platform")
        footer.setObjectName("SidebarFooter")
        footer.setWordWrap(True)
        sidebar_layout.addWidget(footer)

        # Main content area
        main_area = QWidget()
        main_layout = QVBoxLayout(main_area)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        topbar = QFrame()
        topbar.setObjectName("TopBar")
        topbar.setMinimumHeight(66)

        topbar_layout = QHBoxLayout(topbar)
        topbar_layout.setContentsMargins(24, 10, 24, 10)

        topbar_title = QLabel("Workplace Wellbeing Dashboard")
        topbar_title.setObjectName("TopBarTitle")

        self.page_status = QLabel("Audio Analysis")
        self.page_status.setObjectName("ConnectionStatus")

        topbar_layout.addWidget(topbar_title)
        topbar_layout.addStretch()
        topbar_layout.addWidget(self.page_status)

        main_layout.addWidget(topbar)
        main_layout.addWidget(self.pages, 1)

        root.addWidget(sidebar)
        root.addWidget(main_area, 1)

        self._navigate("Audio Analysis")

    def _create_placeholder_page(self, page_name):
        """Create a separate placeholder page for an unimplemented section."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(30, 28, 30, 28)
        layout.setSpacing(16)

        title = QLabel(page_name)
        title.setObjectName("PageTitle")

        subtitle = QLabel(
            f"Welcome to the {page_name} section of MindSync Enterprise."
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)

        card = QFrame()
        card.setObjectName("ContentCard")

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(22, 20, 22, 20)
        card_layout.setSpacing(10)

        card_title = QLabel(f"{page_name} Workspace")
        card_title.setObjectName("SectionTitle")

        description = QLabel(
            "This page is ready for its component-specific interface. "
            "Its functionality has not been implemented yet."
        )
        description.setWordWrap(True)

        card_layout.addWidget(card_title)
        card_layout.addWidget(description)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(card)
        layout.addStretch()

        return page

    def _navigate(self, page_name):
        """Switch the central content to the selected sidebar page."""
        page = self.page_widgets.get(page_name)

        if page is None:
            return

        self.pages.setCurrentWidget(page)
        self.page_status.setText(page_name)

        for name, button in self.nav_buttons.items():
            is_active = name == page_name
            button.setProperty("active", is_active)
            button.style().unpolish(button)
            button.style().polish(button)
            button.update()

    def _apply_styles(self):
        """Load the shared Qt stylesheet."""
        stylesheet = (
            Path(__file__).resolve().parents[1]
            / "resources"
            / "styles"
            / "main.qss"
        )

        if stylesheet.exists():
            self.setStyleSheet(
                stylesheet.read_text(encoding="utf-8")
            )
