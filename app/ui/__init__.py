"""MindSync theme tokens - single source of truth for light/dark palettes.

Custom-painted widgets (gauge, table) read from here so they follow
the active QSS theme. Qt stylesheets cannot use variables, so
app/utils/theme.py renders the QSS from these tokens.
"""

DARK = {
    "BG_APP": "#0F1115",
    "BG_CARD": "#171B22",
    "BG_CARD_2": "#1D232E",
    "BG_INPUT": "#12151B",
    "BG_ALT_ROW": "#161A21",
    "BORDER": "#2A3140",
    "GRID": "#232A38",
    "TEXT": "#E8ECF1",
    "TEXT_DIM": "#9AA3B2",
    "PLACEHOLDER": "#5C6575",
    "ACCENT": "#4F7CFF",
    "ACCENT_2": "#7C5CFF",
    "GREEN": "#35C26E",
    "YELLOW": "#F5B83D",
    "RED": "#F0524F",
    "TRACK": "#262D3C",
    "WARN_TEXT": "#FF8A80",
}

LIGHT = {
    "BG_APP": "#F2F4F8",
    "BG_CARD": "#FFFFFF",
    "BG_CARD_2": "#EDF0F6",
    "BG_INPUT": "#F6F8FB",
    "BG_ALT_ROW": "#F7F9FC",
    "BORDER": "#DCE2EC",
    "GRID": "#E4E9F1",
    "TEXT": "#1B2432",
    "TEXT_DIM": "#64748B",
    "PLACEHOLDER": "#9AA6B8",
    "ACCENT": "#3B63F2",
    "ACCENT_2": "#7C5CFF",
    "GREEN": "#1FA855",
    "YELLOW": "#C98A12",
    "RED": "#D93B3B",
    "TRACK": "#E0E6EF",
    "WARN_TEXT": "#B4232A",
}

_current = DARK
_mode = "dark"


def set_mode(mode: str) -> dict:
    """Set the active palette ('dark' or 'light'). Returns active tokens."""
    global _current, _mode
    _mode = "light" if mode == "light" else "dark"
    _current = LIGHT if _mode == "light" else DARK
    return _current


def mode() -> str:
    """Return the active mode name."""
    return _mode


def tokens() -> dict:
    """Return the active token dict."""
    return _current


def get(name: str) -> str:
    """Return a single token value from the active palette."""
    return _current[name]

