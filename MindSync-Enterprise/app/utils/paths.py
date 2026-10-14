"""Path helpers — temp / app-data locations (Windows-safe).

We never store original uploads in the DB. Webcam recordings go to the
OS temp dir so they can be deleted after processing / cancel.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def get_temp_dir() -> Path:
    d = Path(tempfile.gettempdir()) / "mindsync_vfeba"
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_temp_video_path(suffix: str = ".mp4") -> Path:
    """Unique temp recording path (caller deletes it after use)."""
    import uuid

    return get_temp_dir() / f"webcam_{uuid.uuid4().hex}{suffix}"


def is_temp_mindsync_path(path: str | os.PathLike) -> bool:
    """True only for files we created in our temp dir (safety guard)."""
    try:
        return Path(path).resolve().is_relative_to(get_temp_dir().resolve())
    except Exception:
        # Python < 3.9 fallback
        try:
            Path(path).resolve().relative_to(get_temp_dir().resolve())
            return True
        except Exception:
            return False

