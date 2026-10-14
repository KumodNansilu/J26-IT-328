"""DB connection — local SQLite for daily summaries only."""
from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.models import Base

DB_PATH = os.environ.get("MINDSYNC_DB", "mindsync.db")
_engine = create_engine(f"sqlite:///{DB_PATH}")
Session = sessionmaker(bind=_engine)


def init_db() -> None:
    Base.metadata.create_all(_engine)

