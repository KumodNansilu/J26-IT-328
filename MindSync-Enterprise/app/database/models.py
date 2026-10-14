"""DB models — daily VBS summaries ONLY (never videos/frames)."""
from __future__ import annotations

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DailyVideoResult(Base):
    """One row per employee per day: minimal fields for 14-day dashboard + DSI."""
    __tablename__ = "daily_video_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    employee_ref: Mapped[str] = mapped_column(String(64), index=True)
    date: Mapped[str] = mapped_column(String(10), index=True)  # YYYY-MM-DD
    vbs: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32), default="ok")
    model_version: Mapped[str] = mapped_column(String(64), default="vfeba-0.1")
    created_at: Mapped[str] = mapped_column(DateTime, nullable=True)

