"""Event log tables (FR-9, FR-10, NFR-9): monitoring sessions and their events.

JSON columns are JSONB on PostgreSQL and plain JSON elsewhere (SQLite in unit tests).
"""

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

Json = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class SessionRow(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    scene: Mapped[str] = mapped_column(String(200))
    scene_sha256: Mapped[str] = mapped_column(String(64))
    params: Mapped[dict] = mapped_column(Json)  # gamma_db, tau, k_max, rho, G, waterfall N
    model: Mapped[dict] = mapped_column(Json)  # commit, sha256 of model.pt, T*, ...
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    end_reason: Mapped[str | None] = mapped_column(String(20))  # finished / stopped / error
    error: Mapped[str | None] = mapped_column(Text)

    events: Mapped[list[EventRow]] = relationship(back_populates="session")


class EventRow(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    track_id: Mapped[int] = mapped_column(Integer)
    detected_at_s: Mapped[float] = mapped_column(Float)  # scene time, s
    decided_at_s: Mapped[float] = mapped_column(Float)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))  # wall clock
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    center_freq_hz: Mapped[float] = mapped_column(Float)
    bandwidth_hz: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(20))  # identified / unknown
    label: Mapped[str | None] = mapped_column(String(40))  # class; NULL for unknown
    confidence: Mapped[float | None] = mapped_column(Float)
    n_frames: Mapped[int] = mapped_column(Integer)
    observation_s: Mapped[float] = mapped_column(Float)
    snr_db: Mapped[float | None] = mapped_column(Float)
    bandwidth_out_of_range: Mapped[bool] = mapped_column(Boolean)

    session: Mapped[SessionRow] = relationship(back_populates="events")


EVENT_COLUMNS = [
    "track_id", "detected_at_s", "decided_at_s", "detected_at", "decided_at",
    "center_freq_hz", "bandwidth_hz", "status", "label", "confidence", "n_frames",
    "observation_s", "snr_db", "bandwidth_out_of_range",
]  # fmt: skip
