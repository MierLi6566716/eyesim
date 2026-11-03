"""Database tables: saved cases (simulate or diagnose) and network versions."""

from __future__ import annotations

import enum
import secrets
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _slug() -> str:
    return secrets.token_urlsafe(6)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CaseMode(str, enum.Enum):
    simulate = "simulate"
    diagnose = "diagnose"


class ModelVersion(Base):
    """One row per trained PupilNet checkpoint we've deployed.

    Lets every Diagnose-mode prediction record exactly which checkpoint
    produced it, and lets the gallery/stats surface the model's own
    held-out accuracy (see checkpoints/train_log.csv / scripts/evaluate.py)
    next to the live predictions it made.
    """

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tag: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    checkpoint_path: Mapped[str] = mapped_column(String(512))
    val_mae_m: Mapped[float] = mapped_column(Float)
    val_mae_j0: Mapped[float] = mapped_column(Float)
    val_mae_j45: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    cases: Mapped[list["Case"]] = relationship(back_populates="model_version")


class Case(Base):
    """One persisted result: either a Simulate-mode (known Rx -> rendered
    views) or a Diagnose-mode (uploaded photo -> CNN-estimated Rx) case.
    Created only on an explicit user action (upload/save), never on every
    live-slider tick against /api/psf.
    """

    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    mode: Mapped[CaseMode] = mapped_column(Enum(CaseMode), index=True)
    model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True
    )

    source_image_url: Mapped[str] = mapped_column(String(1024))
    blurred_image_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    predistorted_image_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    S: Mapped[float | None] = mapped_column(Float, nullable=True)
    C: Mapped[float | None] = mapped_column(Float, nullable=True)
    axis: Mapped[float | None] = mapped_column(Float, nullable=True)
    M: Mapped[float] = mapped_column(Float)
    J0: Mapped[float] = mapped_column(Float)
    J45: Mapped[float] = mapped_column(Float)

    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    is_public: Mapped[bool] = mapped_column(Boolean, default=True)
    share_slug: Mapped[str] = mapped_column(String(16), unique=True, index=True, default=_slug)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )

    model_version: Mapped["ModelVersion | None"] = relationship(back_populates="cases")
