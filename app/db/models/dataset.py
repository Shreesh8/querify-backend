"""
db/models/dataset.py
SQLAlchemy ORM models.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDMixin


class Dataset(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "datasets"

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    file_type: Mapped[str] = mapped_column(String(10), nullable=False)

    row_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    column_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    columns_metadata: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    health_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="processing")

    chat_messages: Mapped[list["ChatMessage"]] = relationship(back_populates="dataset")
    forecasts: Mapped[list["Forecast"]] = relationship(back_populates="dataset")


class ChatMessage(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "chat_messages"

    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(10), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    result_data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    operation_spec: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    execution_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    dataset: Mapped["Dataset"] = relationship(back_populates="chat_messages")


class Forecast(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "forecasts"

    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    target_column: Mapped[str] = mapped_column(String(255), nullable=False)
    date_column: Mapped[str] = mapped_column(String(255), nullable=False)
    periods: Mapped[int] = mapped_column(Integer, nullable=False)
    frequency: Mapped[str] = mapped_column(String(10), default="D")
    forecast_data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    model_metrics: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")

    dataset: Mapped["Dataset"] = relationship(back_populates="forecasts")
