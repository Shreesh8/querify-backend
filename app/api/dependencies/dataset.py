"""
db/models/dataset.py

Dataset and related models.
One file per domain — easier to navigate than one giant models.py.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDMixin


class Dataset(Base, UUIDMixin, TimestampMixin):
    """
    Stores metadata about an uploaded file.
    The actual file lives on disk under uploads/.
    The parsed data is re-read from disk on each analytics request
    (stateless — no dataframe stored in memory between requests).
    """
    __tablename__ = "datasets"

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    file_type: Mapped[str] = mapped_column(String(10), nullable=False)   # csv | xlsx

    # Parsed metadata — populated on upload
    row_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    column_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    columns_metadata: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    # e.g. {"col_name": {"dtype": "float64", "null_count": 3, "sample": [1.2, 3.4]}}

    health_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="processing")
    # processing | ready | error

    # Relationships
    chat_messages: Mapped[list["ChatMessage"]] = relationship(back_populates="dataset")
    forecasts: Mapped[list["Forecast"]] = relationship(back_populates="dataset")


class ChatMessage(Base, UUIDMixin, TimestampMixin):
    """One turn of the NL query conversation."""
    __tablename__ = "chat_messages"

    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(10), nullable=False)   # user | assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Structured result from the analytics engine (Plotly JSON + data)
    result_data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    # Operation spec that was executed (audit trail)
    operation_spec: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    execution_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    dataset: Mapped["Dataset"] = relationship(back_populates="chat_messages")


class Forecast(Base, UUIDMixin, TimestampMixin):
    """Stores a forecasting run and its output."""
    __tablename__ = "forecasts"

    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    target_column: Mapped[str] = mapped_column(String(255), nullable=False)
    date_column: Mapped[str] = mapped_column(String(255), nullable=False)
    periods: Mapped[int] = mapped_column(Integer, nullable=False)
    frequency: Mapped[str] = mapped_column(String(10), default="D")   # D | W | M

    # Prophet output stored as JSON (chart-ready)
    forecast_data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    model_metrics: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")

    dataset: Mapped["Dataset"] = relationship(back_populates="forecasts")
