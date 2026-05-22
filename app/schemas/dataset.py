"""
schemas/dataset.py
Pydantic v2 schemas — the contract between frontend and backend.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ColumnMeta(BaseModel):
    name: str
    dtype: str
    null_count: int
    null_percent: float
    sample_values: List[Any]
    is_numeric: bool
    is_datetime: bool


class DatasetBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)


class DatasetUploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    name: str
    original_filename: str
    file_size_bytes: int
    file_type: str
    row_count: Optional[int]
    column_count: Optional[int]
    health_score: Optional[float]
    status: str
    created_at: datetime


class DatasetPreview(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    name: str
    row_count: int
    column_count: int
    columns: List[ColumnMeta]
    preview_rows: List[Dict[str, Any]]
    health_score: float


class SummaryStats(BaseModel):
    column: str
    mean: Optional[float]
    median: Optional[float]
    std: Optional[float]
    min: Optional[float]
    max: Optional[float]
    q25: Optional[float]
    q75: Optional[float]


class NullAnalysis(BaseModel):
    column: str
    null_count: int
    null_percent: float


class OutlierInfo(BaseModel):
    column: str
    outlier_count: int
    outlier_values: List[float]


class AnalyticsResponse(BaseModel):
    dataset_id: uuid.UUID
    row_count: int
    column_count: int
    duplicate_count: int
    health_score: float
    summary_stats: List[SummaryStats]
    null_analysis: List[NullAnalysis]
    outliers: List[OutlierInfo]
    correlation_matrix: Optional[Dict[str, Dict[str, float]]]
    top_categories: Dict[str, List[Dict[str, Any]]]
    chart_data: Dict[str, Any]


class ChatQueryRequest(BaseModel):
    dataset_id: uuid.UUID
    question: str = Field(..., min_length=3, max_length=1000)


class ChatQueryResponse(BaseModel):
    message_id: uuid.UUID
    question: str
    answer: str
    result_data: Optional[Dict[str, Any]]
    execution_time_ms: int
    created_at: datetime


class ChatHistoryResponse(BaseModel):
    messages: List[ChatQueryResponse]


class InsightItem(BaseModel):
    category: str
    title: str
    description: str
    severity: str


class InsightsResponse(BaseModel):
    dataset_id: uuid.UUID
    executive_summary: str
    insights: List[InsightItem]
    generated_at: datetime


class ForecastRequest(BaseModel):
    dataset_id: uuid.UUID
    date_column: str
    target_column: str
    periods: int = Field(default=30, ge=7, le=365)
    frequency: str = Field(default="D", pattern="^(D|W|M)$")


class ForecastPoint(BaseModel):
    ds: str
    yhat: float
    yhat_lower: float
    yhat_upper: float


class ForecastResponse(BaseModel):
    forecast_id: uuid.UUID
    dataset_id: uuid.UUID
    target_column: str
    periods: int
    forecast_points: List[ForecastPoint]
    chart_data: Dict[str, Any]
    model_metrics: Dict[str, Any]
    created_at: datetime


class SuccessResponse(BaseModel):
    success: bool = True
    message: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[Any] = None
