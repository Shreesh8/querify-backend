"""
core/exceptions.py

Custom exception hierarchy.

Design rule: every raised exception in services/utils maps to
exactly one HTTP status code via the handlers registered in main.py.
No HTTPException leaks into service layer — that stays in routes only.
"""

from typing import Any, Optional


class InsightEngineError(Exception):
    """Base for all app exceptions."""
    def __init__(self, message: str, detail: Optional[Any] = None):
        self.message = message
        self.detail = detail
        super().__init__(message)


# ── Dataset errors ────────────────────────────────────────────
class DatasetNotFoundError(InsightEngineError):
    """Raised when a dataset ID doesn't exist in the DB."""


class DatasetUploadError(InsightEngineError):
    """Raised on file parse / validation failure."""


class InvalidFileTypeError(InsightEngineError):
    """Raised when uploaded file extension is not allowed."""


class FileTooLargeError(InsightEngineError):
    """Raised when uploaded file exceeds MAX_FILE_SIZE_MB."""


# ── Analytics errors ──────────────────────────────────────────
class AnalyticsError(InsightEngineError):
    """Raised when Pandas analytics computation fails."""


class InsufficientDataError(InsightEngineError):
    """Raised when the dataset has too few rows to compute something."""


# ── AI / query errors ─────────────────────────────────────────
class AIServiceError(InsightEngineError):
    """Raised when Gemini API call fails or returns unexpected output."""


class UnsafeQueryError(InsightEngineError):
    """Raised when a query operation spec fails security validation."""


class QueryExecutionError(InsightEngineError):
    """Raised when a validated operation spec fails during execution."""


# ── Forecasting errors ────────────────────────────────────────
class ForecastError(InsightEngineError):
    """Raised when Prophet / sklearn forecasting fails."""


class NoTimeseriesColumnError(InsightEngineError):
    """Raised when dataset has no parseable date column."""


# ── Auth errors ───────────────────────────────────────────────
class AuthenticationError(InsightEngineError):
    """Raised on invalid credentials."""


class AuthorizationError(InsightEngineError):
    """Raised when user tries to access a resource they don't own."""
