"""
services/datasets/upload_service.py

Handles file upload, validation, parsing, and metadata extraction.

Design decisions:
- Validation happens before any DB write
- File is saved with a UUID filename to prevent collisions and path traversal
- Parsing is synchronous Pandas but called inside run_in_executor
  so it doesn't block the async event loop
"""

import asyncio
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple

import aiofiles
import pandas as pd
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import (
    DatasetUploadError,
    FileTooLargeError,
    InvalidFileTypeError,
)
from app.core.logging import get_logger
from app.db.models.dataset import Dataset
from app.services.analytics.engine import AnalyticsEngine

logger = get_logger(__name__)


class DatasetUploadService:

    def __init__(self, db: AsyncSession):
        self.db = db
        self.analytics = AnalyticsEngine()

    async def process_upload(
        self,
        file: UploadFile,
        user_id: uuid.UUID,
        display_name: str,
    ) -> Dataset:
        """
        Full upload pipeline:
        1. Validate → 2. Save to disk → 3. Parse → 4. Extract metadata → 5. Store in DB
        """
        # Step 1 — validate
        file_ext = self._validate_file(file)

        # Step 2 — read content and check size
        content = await file.read()
        if len(content) > settings.max_file_size_bytes:
            raise FileTooLargeError(
                f"File exceeds {settings.MAX_FILE_SIZE_MB}MB limit.",
                detail={"size_bytes": len(content)},
            )

        # Step 3 — save to disk with UUID name
        file_id = uuid.uuid4()
        safe_filename = f"{file_id}.{file_ext}"
        file_path = settings.upload_path / safe_filename

        await self._save_file(file_path, content)
        logger.info("file_saved", path=str(file_path), size=len(content))

        # Step 4 — parse in thread pool (CPU-bound Pandas work)
        try:
            df, columns_meta = await asyncio.get_event_loop().run_in_executor(
                None, self._parse_and_extract, file_path, file_ext
            )
        except Exception as e:
            # Clean up orphaned file if parsing fails
            file_path.unlink(missing_ok=True)
            raise DatasetUploadError(f"Failed to parse file: {str(e)}")

        # Step 5 — compute health score
        health_score = self.analytics.compute_health_score(df)

        # Step 6 — persist metadata
        dataset = Dataset(
            id=file_id,
            user_id=user_id,
            name=display_name,
            original_filename=file.filename or safe_filename,
            file_path=str(file_path),
            file_size_bytes=len(content),
            file_type=file_ext,
            row_count=len(df),
            column_count=len(df.columns),
            columns_metadata=columns_meta,
            health_score=health_score,
            status="ready",
        )
        self.db.add(dataset)
        await self.db.flush()   # get ID without committing (session commits in get_db)

        logger.info(
            "dataset_uploaded",
            dataset_id=str(file_id),
            rows=len(df),
            cols=len(df.columns),
        )
        return dataset

    # ── Private helpers ───────────────────────────────────────

    def _validate_file(self, file: UploadFile) -> str:
        if not file.filename:
            raise InvalidFileTypeError("Filename is missing.")
        ext = Path(file.filename).suffix.lstrip(".").lower()
        if ext not in settings.allowed_extensions:
            raise InvalidFileTypeError(
                f"File type '.{ext}' not allowed.",
                detail={"allowed": settings.allowed_extensions},
            )
        return ext

    @staticmethod
    async def _save_file(path: Path, content: bytes) -> None:
        async with aiofiles.open(path, "wb") as f:
            await f.write(content)

    @staticmethod
    def _parse_and_extract(
        file_path: Path, file_ext: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Synchronous — always called via run_in_executor."""
        if file_ext == "csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        # Standardise column names
        df.columns = [str(c).strip() for c in df.columns]

        # Build column metadata
        columns_meta: Dict[str, Any] = {}
        for col in df.columns:
            series = df[col]
            null_count = int(series.isna().sum())
            columns_meta[col] = {
                "dtype": str(series.dtype),
                "null_count": null_count,
                "null_percent": round(null_count / len(df) * 100, 2) if len(df) > 0 else 0,
                "is_numeric": pd.api.types.is_numeric_dtype(series),
                "is_datetime": pd.api.types.is_datetime64_any_dtype(series),
                "sample_values": series.dropna().head(5).tolist(),
            }

        return df, columns_meta
