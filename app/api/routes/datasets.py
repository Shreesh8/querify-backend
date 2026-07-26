"""
api/routes/datasets.py

Dataset management routes — thin layer that delegates to services.
No business logic here. No Pandas here. No AI calls here.
"""

import uuid
from typing import List

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import get_current_user
from app.api.dependencies.dataset import get_dataset_or_404, load_dataset_df
from app.db.models.dataset import Dataset
from app.db.session import get_db
from app.schemas.dataset import DatasetPreview, DatasetUploadResponse
from app.services.analytics.engine import AnalyticsEngine
from app.services.datasets.upload_service import DatasetUploadService

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post(
    "/upload",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a CSV or Excel dataset",
)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    db: AsyncSession = Depends(get_db),
    current_user_id: uuid.UUID = Depends(get_current_user),
):
    service = DatasetUploadService(db)
    dataset = await service.process_upload(file, current_user_id, name)
    return DatasetUploadResponse.model_validate(dataset)


@router.get(
    "/",
    response_model=List[DatasetUploadResponse],
    summary="List all datasets for the authenticated user",
)
async def list_datasets(
    db: AsyncSession = Depends(get_db),
    current_user_id: uuid.UUID = Depends(get_current_user),
):
    result = await db.execute(
        select(Dataset)
        .where(Dataset.user_id == current_user_id)
        .order_by(Dataset.created_at.desc())
    )
    datasets = result.scalars().all()
    return [DatasetUploadResponse.model_validate(d) for d in datasets]


@router.get(
    "/{dataset_id}",
    response_model=DatasetUploadResponse,
    summary="Get dataset metadata",
)
async def get_dataset(
    dataset: Dataset = Depends(get_dataset_or_404),
):
    return DatasetUploadResponse.model_validate(dataset)


@router.get(
    "/{dataset_id}/preview",
    response_model=DatasetPreview,
    summary="Get dataset preview with column metadata and sample rows",
)
async def preview_dataset(
    dataset_and_df=Depends(load_dataset_df),
):
    dataset, df = dataset_and_df
    columns = []
    for col in df.columns:
        s = df[col]
        columns.append({
            "name": col,
            "dtype": str(s.dtype),
            "null_count": int(s.isna().sum()),
            "null_percent": round(s.isna().mean() * 100, 2),
            "sample_values": s.dropna().head(5).tolist(),
            "is_numeric": bool(s.dtype in ["int64", "float64"]),
            "is_datetime": bool(hasattr(s, "dt")),
        })
    return DatasetPreview(
        id=dataset.id,
        name=dataset.name,
        row_count=len(df),
        column_count=len(df.columns),
        columns=columns,
        preview_rows=df.head(10).fillna("").to_dict(orient="records"),
        health_score=dataset.health_score or 0.0,
    )


@router.delete(
    "/{dataset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a dataset and its file",
)
async def delete_dataset(
    dataset: Dataset = Depends(get_dataset_or_404),
    db: AsyncSession = Depends(get_db),
):
    from pathlib import Path
    from sqlalchemy import delete as sql_delete
    from app.db.models.dataset import ChatMessage, Forecast
    # Delete all related records first (dataset_id is NOT NULL on both)
    await db.execute(sql_delete(ChatMessage).where(ChatMessage.dataset_id == dataset.id))
    await db.execute(sql_delete(Forecast).where(Forecast.dataset_id == dataset.id))
    Path(dataset.file_path).unlink(missing_ok=True)
    await db.delete(dataset)
    await db.commit()
