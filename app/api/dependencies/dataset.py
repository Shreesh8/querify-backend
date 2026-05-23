"""
api/dependencies/dataset.py
Reusable dependencies for dataset access.
"""

import asyncio
import uuid
from pathlib import Path
from typing import Tuple

import pandas as pd
from fastapi import Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import get_current_user
from app.db.models.dataset import Dataset
from app.db.session import get_db


async def get_dataset_or_404(
    dataset_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user_id: uuid.UUID = Depends(get_current_user),
) -> Dataset:
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found.")

    if dataset.user_id != current_user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    return dataset


def _read_file(file_path: Path, file_type: str) -> pd.DataFrame:
    if file_type == "csv":
        return pd.read_csv(file_path)
    return pd.read_excel(file_path)


async def load_dataset_df(
    dataset: Dataset = Depends(get_dataset_or_404),
) -> Tuple[Dataset, pd.DataFrame]:
    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found on disk.",
        )

    try:
        df = await asyncio.get_event_loop().run_in_executor(
            None, _read_file, file_path, dataset.file_type
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load dataset file: {str(e)}",
        )

    return dataset, df
