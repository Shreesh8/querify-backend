"""api/routes/forecast.py"""

import uuid
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import get_current_user
from app.api.dependencies.dataset import load_dataset_df
from app.db.models.dataset import Forecast
from app.db.session import get_db
from app.schemas.dataset import ForecastRequest, ForecastResponse
from app.services.forecasting.forecast_service import ForecastService

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.post(
    "/generate",
    response_model=ForecastResponse,
    summary="Generate a time-series forecast using Prophet",
)
async def generate_forecast(
    request: ForecastRequest,
    db: AsyncSession = Depends(get_db),
    current_user_id: uuid.UUID = Depends(get_current_user),
):
    from pathlib import Path
    import asyncio
    from sqlalchemy import select
    from app.db.models.dataset import Dataset
    from app.api.dependencies.dataset import _read_file

    result = await db.execute(select(Dataset).where(Dataset.id == request.dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        from fastapi import HTTPException, status
        raise HTTPException(status_code=404, detail="Dataset not found.")

    df = await asyncio.get_event_loop().run_in_executor(
        None, _read_file, Path(dataset.file_path), dataset.file_type
    )

    service = ForecastService()
    forecast_result = await service.generate_forecast(
        df=df,
        date_column=request.date_column,
        target_column=request.target_column,
        periods=request.periods,
        frequency=request.frequency,
        dataset_id=str(request.dataset_id),
    )

    # Persist forecast record
    forecast_record = Forecast(
        dataset_id=request.dataset_id,
        user_id=current_user_id,
        target_column=request.target_column,
        date_column=request.date_column,
        periods=request.periods,
        frequency=request.frequency,
        forecast_data=forecast_result,
        model_metrics=forecast_result["model_metrics"],
        status="complete",
    )
    db.add(forecast_record)
    await db.flush()

    from datetime import datetime
    return ForecastResponse(
        forecast_id=forecast_record.id,
        dataset_id=request.dataset_id,
        target_column=request.target_column,
        periods=request.periods,
        forecast_points=forecast_result["forecast_points"],
        chart_data=forecast_result["chart_data"],
        model_metrics=forecast_result["model_metrics"],
        created_at=forecast_record.created_at,
    )
