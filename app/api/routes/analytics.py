"""api/routes/analytics.py"""

import uuid
from fastapi import APIRouter, Depends
from app.api.dependencies.dataset import load_dataset_df
from app.schemas.dataset import AnalyticsResponse
from app.services.analytics.engine import AnalyticsEngine

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get(
    "/{dataset_id}",
    response_model=AnalyticsResponse,
    summary="Run full analytics on a dataset",
)
async def get_analytics(
    dataset_id: uuid.UUID,
    dataset_and_df=Depends(load_dataset_df),
):
    dataset, df = dataset_and_df
    engine = AnalyticsEngine()
    result = engine.full_analysis(df, str(dataset.id))
    return AnalyticsResponse(dataset_id=dataset.id, **result)
