"""api/routes/insights.py"""

import uuid
from fastapi import APIRouter, Depends
from app.api.dependencies.dataset import load_dataset_df
from app.schemas.dataset import InsightsResponse
from app.services.analytics.engine import AnalyticsEngine
from app.services.ai.insights_service import InsightsService

router = APIRouter(prefix="/insights", tags=["insights"])


@router.get(
    "/{dataset_id}",
    response_model=InsightsResponse,
    summary="Generate AI business insights for a dataset",
)
async def get_insights(
    dataset_id: uuid.UUID,
    dataset_and_df=Depends(load_dataset_df),
):
    dataset, df = dataset_and_df
    analytics_data = AnalyticsEngine().full_analysis(df, str(dataset.id))
    service = InsightsService()
    result = await service.generate_insights(analytics_data, str(dataset.id), dataset.name)
    from datetime import datetime
    result["generated_at"] = datetime.fromisoformat(result["generated_at"])
    return InsightsResponse(**result)
