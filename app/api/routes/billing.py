"""
api/routes/billing.py
Usage stats + admin upgrade endpoint.
"""
import uuid
from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.dependencies.auth import get_current_user
from app.db.session import get_db
from app.db.models.dataset import UserLimit
from app.services.usage_service import get_usage_stats
from app.core.config import settings

router = APIRouter(prefix="/billing", tags=["billing"])

@router.get("/usage")
async def get_my_usage(
    db: AsyncSession = Depends(get_db),
    current_user_id: uuid.UUID = Depends(get_current_user),
):
    return await get_usage_stats(db, current_user_id)

@router.post("/admin/upgrade/{user_id}")
async def admin_upgrade_user(
    user_id: uuid.UUID,
    x_admin_key: str = Header(...),
    query_limit: int = 300,
    forecast_limit: int = 50,
    analytics_limit: int = 500,
    dataset_limit: int = 20,
    notes: str = "Manual upgrade approved",
    db: AsyncSession = Depends(get_db),
):
    if x_admin_key != settings.ADMIN_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key")

    result = await db.execute(select(UserLimit).where(UserLimit.user_id == user_id))
    limit = result.scalar_one_or_none()
    if limit:
        limit.query_limit = query_limit
        limit.forecast_limit = forecast_limit
        limit.analytics_limit = analytics_limit
        limit.dataset_limit = dataset_limit
        limit.plan = "pro"
        limit.notes = notes
    else:
        limit = UserLimit(
            user_id=user_id,
            plan="pro",
            query_limit=query_limit,
            forecast_limit=forecast_limit,
            analytics_limit=analytics_limit,
            dataset_limit=dataset_limit,
            notes=notes,
        )
        db.add(limit)
    await db.commit()
    return {"status": "upgraded", "user_id": str(user_id), "plan": "pro"}
