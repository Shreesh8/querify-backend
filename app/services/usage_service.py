"""
services/usage_service.py
Tracks and enforces usage limits per user per month.
"""
import uuid
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models.dataset import UserUsage, UserLimit
from fastapi import HTTPException, status

FREE_LIMITS = {
    "query": 60,
    "forecast": 10,
    "analytics": 100,
    "dataset": 5,
}

def current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")

async def get_or_create_usage(db: AsyncSession, user_id: uuid.UUID) -> UserUsage:
    month = current_month()
    result = await db.execute(
        select(UserUsage).where(UserUsage.user_id == user_id, UserUsage.month == month)
    )
    usage = result.scalar_one_or_none()
    if not usage:
        usage = UserUsage(user_id=user_id, month=month)
        db.add(usage)
        await db.flush()
    return usage

async def get_limits(db: AsyncSession, user_id: uuid.UUID) -> dict:
    result = await db.execute(select(UserLimit).where(UserLimit.user_id == user_id))
    limit = result.scalar_one_or_none()
    if not limit:
        return FREE_LIMITS.copy()
    return {
        "query": limit.query_limit,
        "forecast": limit.forecast_limit,
        "analytics": limit.analytics_limit,
        "dataset": limit.dataset_limit,
    }

async def check_and_increment(
    db: AsyncSession,
    user_id: uuid.UUID,
    action: str,  # "query" | "forecast" | "analytics" | "dataset"
):
    usage = await get_or_create_usage(db, user_id)
    limits = await get_limits(db, user_id)
    count_field = f"{action}_count"
    current = getattr(usage, count_field, 0)
    limit = limits.get(action, 999)
    if current >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "limit_exceeded",
                "action": action,
                "used": current,
                "limit": limit,
                "month": current_month(),
            }
        )
    setattr(usage, count_field, current + 1)
    await db.flush()

async def get_usage_stats(db: AsyncSession, user_id: uuid.UUID) -> dict:
    usage = await get_or_create_usage(db, user_id)
    limits = await get_limits(db, user_id)
    return {
        "month": current_month(),
        "usage": {
            "query": usage.query_count,
            "forecast": usage.forecast_count,
            "analytics": usage.analytics_count,
            "dataset": usage.dataset_count,
        },
        "limits": limits,
    }
