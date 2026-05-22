"""
db/session.py

Async SQLAlchemy engine and session factory.

AsyncSession is injected into routes via the get_db dependency.
One session per request, closed automatically via the context manager.
Connection pooling is handled by asyncpg under the hood.
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings

# ── Engine ────────────────────────────────────────────────────
# pool_pre_ping=True: test connections on checkout → avoids stale connection errors
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,           # logs SQL in debug mode
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

# ── Session factory ───────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,        # objects stay accessible after commit
    autoflush=False,
)


# ── Dependency ────────────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency — yields one session per request.

    Usage in a route:
        async def my_route(db: AsyncSession = Depends(get_db)):
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
