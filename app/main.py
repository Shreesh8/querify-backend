"""
main.py

FastAPI application factory.

Everything is registered here:
- Lifespan (startup/shutdown)
- CORS middleware
- Exception handlers (maps custom exceptions → HTTP status codes)
- API v1 router with all sub-routes
- Health check endpoint
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import analytics, chat, datasets, forecast, insights
from app.core.config import settings
from app.core.exceptions import (
    AIServiceError,
    AnalyticsError,
    AuthenticationError,
    AuthorizationError,
    DatasetNotFoundError,
    DatasetUploadError,
    FileTooLargeError,
    ForecastError,
    InsightEngineError,
    InvalidFileTypeError,
    UnsafeQueryError,
)
from app.core.logging import get_logger, setup_logging
from app.db.session import engine
from app.db.base import Base

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown logic.
    Using lifespan instead of deprecated on_event decorators.
    """
    # ── Startup ───────────────────────────────────────────────
    setup_logging()
    logger.info("insight_engine_starting", env=settings.APP_ENV, version=settings.APP_VERSION)

    # Ensure upload directory exists
    settings.upload_path.mkdir(parents=True, exist_ok=True)

    # Create tables (use Alembic for production migrations)
    if settings.APP_ENV == "development":
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    logger.info("insight_engine_ready")
    yield

    # ── Shutdown ──────────────────────────────────────────────
    await engine.dispose()
    logger.info("insight_engine_shutdown")


# ── Application factory ───────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="AI-powered Business Intelligence platform",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception handlers ────────────────────────────────────
    # Maps every custom exception → exactly one HTTP status code
    # No HTTPException ever leaks from service layer

    @app.exception_handler(DatasetNotFoundError)
    async def dataset_not_found(_: Request, exc: DatasetNotFoundError):
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND,
                            content={"success": False, "error": exc.message})

    @app.exception_handler(InvalidFileTypeError)
    async def invalid_file_type(_: Request, exc: InvalidFileTypeError):
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            content={"success": False, "error": exc.message, "detail": exc.detail})

    @app.exception_handler(FileTooLargeError)
    async def file_too_large(_: Request, exc: FileTooLargeError):
        return JSONResponse(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            content={"success": False, "error": exc.message})

    @app.exception_handler(DatasetUploadError)
    async def upload_error(_: Request, exc: DatasetUploadError):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
                            content={"success": False, "error": exc.message})

    @app.exception_handler(UnsafeQueryError)
    async def unsafe_query(_: Request, exc: UnsafeQueryError):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
                            content={"success": False, "error": exc.message})

    @app.exception_handler(AIServiceError)
    async def ai_error(_: Request, exc: AIServiceError):
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            content={"success": False, "error": exc.message})

    @app.exception_handler(ForecastError)
    async def forecast_error(_: Request, exc: ForecastError):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
                            content={"success": False, "error": exc.message})

    @app.exception_handler(AuthenticationError)
    async def auth_error(_: Request, exc: AuthenticationError):
        return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED,
                            content={"success": False, "error": exc.message})

    @app.exception_handler(AuthorizationError)
    async def authz_error(_: Request, exc: AuthorizationError):
        return JSONResponse(status_code=status.HTTP_403_FORBIDDEN,
                            content={"success": False, "error": exc.message})

    @app.exception_handler(InsightEngineError)
    async def generic_app_error(_: Request, exc: InsightEngineError):
        logger.error("unhandled_app_error", error=exc.message)
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            content={"success": False, "error": exc.message})

    # ── Routers ───────────────────────────────────────────────
    # All routes live under /api/v1 — ready for v2 without breaking changes
    prefix = "/api/v1"
    app.include_router(datasets.router, prefix=prefix)
    app.include_router(analytics.router, prefix=prefix)
    app.include_router(chat.router, prefix=prefix)
    app.include_router(insights.router, prefix=prefix)
    app.include_router(forecast.router, prefix=prefix)

    # ── Health check ──────────────────────────────────────────
    @app.get("/health", tags=["system"])
    async def health_check():
        return {
            "status": "healthy",
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "env": settings.APP_ENV,
        }

    return app


app = create_app()


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_config=None,   # structlog handles logging
    )
