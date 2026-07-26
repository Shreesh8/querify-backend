"""
core/config.py

Central settings loaded from environment variables.
Using Pydantic BaseSettings means every value is typed, validated,
and IDE-autocompleted. No raw os.getenv() calls anywhere else.
"""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Application ───────────────────────────────────────────
    APP_NAME: str = "Insight Engine"
    APP_ENV: str = "development"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    SECRET_KEY: str

    # ── Server ────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── Database ──────────────────────────────────────────────
    DATABASE_URL: str         # async URL  (asyncpg)
    DATABASE_URL_SYNC: str    # sync URL   (alembic migrations)

    # ── AI ────────────────────────────────────────────────────
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    ADMIN_SECRET_KEY: str = "querify-admin-secret"

    # ── File uploads ──────────────────────────────────────────
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_FILE_TYPES: str = "csv,xlsx,xls"

    # ── CORS ──────────────────────────────────────────────────
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    # ── Security ──────────────────────────────────────────────
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ALGORITHM: str = "HS256"

    # ── Logging ───────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    # ── Computed properties ───────────────────────────────────
    @property
    def upload_path(self) -> Path:
        path = Path(self.UPLOAD_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    @property
    def allowed_extensions(self) -> List[str]:
        return [ext.strip().lower() for ext in self.ALLOWED_FILE_TYPES.split(",")]

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings instance — lru_cache means this is only
    instantiated once per process, not on every request.
    """
    return Settings()


# Module-level singleton for import convenience
settings = get_settings()
