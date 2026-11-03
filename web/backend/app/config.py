"""
Central settings, loaded from environment variables / a .env file.

Local dev works with zero configuration: DATABASE_URL defaults to a local
SQLite file so `/api/health` and `/api/psf` (neither touches the DB) run
immediately. Persistence-backed routes (simulate/diagnose/gallery) need a
real Postgres URL — set DATABASE_URL to the Supabase connection string, or
run `docker-compose up` which provides one.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", protected_namespaces=()
    )

    database_url: str = "sqlite:///./dev.db"

    supabase_url: str = ""
    supabase_service_key: str = ""
    supabase_bucket: str = "eyesim-cases"

    # Resolved relative to the repo root so it works both from a local
    # checkout and inside the Docker image (see web/backend/Dockerfile).
    model_checkpoint_path: str = str(
        Path(__file__).resolve().parents[3] / "checkpoints" / "best.pt"
    )

    cors_origins: list[str] = ["http://localhost:5173"]


settings = Settings()
