"""The web server: sets up the database, mounts the API routes and local image storage."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .db import Base, engine
from . import models  # noqa: F401  (registers ORM models on Base.metadata)
from .routers import diagnose, gallery, psf, simulate
from .services import cnn_service
from .services.storage import LOCAL_STORAGE_DIR


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # SQLite (local dev default) has no migration story worth the ceremony;
    # create tables directly. Postgres (Supabase/docker-compose) is always
    # migrated via Alembic instead — see web/backend/alembic/.
    if settings.database_url.startswith("sqlite"):
        Base.metadata.create_all(bind=engine)

    # Fire-and-forget, not awaited: torch import + checkpoint load takes
    # ~40s on a small shared-cpu machine. Blocking startup on it would make
    # every cold-started request pay that cost, even ones that never touch
    # Diagnose (health/psf/simulate/gallery). Kicking it off in the
    # background lets those respond immediately while the CNN loads in
    # parallel — by the time a user actually navigates to Diagnose it's
    # usually already warm, and worst case it just falls back to the
    # original lazy-load-on-first-request behavior.
    asyncio.create_task(asyncio.to_thread(cnn_service.warm_up))

    yield


app = FastAPI(title="EyeSim Lab API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Local dev only: when Supabase isn't configured, uploaded/rendered images
# are written under _local_storage/ and served from here (see services/storage.py).
app.mount("/static", StaticFiles(directory=LOCAL_STORAGE_DIR), name="static")

app.include_router(psf.router)
app.include_router(simulate.router)
app.include_router(diagnose.router)
app.include_router(gallery.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
