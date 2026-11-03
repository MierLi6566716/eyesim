"""
Image persistence layer with two backends:

- **Supabase Storage** — used whenever SUPABASE_URL / SUPABASE_SERVICE_KEY
  are configured (staging/production).
- **Local filesystem** under `./_local_storage`, served via FastAPI's
  StaticFiles mount — used automatically otherwise, so the API is fully
  testable without cloud credentials. Callers never need to know which
  backend is active; both return a URL that resolves to a working image.
"""

from __future__ import annotations

import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import settings

LOCAL_STORAGE_DIR = Path(__file__).resolve().parents[2] / "_local_storage"
LOCAL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def array_to_png_bytes(arr: np.ndarray) -> bytes:
    img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype("uint8"))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _using_supabase() -> bool:
    return bool(settings.supabase_url and settings.supabase_service_key)


def upload_array(arr: np.ndarray, *, prefix: str) -> str:
    """Upload a float [0,1] (grayscale or RGB) image array as a PNG."""
    return upload_bytes(array_to_png_bytes(arr), prefix=prefix)


def upload_bytes(data: bytes, *, prefix: str) -> str:
    key = f"{prefix}/{uuid.uuid4().hex}.png"

    if _using_supabase():
        from supabase import create_client

        client = create_client(settings.supabase_url, settings.supabase_service_key)
        client.storage.from_(settings.supabase_bucket).upload(
            key, data, {"content-type": "image/png"}
        )
        return client.storage.from_(settings.supabase_bucket).get_public_url(key)

    path = LOCAL_STORAGE_DIR / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return f"/static/{key}"
