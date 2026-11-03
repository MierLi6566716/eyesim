"""Shared upload-image handling for the simulate/diagnose routers."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

MAX_DIM = 640
MAX_UPLOAD_BYTES = 12 * 1024 * 1024  # 12 MB


async def load_upload_as_array(file: UploadFile, *, max_dim: int = MAX_DIM) -> np.ndarray:
    """Read a multipart image upload into a float32 RGB array in [0, 1],
    downscaled so the longer side is at most `max_dim`."""
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large (max 12 MB).")
    try:
        img = Image.open(BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "Could not read file as an image.")

    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

    return np.asarray(img).astype(np.float32) / 255.0


def to_pupilnet_input(arr: np.ndarray, size: int = 64) -> np.ndarray:
    """
    Convert an arbitrary RGB [0, 1] photo into PupilNet's expected
    (size, size, 2) input.

    PupilNet expects a 2-channel eccentric-photorefraction pupil image
    (horizontal-source channel, vertical-source channel — see
    eyesim.photorefraction.render_pair). A single real photo has no
    such structure, so both channels are filled with the same center-cropped,
    grayscale, resized image. This is a documented simplification (see
    services/confidence.py and proposal/), not a claim of equivalence to the
    training distribution — which is exactly why Diagnose mode reports a
    distribution-shift confidence score alongside every prediction.
    """
    h, w = arr.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    cropped = arr[top:top + side, left:left + side]

    img = Image.fromarray((np.clip(cropped, 0, 1) * 255).astype("uint8")).convert("L")
    img = img.resize((size, size), Image.LANCZOS)
    gray = np.asarray(img).astype(np.float32) / 255.0
    return np.stack([gray, gray], axis=-1)
