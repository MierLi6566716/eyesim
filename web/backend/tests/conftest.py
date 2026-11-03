"""Shared test setup: a temporary database, a test client, and sample images."""

from __future__ import annotations

import os
import sys
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Must happen before `app.config` is imported anywhere (including
# transitively via app.main) so the test suite never touches dev.db/a real
# Postgres instance, or (critically) the real Supabase Storage bucket: if a
# real web/backend/.env exists (e.g. for local dev against a deployed
# Supabase project), pydantic-settings would otherwise fall back to its
# real SUPABASE_URL/SUPABASE_SERVICE_KEY for any var not explicitly
# overridden here, silently uploading every test-fixture image into
# production storage on every local test run.
_TMP_DIR = Path(tempfile.mkdtemp())
_TMP_DB = _TMP_DIR / "test.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB}"
os.environ["SUPABASE_URL"] = ""
os.environ["SUPABASE_SERVICE_KEY"] = ""

# Repo root, for `import eyesim` when tests run from web/backend/.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi.testclient import TestClient  # noqa: E402

from app import models  # noqa: E402,F401
from app.db import Base, engine  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _tables():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    return TestClient(app)


def _png_bytes(arr: np.ndarray) -> bytes:
    img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype("uint8"))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_image_bytes() -> bytes:
    """A generic isotropic test image, standing in for 'any uploaded photo'."""
    n = 128
    x = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(x, x)
    img = 0.5 + 0.5 * np.sin(10 * xx) * np.cos(8 * yy)
    return _png_bytes(np.stack([img] * 3, axis=-1))


@pytest.fixture
def synth_crescent_bytes() -> bytes:
    """A simulated infrared pupil photo: should score higher confidence than noise."""
    from eyesim.photorefraction import Eye, Scene, render_pair

    img = render_pair(Eye(M=-2.0, J0=0.2, J45=0.2), scene=Scene(glint_amp=1.0), rng=np.random.default_rng(0))
    return _png_bytes(np.stack([img[..., 0]] * 3, axis=-1))


@pytest.fixture
def noise_image_bytes() -> bytes:
    """Unstructured, full-frame noise — nothing like the synthetic training
    distribution (no zeroed corners, no smooth crescent). Should score low."""
    rng = np.random.default_rng(1)
    return _png_bytes(rng.uniform(0.1, 0.95, (128, 128, 3)).astype(np.float32))
