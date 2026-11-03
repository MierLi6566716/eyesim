"""
Loads the network once when the server starts and runs it on uploads.
Uses eyesim.pipeline.load_estimator, which returns a plain
image -> (M, J0, J45) function.
"""

from __future__ import annotations

import numpy as np

from ..config import settings

ACTIVE_MODEL_TAG = "pupilnet-v3"
# Test-set error (simulated photos with a visible crescent), from
# checkpoints/test_report.json.
ACTIVE_MODEL_STATS = {
    "val_mae_m": 0.465,
    "val_mae_j0": 0.079,
    "val_mae_j45": 0.174,
}

_estimator = None


def _get_estimator():
    global _estimator
    if _estimator is None:
        from eyesim.pipeline import load_estimator

        _estimator = load_estimator(settings.model_checkpoint_path)
    return _estimator


def predict(image_2ch: np.ndarray) -> tuple[float, float, float]:
    """image_2ch: (64, 64, 2) float32 in [0, 1]. Returns (M, J0, J45)."""
    return _get_estimator()(image_2ch)


def warm_up() -> None:
    """Load torch + the checkpoint now. Called from app startup (see
    main.py's lifespan) so an already-warm machine never stalls on
    whichever request happens to be the first Diagnose call."""
    _get_estimator()
