"""
Distribution-shift confidence score for Diagnose mode.

PupilNet (eyesim.cnn) was trained only on simulated infrared pupil photos
(eyesim.photorefraction). An ordinary uploaded photo looks very different
from those, so the app says how different.

Rather than pretend a black-box confidence model, this computes five
simple, inspectable statistics (mean, std, edge/texture energy via the same
Laplacian sharpness metric eyesim.optimize.sharpness uses for the closed-
loop optimizer, bright-pixel fraction, and corner brightness) and reports
how many standard deviations the input sits from the synthetic training
distribution on each one (see train_stats.json, precomputed by
scripts/compute_train_stats.py). This is deliberately legible over accurate
— the point is to be honest about what it does and doesn't detect, not to
build a second undocumented model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eyesim.optimize import sharpness

FEATURE_NAMES = ["mean", "std", "sharpness", "bright_frac", "corner_mean"]
_CORNER = 8

_STATS_PATH = Path(__file__).resolve().parent / "train_stats.json"

# Loaded lazily (not at import time) so scripts/compute_train_stats.py can
# import `image_features` from this module even before train_stats.json
# exists yet (it's the thing that script generates).
_MU: np.ndarray | None = None
_SIGMA: np.ndarray | None = None


def _load_reference() -> tuple[np.ndarray, np.ndarray]:
    global _MU, _SIGMA
    if _MU is None:
        if not _STATS_PATH.exists():
            raise RuntimeError(
                f"{_STATS_PATH} missing — run "
                "`python scripts/compute_train_stats.py` from web/backend/ first."
            )
        stats = json.loads(_STATS_PATH.read_text())
        _MU, _SIGMA = np.array(stats["mu"]), np.array(stats["sigma"])
    return _MU, _SIGMA

# Confidence = 100 * exp(-rms_zscore / _SCALE). At distance 0 -> 100,
# distance ~2.5 -> ~37, distance ~5 -> ~14. Tuned by eye against a handful
# of real vs. synthetic examples, not fit to data — see module docstring.
_SCALE = 2.5


def image_features(img: np.ndarray) -> np.ndarray:
    """img: (H, W, 2) float32 in [0, 1]. Returns the 5-d feature vector."""
    flat = img.reshape(-1)
    mean = flat.mean()
    std = flat.std()
    sharp = np.mean([sharpness(img[..., c]) for c in range(img.shape[-1])])
    bright_frac = (flat > 0.5).mean()
    corners = np.concatenate([
        img[:_CORNER, :_CORNER].reshape(-1), img[:_CORNER, -_CORNER:].reshape(-1),
        img[-_CORNER:, :_CORNER].reshape(-1), img[-_CORNER:, -_CORNER:].reshape(-1),
    ])
    return np.array([mean, std, sharp, bright_frac, corners.mean()])


def score(img: np.ndarray) -> tuple[float, dict[str, float]]:
    """Returns (confidence in [0, 100], per-feature z-scores for a 'why'
    tooltip in the UI)."""
    mu, sigma = _load_reference()
    feats = image_features(img)
    z = (feats - mu) / sigma
    rms_z = float(np.sqrt(np.mean(z**2)))
    confidence = float(np.clip(100.0 * np.exp(-rms_z / _SCALE), 0.0, 100.0))
    detail = {name: round(float(zi), 2) for name, zi in zip(FEATURE_NAMES, z)}
    return round(confidence, 1), detail
