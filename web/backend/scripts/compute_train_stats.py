"""
One-off precomputation: summary statistics of the synthetic training
distribution, used by app/services/confidence.py to score how far a real
uploaded photo (Diagnose mode) sits from what PupilNet was actually trained
on.

Reads data/train_set/train_images.npy (made by scripts/generate_dataset.py) and writes a tiny
JSON of feature means/stds to app/services/train_stats.json (checked into
git — this is the only artifact the deployed app needs). Re-run this
whenever PupilNet is retrained on a new/expanded dataset.

Usage: python scripts/compute_train_stats.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))       # for `import eyesim`
sys.path.insert(0, str(BACKEND_ROOT))    # for `import app`

from app.services.confidence import FEATURE_NAMES, image_features as features  # noqa: E402

TRAIN_IMAGES = REPO_ROOT / "data" / "train_set" / "train_images.npy"
OUT_PATH = BACKEND_ROOT / "app" / "services" / "train_stats.json"
N_SAMPLE = 4000


def main() -> None:
    if not TRAIN_IMAGES.exists():
        raise SystemExit(f"{TRAIN_IMAGES} not found — run scripts/generate_dataset.py first.")

    images = np.load(TRAIN_IMAGES, mmap_mode="r")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(images), size=min(N_SAMPLE, len(images)), replace=False)

    feats = np.stack([features(images[i].astype(np.float32) / 255.0) for i in idx])
    mu = feats.mean(axis=0)
    sigma = feats.std(axis=0) + 1e-6  # avoid div-by-zero downstream

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "feature_names": FEATURE_NAMES,
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "n_samples": int(len(idx)),
    }, indent=2))
    print(f"Wrote {OUT_PATH}")
    for name, m, s in zip(FEATURE_NAMES, mu, sigma):
        print(f"  {name:12s} mu={m:.4f}  sigma={s:.4f}")


if __name__ == "__main__":
    main()
