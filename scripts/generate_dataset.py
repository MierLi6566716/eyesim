"""
Generate synthetic eccentric photorefraction dataset for CNN training.

Samples prescriptions from a population-realistic distribution (see
eyesim.photorefraction.sample_prescription) and renders 2-channel
photorefraction images with realistic noise and hardware variation.

Domain randomisation (enabled by --augment) varies camera distance,
source eccentricity, and pupil radius within physical tolerances to
reduce the sim-to-real gap — essential for deployment on real hardware.

Output files
------------
  data/train.npz   — 80 % of samples
  data/val.npz     — 10 %
  data/test.npz    — 10 %

Each .npz contains:
  images : float32  (N, 64, 64, 2)   — 2-channel pupil images
  labels : float32  (N, 3)           — [M, J0, J45] in dioptres

Usage
-----
  python scripts/generate_dataset.py --n 100000 --augment --seed 42
  python scripts/generate_dataset.py --n 5000                         # quick test
"""

import argparse
import os
import numpy as np
from pathlib import Path

# Add project root to path when run as script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eyesim.photorefraction import (
    render_multichannel,
    sample_prescription,
    prescription_to_power_vector,
)


def generate(
    n: int,
    image_size: int = 64,
    noise_std: float = 0.02,
    augment: bool = False,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate N labelled photorefraction images.

    Returns
    -------
    images : float32  (N, H, W, 2)
    labels : float32  (N, 3)   — [M, J0, J45]
    """
    rng = np.random.default_rng(seed)
    images = np.empty((n, image_size, image_size, 2), dtype=np.float32)
    labels = np.empty((n, 3), dtype=np.float32)

    for i in range(n):
        S, C, axis = sample_prescription(rng)

        if augment:
            # Randomise hardware geometry within realistic tolerances
            D_mm        = float(rng.uniform(900.0, 1100.0))   # ±100 mm at 1 m
            ecc_mm      = float(rng.uniform(8.0,   12.0))     # ±2 mm eccentricity
            pupil_r_mm  = float(rng.uniform(2.5,   3.5))      # ±0.5 mm pupil
            sigma_mm    = float(rng.uniform(2.5,   3.5))      # ±0.5 mm aperture
            noise       = float(rng.uniform(0.01,  0.04))     # noise variability
        else:
            D_mm       = 1000.0
            ecc_mm     = 10.0
            pupil_r_mm = 3.0
            sigma_mm   = 3.0
            noise      = noise_std

        img = render_multichannel(
            S, C, axis,
            eccentricity_mm=ecc_mm,
            D_mm=D_mm,
            pupil_r_mm=pupil_r_mm,
            sigma_mm=sigma_mm,
            image_size=image_size,
            noise_std=noise,
            rng=rng,
        )

        M, J0, J45 = prescription_to_power_vector(S, C, axis)
        images[i] = img
        labels[i] = [M, J0, J45]

        if (i + 1) % 10000 == 0:
            print(f"  {i + 1:>7} / {n} generated")

    return images, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",        type=int,  default=100_000, help="total samples")
    ap.add_argument("--size",     type=int,  default=64,      help="image size (square)")
    ap.add_argument("--noise",    type=float,default=0.02,    help="additive noise std")
    ap.add_argument("--augment",  action="store_true",        help="domain randomisation")
    ap.add_argument("--seed",     type=int,  default=42)
    ap.add_argument("--out",      type=str,  default="data",  help="output directory")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n} samples (augment={args.augment}) ...")
    images, labels = generate(args.n, args.size, args.noise, args.augment, args.seed)

    # 80 / 10 / 10 split
    n     = args.n
    n_val  = n // 10
    n_test = n // 10
    n_train = n - n_val - n_test

    rng = np.random.default_rng(args.seed + 1)
    idx = rng.permutation(n)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train : n_train + n_val]
    test_idx  = idx[n_train + n_val :]

    for split, sidx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        path = out_dir / f"{split}.npz"
        np.savez_compressed(path, images=images[sidx], labels=labels[sidx])
        print(f"Saved {path}  ({len(sidx)} samples)")

    print("Done.")


if __name__ == "__main__":
    main()
