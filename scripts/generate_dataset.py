"""
Makes the training set: lots of fake infrared eye photos with known answers.

Each sample is a random human eye seen by a random rig (distance, LED offset,
lens opening), rendered by eyesim/photorefraction.py. The rig ranges are in
that file (D_RANGE_MM, ECC_RANGE_MM, APERTURE_RANGE_MM); your real rig has to
fall inside them.

Writes to --out (default data/train_set):
  {train,val,test}_images.npy  uint8   (N, 64, 64, 2)  H LED frame, V LED frame
  {train,val,test}_labels.npy  float32 (N, 3)          the answer: M, J0, J45 at 850 nm
  {train,val,test}_aux.npy     float32 (N, 4)          rig numbers the network is told
  {train,val,test}_meta.npy    float32 (N, 8)          extra info for reports (META_COLUMNS)
  config.json

About an hour for 300k samples on 8 cores.

  python scripts/generate_dataset.py
  python scripts/generate_dataset.py --n 2000 --out data/smoke     # quick test
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eyesim.photorefraction import (
    APERTURE_RANGE_MM, D_RANGE_MM, ECC_RANGE_MM, I_REF, M_RANGE_D, aux_vector, reflex, render_pair, sample_eye,
    sample_geometry, sample_scene,
)

META_COLUMNS = ["d_mm", "ecc_mm", "aperture_mm", "pupil_r_mm", "sa_d_per_mm2", "scatter_mrad", "exposure",
                "reflex_peak"]


VISIBLE_PEAK = 0.15      # same threshold as train.py


def _is_dark(eye, geom, scene):
    """Cheap low-resolution check: would this sample sit in the blind band?"""
    I, _, _ = reflex(eye, geom, 24, scene.fill, scene.centre_frac, n_entry=80)
    return float(I.max()) * scene.exposure / I_REF < VISIBLE_PEAK


def _work(job):
    seed, n, size, m_range, dark_keep = job
    rng = np.random.default_rng(seed)
    images = np.empty((n, size, size, 2), np.uint8)
    labels = np.empty((n, 3), np.float32)
    aux = np.empty((n, 4), np.float32)
    meta = np.empty((n, len(META_COLUMNS)), np.float32)
    for i in range(n):
        while True:
            eye, geom, scene = sample_eye(rng, m_range), sample_geometry(rng), sample_scene(rng, size)
            # Blind-band photos carry no refraction; skip most of them so the
            # render time goes to photos with a crescent.
            if dark_keep >= 1.0 or rng.random() < dark_keep or not _is_dark(eye, geom, scene):
                break
        img, peak = render_pair(eye, geom, scene, rng, return_peak=True)
        aperture = 2.0 * geom.aperture_r_mm
        images[i] = np.round(img * 255.0).astype(np.uint8)
        labels[i] = [eye.M, eye.J0, eye.J45]
        aux[i] = aux_vector(geom.d_mm, geom.ecc_mm, eye.pupil_r_mm, aperture)
        meta[i] = [geom.d_mm, geom.ecc_mm, aperture, eye.pupil_r_mm, eye.sa_d_per_mm2, eye.scatter_mrad,
                   scene.exposure, peak]
    return images, labels, aux, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300_000, help="total samples")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/train_set")
    ap.add_argument("--m-min", type=float, default=M_RANGE_D[0], help="most myopic refraction (850 nm, D)")
    ap.add_argument("--m-max", type=float, default=M_RANGE_D[1], help="most hyperopic refraction (850 nm, D)")
    ap.add_argument("--dark-keep", type=float, default=1.0,
                    help="share of blind-band samples to keep (1 = all; 0.3 spends the time on crescents)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_val, n_test = int(args.n * args.val_frac), int(args.n * args.test_frac)
    splits = {"train": args.n - n_val - n_test, "val": n_val, "test": n_test}

    seeds = iter(np.random.SeedSequence(args.seed).generate_state(args.n // args.chunk + 8))
    t0, done = time.time(), 0
    with Pool(args.workers) as pool:
        for split, n in splits.items():
            # Images go straight to disk so training can memory-map them.
            images = np.lib.format.open_memmap(out / f"{split}_images.npy", mode="w+", dtype=np.uint8,
                                               shape=(n, args.size, args.size, 2))
            labels = np.empty((n, 3), np.float32)
            aux = np.empty((n, 4), np.float32)
            meta = np.empty((n, len(META_COLUMNS)), np.float32)
            jobs = [(int(next(seeds)), min(args.chunk, n - s), args.size, (args.m_min, args.m_max), args.dark_keep)
                    for s in range(0, n, args.chunk)]
            pos = 0
            for im, lb, ax, mt in pool.imap(_work, jobs):
                k = len(lb)
                images[pos:pos + k], labels[pos:pos + k], aux[pos:pos + k], meta[pos:pos + k] = im, lb, ax, mt
                pos += k
                done += k
                rate = done / (time.time() - t0)
                print(f"  {split}: {pos:>7}/{n}   {rate:.0f} samples/s   "
                      f"about {(args.n - done) / rate / 60:.0f} min left", flush=True)
            images.flush()
            del images
            np.save(out / f"{split}_labels.npy", labels)
            np.save(out / f"{split}_aux.npy", aux)
            np.save(out / f"{split}_meta.npy", meta)

    (out / "config.json").write_text(json.dumps({
        **vars(args), "splits": splits, "meta_columns": META_COLUMNS,
        "d_range_mm": D_RANGE_MM, "ecc_range_mm": ECC_RANGE_MM, "aperture_range_mm": APERTURE_RANGE_MM,
        "m_range_d": [args.m_min, args.m_max]},
        indent=2))
    print(f"Done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
