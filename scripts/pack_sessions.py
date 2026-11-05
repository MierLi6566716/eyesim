"""
Shrinks session folders about 2x without losing anything: each raw frame
NN_H.npy / NN_V.npy / NN_dark.npy becomes a compressed NN_*.npz, checked
pixel for pixel before the .npy is deleted. measure_eye.py and
model_eye_report.py read both.

  python scripts/pack_sessions.py model_eye/sessions/*
"""

import sys
from pathlib import Path

import numpy as np


def pack(folder: Path) -> tuple[int, int]:
    before = after = 0
    for f in sorted(folder.glob("[0-9][0-9]_*.npy")):
        a = np.load(f)
        out = f.with_suffix(".npz")
        np.savez_compressed(out, frame=a)
        with np.load(out) as z:
            if not np.array_equal(z["frame"], a):
                out.unlink()
                raise SystemExit(f"{out}: round trip failed, kept the .npy")
        before += f.stat().st_size
        after += out.stat().st_size
        f.unlink()
    return before, after


def main():
    folders = [Path(p) for p in sys.argv[1:] if Path(p).is_dir()]
    if not folders:
        raise SystemExit(__doc__)
    tb = ta = 0
    for d in folders:
        b, a = pack(d)
        tb, ta = tb + b, ta + a
        if b:
            print(f"{d.name}: {b / 1e6:.0f} MB -> {a / 1e6:.0f} MB")
    print(f"total {tb / 1e9:.2f} GB -> {ta / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
