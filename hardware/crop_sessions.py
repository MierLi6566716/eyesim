#!/usr/bin/env python3
"""
Makes small copies of a block of sessions for the Mac. Runs on the Pi.

Raw frames are 2304 x 1296 (18 MB per shot); the model eye's pupil is only
about 35-45 px across. This cuts the same SIZE x SIZE box out of every frame
of every session in a block, centred on the model eye, and saves it
compressed (about 50x smaller). The raw folders are not touched.

  python3 crop_sessions.py model_eye/raw/A_37cm_plus4_7mm
      -> model_eye/crops/A_37cm_plus4_7mm/<session>/NN_H.npz ... meta.json preview_H.png

The box centre is found once per block (median brightest spot over its
first sessions) and kept in crops/<block>/crop.json, so sessions added later
get exactly the same box and the pupil clicks on the Mac stay valid.
Sessions already cropped are skipped. If the box is wrong (preview_H.png
shows no pupil), give it by hand: --center X Y (full-frame pixels), --redo.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from eye_capture import bright_spot, preview

HERE = Path(__file__).resolve().parent


def load(path):
    if path.suffix == ".npz":
        with np.load(path) as z:
            return z["frame"]
    return np.load(path)


def frame(session, k, ch):
    for ext in (".npy", ".npz"):
        p = session / f"{k:02d}_{ch}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(session / f"{k:02d}_{ch}.npy")


def spot(session):
    """Brightest spot of (H + V - 2 dark), summed over the first 3 shots."""
    meta = json.loads((session / "meta.json").read_text())
    acc = 0
    for k in range(min(3, meta["shots"])):
        dark = load(frame(session, k, "dark")).astype(np.int32)
        for ch in ("H", "V"):
            acc = acc + np.clip(load(frame(session, k, ch)).astype(np.int32) - dark, 0, None)
    x, y, _ = bright_spot(acc)
    return x, y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("blocks", nargs="+", help="block folders under model_eye/raw/")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--center", type=int, nargs=2, metavar=("X", "Y"))
    ap.add_argument("--redo", action="store_true", help="recompute the box and recrop every session")
    ap.add_argument("--out", default=str(HERE / "model_eye" / "crops"))
    args = ap.parse_args()

    for block in map(Path, args.blocks):
        sessions = sorted(s for s in block.iterdir() if (s / "meta.json").exists())
        if not sessions:
            print(f"{block}: no sessions")
            continue
        out = Path(args.out) / block.name
        out.mkdir(parents=True, exist_ok=True)
        box_file = out / "crop.json"

        found = {}
        if args.center:
            cx, cy = args.center
        elif box_file.exists() and not args.redo:
            cx, cy = json.loads(box_file.read_text())["center"]
        else:
            for s in sessions[:5]:
                found[s.name] = spot(s)
            cx, cy = (int(np.median([p[i] for p in found.values()])) for i in (0, 1))
        h, w = load(frame(sessions[0], 0, "H")).shape
        n = args.size
        x0, y0 = int(np.clip(cx - n // 2, 0, w - n)), int(np.clip(cy - n // 2, 0, h - n))
        box_file.write_text(json.dumps({"center": [int(cx), int(cy)], "x0": x0, "y0": y0, "size": n,
                                        "full_shape": [h, w]}, indent=2))
        print(f"{block.name}: box {n}x{n} at x {x0}..{x0 + n}, y {y0}..{y0 + n} (centre {cx}, {cy})")

        done = 0
        for s in sessions:
            d = out / s.name
            if (d / "meta.json").exists() and not args.redo:
                continue
            meta = json.loads((s / "meta.json").read_text())
            sx, sy = found.get(s.name) or spot(s)
            moved = np.hypot(sx - cx, sy - cy)
            d.mkdir(exist_ok=True)
            for k in range(meta["shots"]):
                for ch in ("H", "V", "dark"):
                    a = load(frame(s, k, ch))[y0:y0 + n, x0:x0 + n]
                    np.savez_compressed(d / f"{k:02d}_{ch}.npz", frame=a)
            preview(load(frame(s, 0, "H"))[y0:y0 + n, x0:x0 + n], d / "preview_H.png")
            meta.update({"crop_x0": x0, "crop_y0": y0, "crop_size": n, "raw_folder": str(s)})
            (d / "meta.json").write_text(json.dumps(meta, indent=2))
            done += 1
            warn = "   <-- eye moved? spot is %.0f px from the block centre" % moved if moved > 40 else ""
            print(f"  {s.name}{warn}")
        mb = sum(f.stat().st_size for f in out.rglob("*.npz")) / 1e6
        print(f"{block.name}: cropped {done} new session(s); crops folder is {mb:.0f} MB")


if __name__ == "__main__":
    main()
