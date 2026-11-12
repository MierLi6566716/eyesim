"""
Refraction over time from a video recording (eye_capture.py --video). Runs on the Mac.

  python scripts/video_refraction.py own_eye/<video folder> [--checkpoint ...] [--click]

For every clean H frame and the next clean V frame (a "pair") it subtracts the
nearest dark frame, follows the eye by its corneal glint, crops the pupil the
way the network expects, and runs the network. Writes into the folder:
  refraction.csv   t_s, reading (eye + lens, IR, D), readable (crescent), glint_x, glint_y
  refraction.png   reading vs time

The pupil is clicked (--click, 1 + 3 clicks as in measure_eye.py; recommended
for real eyes, where the iris edge is visible), given (--centre X Y R), or found
automatically from the outer edge of the crescents. The automatic way is biased
toward the lit side when every crescent is on one side (a myopic eye lights only
the LED side): 3-4 px on a synthetic test. The pupil's offset from the glint is
kept fixed, so the crop follows small eye movements.

Raw video frames are a Bayer mosaic (the sensor bins 2x2 itself); at 850 nm the
four colour sites respond differently, so each site is scaled to the mean,
using the bright pixels of the clean H frames.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import measure_eye as me  # noqa: E402


def bayer_gains(frames, idx_h, dark):
    acc = np.zeros(4); n = np.zeros(4)
    for i in idx_h[:20]:
        a = np.clip(frames[i].astype(np.float32) - dark, 0, None)
        thr = np.percentile(a, 95)
        for k, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            s = a[dy::2, dx::2]
            m = s > thr
            acc[k] += s[m].sum(); n[k] += m.sum()
    mean = acc / np.maximum(n, 1)
    return mean.mean() / np.maximum(mean, 1e-6)


def equalise(a, g):
    a = a.astype(np.float32).copy()
    for k, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        a[dy::2, dx::2] *= g[k]
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints/wide_m_only/best.pt"))
    ap.add_argument("--click", action="store_true", help="click the pupil (recommended for real eyes)")
    ap.add_argument("--centre", type=float, nargs=3, metavar=("X", "Y", "R"), help="pupil centre and radius in px")
    ap.add_argument("--pupil-mm", type=float, default=None, help="known pupil (model eye); default: measured")
    args = ap.parse_args()

    import torch
    from model_eye_report import load_net
    from eyesim.photorefraction import aux_vector

    folder = Path(args.folder)
    meta = json.loads((folder / "meta.json").read_text())
    frames = np.load(folder / "frames.npy", mmap_mode="r")
    rows = list(csv.DictReader(open(folder / "frames.csv")))
    led = [r["led"] for r in rows]
    ts = np.array([int(r["sensor_ts_ns"]) for r in rows]) / 1e9
    ts = ts - ts[0]
    i_h = [i for i, l in enumerate(led) if l == "H"]
    i_v = [i for i, l in enumerate(led) if l == "V"]
    i_d = [i for i, l in enumerate(led) if l == "dark"]
    if not (i_h and i_v and i_d):
        raise SystemExit("need clean H, V and dark frames")
    dark0 = np.mean([frames[i] for i in i_d[:3]], 0).astype(np.float32)
    g = bayer_gains(frames, i_h, dark0) if meta.get("bayer_raw") else np.ones(4)
    white = float(meta.get("white_level", 1023)) * float(g.max())
    model, cfg = load_net(args.checkpoint)

    d = meta["dist_mm"]; ecc = meta["ecc_mm"]; aperture = meta["aperture_mm"]
    ppm = meta["px_per_mm"] * meta.get("px_per_mm_dist_mm", d) / d

    def frame(i):
        j = min(i_d, key=lambda k: abs(k - i))                    # nearest dark frame
        return np.clip(equalise(frames[i], g) - equalise(frames[j], g), 0, None) / white

    h0 = frame(i_h[0])
    if args.centre:
        cx, cy, r = args.centre
    elif args.click:
        cx, cy, r = me.get_clicks(folder, h0, True)["pupil_xy_r"]
    else:
        # The crescent's OUTER edge is the pupil edge (model-eye lesson). Take the brightest
        # value each pixel reaches over the H and V frames, then fit a circle to the outer
        # half-maximum edge of that light, searching outward from the glint.
        from scipy.ndimage import gaussian_filter
        from scipy.optimize import least_squares
        mx = np.max([gaussian_filter(frame(k), 1.0) for k in (i_h[::5] + i_v[::5])[:80]], 0)
        gx, gy = me.glint(h0, h0.shape[1] / 2, h0.shape[0] / 2, h0.shape[0] / 2 - 4)
        yy, xx = np.indices(mx.shape)
        mx[np.hypot(xx - gx, yy - gy) < 4] = 0                     # the glint is not reflex
        pts = []
        for a in np.linspace(0, 2 * np.pi, 180, endpoint=False):
            rs = np.arange(0.5 * ppm, 5.0 * ppm, 0.25)
            xs = np.clip((gx + rs * np.cos(a)).astype(int), 0, mx.shape[1] - 1)
            ys = np.clip((gy + rs * np.sin(a)).astype(int), 0, mx.shape[0] - 1)
            prof = mx[ys, xs]
            pk = int(np.argmax(prof))
            if prof[pk] < 0.25 * mx.max():
                continue
            o = np.nonzero(prof[pk:] < 0.5 * prof[pk])[0]
            if len(o):
                pts.append((xs[pk + o[0]], ys[pk + o[0]]))
        pts = np.array(pts, float)
        if len(pts) < 20:
            raise SystemExit("could not find the pupil edge automatically: use --click")
        if args.pupil_mm:                                            # known radius: fit the centre only
            r = args.pupil_mm / 2 * ppm
            cx, cy = least_squares(lambda c: np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) - r,
                                   pts.mean(0), loss="soft_l1").x
        else:
            cx, cy, r = me.fit_circle(pts)
        print(f"pupil found from {len(pts)} edge points: centre ({cx:.1f}, {cy:.1f}), diameter {2 * r / ppm:.1f} mm")
    if args.pupil_mm:
        r = args.pupil_mm / 2 * ppm
    gx0, gy0 = me.glint(h0, cx, cy, r)
    off = (cx - gx0, cy - gy0)
    pupil_r = r / ppm

    out = []
    for i in i_h:
        later = [j for j in i_v if j > i]
        if not later or later[0] - i > 6:
            continue
        j = later[0]
        H, V = frame(i), frame(j)
        gx, gy = me.glint(H, gx0, gy0, 1.3 * r)
        x0, y0 = gx + off[0], gy + off[1]
        peak = me.gaussian_filter(H[gy - 3:gy + 4, gx - 3:gx + 4], 0.7).max()
        gain = min(me.GLINT_LEVEL / max(peak, 1e-4), 200.0)
        x = np.clip(np.stack([me.crop(H, x0, y0, r), me.crop(V, x0, y0, r)]) * gain, 0, 1).astype(np.float32)
        side = me.crescent_side(x[0], 0)
        with torch.no_grad():
            m = float(model(torch.from_numpy(x[None]), torch.from_numpy(aux_vector(d, ecc, pupil_r, aperture)[None]))[0, 0])
        out.append((0.5 * (ts[i] + ts[j]), m + meta.get("lens_d", 0.0), abs(side) >= me.SIDE_MIN, gx, gy))
    with open(folder / "refraction.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "eye_ir_d", "crescent", "glint_x", "glint_y"])
        w.writerows([(round(a, 4), round(b, 3), int(c), dx, dy) for a, b, c, dx, dy in out])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array([o[0] for o in out]); y = np.array([o[1] for o in out]); c = np.array([o[2] for o in out])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t[c], y[c], ".", ms=4, label="crescent visible")
    ax.plot(t[~c], y[~c], ".", ms=4, color="0.7", label="no crescent (reading unreliable)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("eye refraction, IR (D)"); ax.legend()
    ax.set_title(f"{folder.name}: {len(out)} pairs in {t[-1]:.1f} s ({len(out) / max(t[-1], 1e-9):.1f} per s)")
    fig.tight_layout(); fig.savefig(folder / "refraction.png", dpi=120)
    print(f"{len(out)} pairs, {len(out) / max(t[-1], 1e-9):.1f} per second, pupil {2 * pupil_r:.1f} mm. "
          f"Wrote refraction.csv and refraction.png in {folder}")


if __name__ == "__main__":
    main()
