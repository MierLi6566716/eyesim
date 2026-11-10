"""
Reads your eye photos and works out your prescription. Runs on the Mac.

It gives two answers that don't depend on each other:

  1. The network: reads each session's photos, then the trial lens you held
     is added back. Sessions where the pupil was dark (blind band) are skipped.
  2. The lens sweep (no network): for each LED direction it finds the trial
     lens where the crescent jumps from one side of the pupil to the other.
     That's where eye + lens sits exactly in the blind band, which is at a
     known power (-1000/d D), so your eye = that + the lens. This is what an
     optometrist does by hand with retinoscopy.

  python scripts/measure_eye.py data/eyes/*right*
  python scripts/measure_eye.py data/eyes/*right* --known "-2.25 -0.50 x 180" --age 21
  python scripts/measure_eye.py --calibrate-scale data/scale_H.npy --dist-cm 37

For each session you click once on the pupil, then 3 points on its edge. The
clicks are saved (clicks.json) so you only do it once (--reclick to redo).
The rest of the frames are lined up using the bright corneal reflection.

The network expects the H LED to the RIGHT of the lens and the V LED ABOVE it,
as seen in the photo. If yours is mirrored, add --hflip and/or --vflip.

The answer is given at the eye and as a glasses prescription (12 mm in front).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scipy.ndimage import gaussian_filter, map_coordinates

from eyesim.optimize import power_vector_to_prescription as to_rx
from eyesim.photorefraction import (LCA_850_D, Eye, Geometry, Scene, I_REF, aux_vector, ir_to_visible, reflex,
                                    render_pair)

SIZE = 64                  # PupilNet input
FILL = 0.75                # pupil diameter / crop width, middle of the training range
VISIBLE_PEAK = 0.15        # same threshold as train.py
SIDE_MIN = 0.08            # |crescent centroid| / pupil radius below this = no crescent
GLINT_LEVEL = 1.0          # the glint is scaled to full scale, as in most training frames


# ------------------------------------------------------------------ frames
def load_frame(path, white, hflip, vflip):
    """Linear grey float32, 1.0 = sensor full scale."""
    if path.suffix == ".npy":
        lin = np.load(path).astype(np.float32) / white
    elif path.suffix == ".npz":                 # packed by scripts/pack_sessions.py (lossless)
        with np.load(path) as z:
            lin = z["frame"].astype(np.float32) / white
    else:                                   # old rpicam-still PNG sessions: undo the tone curve
        from PIL import Image
        g = np.asarray(Image.open(path).convert("RGB"), np.float32).mean(-1) / 255.0
        lin = np.where(g <= 0.04045, g / 12.92, ((g + 0.055) / 1.055) ** 2.4)
    if hflip:
        lin = lin[:, ::-1]
    if vflip:
        lin = lin[::-1]
    return np.ascontiguousarray(lin, dtype=np.float32)


def frame_path(session, k, ch):
    for ext in (".npy", ".npz"):
        p = session / f"{k:02d}_{ch}{ext}"
        if p.exists():
            return p
    return session / f"{k:02d}_{ch}.png"


def fit_circle(pts):
    pts = np.asarray(pts, float)
    A = np.c_[2 * pts, np.ones(len(pts))]
    cx, cy, c = np.linalg.lstsq(A, (pts ** 2).sum(1), rcond=None)[0]
    return cx, cy, float(np.sqrt(c + cx ** 2 + cy ** 2))


def show(img):
    return np.clip(img / (np.percentile(img, 99.5) + 1e-9), 0, 1) ** 0.5


def get_clicks(session, img, reclick):
    store = session / "clicks.json"
    if store.exists() and not reclick:
        return json.loads(store.read_text())
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(show(img), cmap="gray")
    ax.set_title(f"{session.name}: click once on the pupil of the eye to measure")
    (x0, y0), = plt.ginput(1, timeout=0)
    h = max(60, max(img.shape) / 25)          # zoom half-width (crops are only 512 px)
    ax.set_xlim(x0 - h, x0 + h)
    ax.set_ylim(y0 + h, y0 - h)
    ax.set_title("click 3 points spread around the PUPIL edge")
    fig.canvas.draw()
    cx, cy, r = fit_circle(plt.ginput(3, timeout=0))
    ax.add_patch(plt.Circle((cx, cy), r, fill=False, color="c"))
    ax.set_title("cyan circle = your pupil. Click anywhere to accept")
    fig.canvas.draw()
    plt.ginput(1, timeout=0)
    plt.close(fig)
    clicks = {"pupil_xy_r": [cx, cy, r]}
    store.write_text(json.dumps(clicks, indent=2))
    return clicks


def glint(img, x, y, half):
    """Corneal glint (sharpest bright spot) within half px of (x, y)."""
    x, y, half = int(round(x)), int(round(y)), int(half)
    w = img[y - half:y + half, x - half:x + half]
    dog = gaussian_filter(w, 1.2) - gaussian_filter(w, 5)
    j, i = np.unravel_index(np.argmax(dog), dog.shape)
    return x - half + i, y - half + j


def refine_pupil(avg, r0, search=None):
    """Centre near the window centre of the circle of radius r0 along which the
    image has the strongest radial edge (mean |d/dr|, so a bright crescent
    inside a dark pupil and a dark pupil inside a brighter iris both count).
    The radius comes from the clicks, which are more reliable than any fit
    against iris texture."""
    n = avg.shape[0]
    search = search or max(3, int(0.12 * r0))
    g = gaussian_filter(avg, max(1.0, 0.03 * r0))
    gy, gx = np.gradient(g)
    th = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    c, s = np.cos(th), np.sin(th)
    best = (-np.inf, 0.0, 0.0, r0)
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            X, Y = n / 2 + dx + r0 * c, n / 2 + dy + r0 * s
            gr = map_coordinates(gx, [Y, X], order=1) * c + map_coordinates(gy, [Y, X], order=1) * s
            score = np.abs(gr).mean()
            if score > best[0]:
                best = (score, dx, dy, r0)
    return float(best[1]), float(best[2]), float(best[3])


def crop(lin, cx, cy, r_px, fill=FILL, shift=(0.0, 0.0)):
    """Resample into the simulator's frame: SIZE x SIZE, pupil centred,
    pupil diameter = fill * width, row 0 at the top."""
    half = 1.0 / fill
    ax = (np.arange(SIZE) + 0.5) / SIZE * 2 * half - half
    U, V = np.meshgrid(ax + shift[0], -ax + shift[1])
    src_per_dst = 2 * half * r_px / SIZE
    img = gaussian_filter(lin, 0.5 * src_per_dst) if src_per_dst > 1.5 else lin
    return map_coordinates(img, [cy - r_px * V, cx + r_px * U], order=1, mode="nearest").astype(np.float32)


def crescent_side(x, led_deg, glint_uv=None):
    """
    Where the reflex sits along the LED direction, in pupil radii, for a
    centred crop x (SIZE x SIZE, fill FILL). + = on the LED's side (eye more
    myopic than the null), - = opposite (more hyperopic), ~0 = dark zone.
    """
    half = 1.0 / FILL
    ax = (np.arange(SIZE) + 0.5) / SIZE * 2 * half - half
    U, V = np.meshgrid(ax, -ax)
    m = np.hypot(U, V) < 0.75            # stays inside the pupil if the centre is a bit off
    if glint_uv is not None:                         # the glint is not reflex
        m &= np.hypot(U - glint_uv[0], V - glint_uv[1]) > 0.25
    w = np.clip(x[m] - np.percentile(x[m], 10), 0, None)
    if w.sum() <= 0:
        return 0.0
    e = np.array([np.cos(np.radians(led_deg)), np.sin(np.radians(led_deg))])
    return float((w * (U[m] * e[0] + V[m] * e[1])).sum() / w.sum())


# ------------------------------------------------------------- prescription
def parse_rx(text):
    nums = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+", text)]
    S, C, A = (nums + [0.0, 0.0])[:3]
    th = np.deg2rad(A)
    return S + C / 2, -(C / 2) * np.cos(2 * th), -(C / 2) * np.sin(2 * th)


def vertex(M, J0, J45, v_mm, inverse=False):
    """Move a refraction between the eye and a plane v_mm in front of it,
    meridian by meridian. Forward: a lens there -> its power at the eye.
    Inverse: refraction at the eye -> the glasses prescription there."""
    S, C, A = to_rx(M, J0, J45)
    v = (-1 if inverse else 1) * v_mm * 1e-3
    s1, s2 = S / (1 - v * S), (S + C) / (1 - v * (S + C))
    th = np.deg2rad(A)
    C = s2 - s1
    return s1 + C / 2, -(C / 2) * np.cos(2 * th), -(C / 2) * np.sin(2 * th)


def lens_at_eye(meta):
    """Trial lens in the frame -> (M, J0, J45) at the eye."""
    S, C, A = meta.get("lens_d", 0.0), meta.get("lens_cyl", 0.0), meta.get("lens_axis", 0.0)
    th = np.deg2rad(A)
    return vertex(S + C / 2, -(C / 2) * np.cos(2 * th), -(C / 2) * np.sin(2 * th), meta.get("vertex_mm", 12.0))


def to_glasses(M, J0, J45):
    return vertex(M, J0, J45, 12.0, inverse=True)


def dark_band(geom, pupil_r):
    """Range of eye+lens refraction (IR, sphere) with no usable crescent."""
    Ms = np.arange(-14.0, 8.01, 0.25)
    peak = [reflex(Eye(M=float(m), pupil_r_mm=pupil_r, scatter_mrad=2.0), geom, 24, FILL, n_entry=150)[0].max()
            * 0.8 / I_REF for m in Ms]
    dark = np.array(peak) < VISIBLE_PEAK
    i = int(np.argmin(np.abs(Ms + 1000.0 / geom.d_mm)))       # the null
    if not dark[i]:
        return (np.nan, np.nan)
    lo, hi = i, i                                              # the dark run around the null
    while lo > 0 and dark[lo - 1]:
        lo -= 1
    while hi < len(Ms) - 1 and dark[hi + 1]:
        hi += 1
    return float(Ms[lo]), float(Ms[hi])


def fmt(M, J0, J45):
    S, C, A = to_rx(M, J0, J45)
    return f"{S:+.2f} {C:+.2f} x {A:03.0f}  (M {M:+.2f})"


def train_ranges(cfg):
    """Rig ranges the checkpoint was trained on (older checkpoints: +-15 % around 37 cm)."""
    if "d_range_mm" not in cfg:
        return {"distance mm": (315.0, 425.0), "LED offset mm": (8.0, 12.0), "aperture mm": (2.0, 3.6),
                "pupil diameter mm": (3.0, 8.0)}
    return {"distance mm": cfg["d_range_mm"], "LED offset mm": cfg["ecc_range_mm"],
            "aperture mm": cfg["aperture_range_mm"], "pupil diameter mm": (3.0, 8.0)}


def rig_input(model, d, ecc, pupil_r, aperture):
    """The rig numbers in the form this checkpoint was trained with."""
    if model.aux_dim == 3:                 # checkpoints from before November 2026
        return np.array([(d - 370.0) / 100.0, (ecc - 10.0) / 5.0, (pupil_r - 3.5) / 2.0], np.float32)
    return aux_vector(d, ecc, pupil_r, aperture)


# --------------------------------------------------------------------- session
def measure(session, model, cfg, args):
    import torch

    meta = json.loads((session / "meta.json").read_text())
    d, ecc = meta["dist_mm"], meta["ecc_mm"]
    aperture = meta.get("aperture_mm") or cfg.get("aperture_mm", 2.6)
    white = meta.get("white_level", 1.0)
    frames = []
    for k in range(meta["shots"]):
        dark = load_frame(frame_path(session, k, "dark"), white, args.hflip, args.vflip)
        frames.append([np.clip(load_frame(frame_path(session, k, ch), white, args.hflip, args.vflip) - dark, 0, None)
                       for ch in ("H", "V")])
    c = get_clicks(session, frames[0][1], args.reclick)
    cx, cy, r_px = c["pupil_xy_r"]

    # Track the eye by its corneal glint; refine the clicked circle on the
    # average of all glint-aligned frames.
    gx0, gy0 = glint(frames[0][1], cx, cy, 1.0 * r_px)
    pos = [[glint(img, gx0, gy0, 1.3 * r_px) for img in pair] for pair in frames]
    # Each LED puts its glint at a slightly different place on the cornea. The
    # mean H-minus-V glint offset over all shots is that LED effect (eye motion
    # between frames averages out); remove it, then fit one pupil circle on
    # the average of all frames.
    dH = np.mean([np.subtract(ps[0], ps[1]) for ps in pos], 0)
    led_off = [dH, np.zeros(2)]
    half = int(2 * r_px)
    wins = []
    for pair, ps in zip(frames, pos):
        for ch in range(2):
            x = int(round(cx + ps[ch][0] - led_off[ch][0] - gx0))
            y = int(round(cy + ps[ch][1] - led_off[ch][1] - gy0))
            wins.append(pair[ch][y - half:y + half, x - half:x + half])
    if getattr(args, "refine", True):     # off when the centre already comes from the reflex itself (model eye)
        dcx, dcy, r_px = refine_pupil(np.mean(wins, 0), r_px)
        cx, cy = cx + dcx, cy + dcy
    glint_uv = [((gx0 + o[0] - cx) / r_px, -(gy0 + o[1] - cy) / r_px) for o in led_off]

    # Brightness: the glint is a mirror image of the LED, independent of the
    # refraction, so scaling it to a fixed level fixes exposure without
    # brightening a pupil that is genuinely dark.
    peaks = [gaussian_filter(img[gy - 3:gy + 4, gx - 3:gx + 4], 0.7).max()
             for pair, ps in zip(frames, pos) for img, (gx, gy) in zip(pair, ps)]
    gain = GLINT_LEVEL / max(float(np.median(peaks)), 1e-4) if args.brightness == "glint" else 1.0
    gain = min(gain, 200.0)

    if meta.get("px_per_mm"):
        px_per_mm = meta["px_per_mm"] * meta.get("px_per_mm_dist_mm", d) / d
    elif frame_path(session, 0, "H").suffix in (".npy", ".npz"):
        raise SystemExit(f"{session}: no px_per_mm in meta.json. Do the ruler calibration (README, Mac step 2).")
    else:
        px_per_mm = args.focal_mm / ((d - args.focal_mm) * args.pixel_um * 1e-3)
    pupil_r = r_px / px_per_mm
    moves = [np.hypot(gx - gx0, gy - gy0) for ps in pos for gx, gy in ps]

    xs, auxs, shot_id, centred, sides = [], [], [], [], []
    for k, (pair, ps) in enumerate(zip(frames, pos)):
        chans = [(img, cx + gx - led_off[ch][0] - gx0, cy + gy - led_off[ch][1] - gy0)
                 for ch, (img, (gx, gy)) in enumerate(zip(pair, ps))]
        base = np.clip(np.stack([crop(im, x0, y0, r_px) for im, x0, y0 in chans]) * gain, 0, 1)
        centred.append(base)
        sides.append([crescent_side(base[0], 0, glint_uv[0]), crescent_side(base[1], 90, glint_uv[1])])
        for fill in (0.65, 0.75, 0.85):
            for shift in [(0, 0), (0.06, 0), (-0.06, 0), (0, 0.06), (0, -0.06)]:
                for dscale in (0.95, 1.0, 1.05):
                    x = np.stack([crop(im, x0, y0, r_px, fill, shift) for im, x0, y0 in chans])
                    xs.append(np.clip(x * gain, 0, 1))
                    auxs.append(rig_input(model, d * dscale, ecc, pupil_r, aperture))
                    shot_id.append(k)
    with torch.no_grad():
        out = model(torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(auxs))).numpy()
    if getattr(model, "m_only", False):                # trained for the spherical equivalent only
        out[:, 1:] = 0.0
    shot_id = np.array(shot_id)
    per_shot = np.array([np.median(out[shot_id == k], 0) for k in range(len(frames))])
    sys_ir = np.median(per_shot, 0)                     # eye + trial lens, IR, image frame
    lens = np.array(lens_at_eye(meta))
    if args.hflip != args.vflip:                         # trial-frame axes are in the real-world frame
        lens[2] = -lens[2]
    eye_ir = sys_ir + lens                               # the lens alone would need -lens to cancel
    side = np.median(sides, 0)

    geom = Geometry(d_mm=d, ecc_mm=ecc, aperture_r_mm=aperture / 2)
    I, _, _ = reflex(Eye(M=float(sys_ir[0]), J0=float(sys_ir[1]), J45=float(sys_ir[2]), pupil_r_mm=pupil_r,
                         scatter_mrad=2.0), geom, 32, FILL)
    dark_zone = float(I.max()) * 0.8 / I_REF < VISIBLE_PEAK

    lens_txt = f"S{meta.get('lens_d', 0):+g}" + (f" C{meta['lens_cyl']:+g}x{meta['lens_axis']:.0f}"
                                                 if meta.get("lens_cyl") else "")
    print(f"\n{session.name}: d {d:.0f} mm, LED {ecc:.1f} mm, lens {lens_txt}, pupil {2 * pupil_r:.1f} mm, "
          f"{len(frames)} shots, eye moved <= {max(moves):.0f} px, brightness x{gain:.1f}")
    for key, val in [("distance mm", d), ("LED offset mm", ecc), ("aperture mm", aperture),
                     ("pupil diameter mm", 2 * pupil_r)]:
        lo, hi = train_ranges(cfg)[key]
        if not lo <= val <= hi:
            print(f"  WARNING: {key} = {val:.1f} is outside what the network was trained on ({lo:.1f}-{hi:.1f})."
                  + (" Change the rig or retrain (README)." if key != "pupil diameter mm" else ""))
    if args.brightness == "glint" and np.median(peaks) >= 0.98:
        print("  note: the glint is clipped, so brightness was not rescaled; lower --gain if the pupil is clipped too")
    print("  per shot M (eye, visible):  " + "  ".join(f"{m + lens[0] - LCA_850_D:+.2f}" for m in per_shot[:, 0]))
    print(f"  CNN result (eye, visible):  {fmt(ir_to_visible(eye_ir[0]), eye_ir[1], eye_ir[2])}"
          f"   spread {per_shot[:, 0].std():.2f} D")
    words = {1: "LED side (myopic of null)", -1: "far side (hyperopic of null)", 0: "none (dark)"}
    sign = lambda a: 0 if abs(a) < SIDE_MIN else int(np.sign(a))
    print(f"  crescent: H {side[0]:+.2f} {words[sign(side[0])]},  V {side[1]:+.2f} {words[sign(side[1])]}")
    if dark_zone:
        lo, hi = dark_band(geom, pupil_r)
        sh = lens[0] - LCA_850_D
        print(f"  DARK ZONE: no crescent for an eye (visible) between {lo + sh:+.1f} and {hi + sh:+.1f} D "
              f"with this lens; the CNN number is a guess.")
    return dict(name=session.name, d=d, ecc=ecc, aperture=aperture, pupil_r=pupil_r, eye_ir=eye_ir,
                sys_ir=sys_ir, lens=lens, meta=meta, dark=dark_zone, centred=centred[0], side=side)


_CURVES = {}


def side_curve(d, ecc, aperture, pupil_r):
    """Simulated crescent class (+1 LED side, 0 dark, -1 far side) against the
    refraction of eye + lens (IR), for this rig and pupil, using the same
    detector as for the photos."""
    key = (round(d), round(ecc, 1), round(aperture, 1), round(pupil_r, 1))
    if key not in _CURVES:
        Ms = np.arange(-16.0, 10.01, 0.25)
        geom = Geometry(d_mm=d, ecc_mm=ecc, aperture_r_mm=aperture / 2, led_angles_deg=(0.0,))
        rng = np.random.default_rng(0)
        cls = []
        for m in Ms:
            img = render_pair(Eye(M=float(m), pupil_r_mm=pupil_r, scatter_mrad=2.0, sa_d_per_mm2=0.12), geom,
                              Scene(fill=FILL, red_reflex=0.03, read_noise=0.003, shot_noise=0.005), rng)[..., 0]
            a = crescent_side(img / max(img.max(), 1e-6), 0)
            cls.append(0 if abs(a) < SIDE_MIN else int(np.sign(a)))
        _CURVES[key] = (Ms, np.array(cls))
    return _CURVES[key]


def accommodated(k_sys, amplitude):
    """
    Eye + lens refraction (IR) after the eye focuses on the far target. A
    young eye cancels hyperopia (an over-minus lens) up to its amplitude of
    accommodation, but cannot cancel myopia (a plus lens): that is why the
    sweep uses plus lenses. The 4 m target needs -0.25 D in visible light.
    """
    target = -0.25 + LCA_850_D
    excess = np.clip(k_sys - target, 0.0, None)
    return k_sys - np.minimum(excess, amplitude)


def neutralize(res, amplitude):
    """
    Per LED meridian, the eye refraction that best explains which side the
    crescent was on (or that there was none) in every trial-lens session.
    Eye + lens = eye - lens power, then accommodation; the simulator says which
    class each value gives for this rig and pupil.
    """
    out = {}
    K = np.arange(-15.0, 10.0, 0.05)
    for ch, (name, sgn) in enumerate([("horizontal (H LED)", +1), ("vertical (V LED)", -1)]):
        score = np.zeros_like(K)
        seen = set()
        for r in res:
            x = r["lens"][0] + sgn * r["lens"][1]           # lens power along this meridian, at the eye
            obs = 0 if abs(r["side"][ch]) < SIDE_MIN else int(np.sign(r["side"][ch]))
            seen.add(obs)
            Ms, cls = side_curve(r["d"], r["ecc"], r["aperture"], r["pupil_r"])
            k_eff = accommodated(K - x, amplitude)
            idx = np.clip(np.round((k_eff - Ms[0]) / 0.25).astype(int), 0, len(Ms) - 1)
            score += cls[idx] == obs
        best = K[score == score.max()]
        width = best.max() - best.min()
        if seen == {0} or width > 3.0 or best.min() <= K[0] + 0.1 or best.max() >= K[-1] - 0.1:
            if seen == {1}:
                out[name] = "the crescent is on the LED side in every session: add MINUS lenses in 1 D " \
                            "steps until it disappears"
            elif 1 not in seen:
                out[name] = "add PLUS lenses in 1 D steps until the crescent shows on the LED side"
            else:
                out[name] = "add lenses in 0.5 D steps between the last dark and the first LED-side session"
            continue
        out[name] = (float(np.median(best)) - LCA_850_D, float(0.5 * width + 0.125), int(score.max()), len(res))
    return out


# --------------------------------------------------------------------- main
def calibrate_scale(path, dist_cm):
    import matplotlib.pyplot as plt
    img = np.load(path).astype(np.float32) if path.endswith(".npy") else \
        np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(path).convert("L"), np.float32)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(show(img), cmap="gray")
    ax.set_title("zoom with the toolbar if needed, then click two ruler marks far apart")
    p = plt.ginput(2, timeout=0)
    plt.close(fig)
    mm = float(input("How many mm apart are the two marks you clicked? "))
    ppm = float(np.hypot(p[1][0] - p[0][0], p[1][1] - p[0][1]) / mm)
    print(f"\n{ppm:.3f} px per mm at {dist_cm:g} cm. Add to rig.json on the Pi:\n"
          f'  "px_per_mm": {ppm:.3f}, "px_per_mm_dist_mm": {10 * dist_cm:.0f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--known", default=None, help='your real glasses prescription, e.g. "-2.25 -0.50 x 180"')
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints/best.pt"))
    ap.add_argument("--hflip", action="store_true", help="H LED is LEFT of the lens in the image")
    ap.add_argument("--vflip", action="store_true", help="V LED is BELOW the lens in the image")
    ap.add_argument("--brightness", choices=["glint", "none"], default="glint")
    ap.add_argument("--focal-mm", type=float, default=4.74, help="old .png sessions only")
    ap.add_argument("--pixel-um", type=float, default=1.4, help="old .png sessions only")
    ap.add_argument("--age", type=float, default=25.0,
                    help="your age; sets how much an over-minus lens can be cancelled by focusing")
    ap.add_argument("--reclick", action="store_true")
    ap.add_argument("--out", default=None, help="figure path; default <first session>/result.png")
    ap.add_argument("--calibrate-scale", metavar="IMAGE", help="ruler frame from eye_capture.py --scale")
    ap.add_argument("--dist-cm", type=float, default=None, help="with --calibrate-scale")
    args = ap.parse_args()

    args.accommodation = max(0.0, 15.0 - 0.25 * args.age)   # Hofstetter's minimum amplitude
    if args.calibrate_scale:
        if not args.dist_cm:
            raise SystemExit("--calibrate-scale needs --dist-cm (camera lens to ruler)")
        calibrate_scale(args.calibrate_scale, args.dist_cm)
        return
    if not args.sessions:
        ap.error("give one or more session folders")

    import torch
    from eyesim.cnn import PupilNet, normalize_frames

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PupilNet(**ck["model_kwargs"]).eval()
    model.load_state_dict(ck["model_state_dict"])
    model.m_only = bool(ck.get("m_only", False))
    cfg = ck.get("dataset_config", {})
    m_lo, m_hi = cfg.get("m_range_d", (-9.0, 5.0))

    res = [measure(Path(s), model, cfg, args) for s in sorted(args.sessions)]
    for r in res:                                       # mirror flip: J45 changes sign
        if args.hflip != args.vflip:
            r["eye_ir"][2] = -r["eye_ir"][2]

    print("\n================ your eye (visible light) ================")
    # The CNN is only trusted with a crescent present and eye + lens inside its
    # training labels (IR; the signal saturates near the ends, so keep 1 D clear).
    good = [r for r in res if not r["dark"] and m_lo + 1 <= r["sys_ir"][0] <= m_hi - 1]
    if len(good) < len([r for r in res if not r["dark"]]):
        print(f"  (sessions where eye + lens reads beyond {m_lo + 1:+.0f}..{m_hi - 1:+.0f} D left out of the CNN "
              "result: outside training)")
    if good:
        # A lens more minus than the eye needs is cancelled by focusing, which
        # makes the CNN read the eye too myopic: drop those sessions.
        e = np.median([r["eye_ir"] for r in good], 0)
        relaxed = [r for r in good if r["lens"][0] >= e[0] - LCA_850_D - 0.5]
        if relaxed and len(relaxed) < len(good):
            print(f"  ({len(good) - len(relaxed)} over-minus session(s) left out of the CNN result: "
                  "the eye can focus through them)")
            good = relaxed
            e = np.median([r["eye_ir"] for r in good], 0)
        M, J0, J45 = ir_to_visible(float(e[0])), float(e[1]), float(e[2])
        print(f"  CNN, at the eye       {fmt(M, J0, J45)}   from {len(good)} of {len(res)} sessions")
        print(f"  CNN, glasses (12 mm)  {fmt(*to_glasses(M, J0, J45))}")
        if model.m_only:
            print("  (this network gives the spherical equivalent only: sphere + half the cylinder)")
        elif abs(to_rx(M, J0, J45)[1]) < 0.5:
            print("  (cylinder under 0.50 D is within the network's noise)")
    else:
        print("  CNN: every session was in its dark zone; no measurement. Add trial-lens sessions.")
    neut = neutralize(res, args.accommodation)
    for name, v in neut.items():
        if isinstance(v, tuple):
            print(f"  lens sweep, {name:<19} {v[0]:+.2f} D +- {v[1]:.2f} at the eye "
                  f"(explains {v[2]} of {v[3]} sessions)")
        else:
            print(f"  lens sweep, {name:<19} not pinned down yet: {v}")
    if all(isinstance(v, tuple) for v in neut.values()):
        h, v = neut["horizontal (H LED)"][0], neut["vertical (V LED)"][0]
        Mn, J0n = 0.5 * (h + v), 0.5 * (h - v)
        print(f"  lens sweep, combined  M {Mn:+.2f}, J0 {J0n:+.2f}  ->  glasses {fmt(*to_glasses(Mn, J0n, 0.0))}"
              "  (J45 not measured by 2 LEDs)")
    if args.known:
        Mk, J0k, J45k = parse_rx(args.known)
        print(f"  your known Rx         {fmt(Mk, J0k, J45k)}   (glasses)")
        if good:
            Ms, J0s, J45s = to_glasses(M, J0, J45)
            print(f"  CNN error             {Ms - Mk:+.2f} D in M"
                  + ("" if model.m_only else f", {np.hypot(J0s - J0k, J45s - J45k):.2f} D astigmatism"))
        if all(isinstance(v, tuple) for v in neut.values()):
            print(f"  lens-sweep error      {to_glasses(Mn, J0n, 0.0)[0] - Mk:+.2f} D in M")
    print("==========================================================")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    cols = 6 if args.known else 4
    fig, axes = plt.subplots(len(res), cols, figsize=(1.6 * cols, 1.9 * len(res)), squeeze=False)
    for r, row in zip(res, axes):
        geom = Geometry(d_mm=r["d"], ecc_mm=r["ecc"], aperture_r_mm=r["aperture"] / 2)
        sim = lambda v: render_pair(Eye(M=float(v[0]), J0=float(v[1]), J45=float(v[2]), pupil_r_mm=r["pupil_r"],
                                        scatter_mrad=2.0), geom, Scene(fill=FILL, glint_amp=1.0, red_reflex=0.03),
                                    rng).transpose(2, 0, 1)
        panels = [("photo", r["centred"]), ("sim @ CNN", sim(r["sys_ir"]))]
        if args.known:
            eye_vis = np.array(vertex(Mk, J0k, J45k, 12.0))           # glasses -> eye
            sys_known = eye_vis + np.array([LCA_850_D, 0, 0]) - r["lens"]
            if args.hflip != args.vflip:
                sys_known[2] = -sys_known[2]
            panels.append(("sim @ your Rx", sim(sys_known)))
        for p, (label, x) in enumerate(panels):
            x = normalize_frames(torch.from_numpy(np.asarray(x, np.float32))[None])[0].numpy()
            for ch in range(2):
                ax = row[2 * p + ch]
                ax.imshow(x[ch], cmap="gray", vmin=0, vmax=1.2)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{label} {'HV'[ch]}", fontsize=7)
        m = r["meta"]
        row[0].set_ylabel(f"S{m.get('lens_d', 0):+g}" + (f" C{m['lens_cyl']:+g}" if m.get("lens_cyl") else "")
                          + (" dark" if r["dark"] else ""), fontsize=8)
    fig.tight_layout()
    out = Path(args.out) if args.out else Path(sorted(args.sessions)[0]) / "result.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
