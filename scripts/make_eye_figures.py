"""
Draws what your photos should look like on your rig, for the README:
  docs/figures/sweep_frames.png  the H frame as eye + trial lens goes from very myopic to hyperopic
  docs/figures/side_curve.png    where the crescent sits vs eye + lens, for 3 pupil sizes

  python scripts/make_eye_figures.py                         # IMX708 rig at 37 cm
  python scripts/make_eye_figures.py --dist-cm 50 --ecc-mm 18 --aperture-mm 2.9
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eyesim.photorefraction import Eye, Geometry, Scene, render_pair, LCA_850_D
from measure_eye import crescent_side, SIDE_MIN, FILL

OUT = ROOT / "docs" / "figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_LED, C_DARK, C_FAR = "#2a78d6", "#8a8984", "#eb6834"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white"})

ap = argparse.ArgumentParser()
ap.add_argument("--dist-cm", type=float, default=37.0)
ap.add_argument("--ecc-mm", type=float, default=10.0)
ap.add_argument("--aperture-mm", type=float, default=2.6)
args = ap.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
geom = Geometry(d_mm=10 * args.dist_cm, ecc_mm=args.ecc_mm, aperture_r_mm=args.aperture_mm / 2)
null = -100.0 / args.dist_cm
rng = np.random.default_rng(1)

# 1. what the H frame looks like as eye + lens sweeps
Ks = [round(null + k, 1) for k in (-7, -5.5, -4, -3, -1.5, 0, 1.5, 3, 4, 5.5, 7)]
fig, axes = plt.subplots(1, len(Ks), figsize=(11, 1.75))
for ax, k in zip(axes, Ks):
    img = render_pair(Eye(M=k, pupil_r_mm=3.5, scatter_mrad=2, sa_d_per_mm2=0.12), geom,
                      Scene(fill=FILL, glint_amp=1.0, red_reflex=0.03), rng)[..., 0]
    a = crescent_side(img, 0, (0.04, 0.0))
    cls = 0 if abs(a) < SIDE_MIN else int(np.sign(a))
    col = {1: C_LED, 0: C_DARK, -1: C_FAR}[cls]
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(col); s.set_linewidth(2.5)
    ax.set_title(f"{k - LCA_850_D:+.1f} D", fontsize=8, color=INK)
fig.text(0.01, 0.02, "eye + lens, visible light. H LED is to the RIGHT of the lens.   "
         "blue = crescent on LED side   grey = blind band   orange = far side", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(OUT / "sweep_frames.png", dpi=200)
plt.close(fig)

# 2. crescent side vs eye + lens for 3 pupil sizes
Ms = np.arange(null - 9, null + 9.01, 0.25)
fig, ax = plt.subplots(figsize=(7.5, 2.8))
for pr, c, ls in [(2.5, "#1baf7a", ":"), (3.0, C_LED, "--"), (4.0, INK, "-")]:
    vals = []
    for m in Ms:
        img = render_pair(Eye(M=float(m), pupil_r_mm=pr, scatter_mrad=2, sa_d_per_mm2=0.12), geom,
                          Scene(fill=FILL, red_reflex=0.03), rng)[..., 0]
        vals.append(crescent_side(img / max(img.max(), 1e-6), 0))
    ax.plot(Ms - LCA_850_D, vals, color=c, ls=ls, lw=1.8, label=f"pupil {2 * pr:.0f} mm")
ax.axhspan(-SIDE_MIN, SIDE_MIN, color=GRID)
ax.axvline(null - LCA_850_D, color=INK2, lw=0.8)
ax.text(null - LCA_850_D + 0.15, 0.62, "middle of blind band", color=INK2, fontsize=8)
ax.set(xlabel="eye + trial lens, visible light (D)", ylabel="crescent position\n(pupil radii, + = LED side)")
ax.legend(frameon=False, fontsize=8, loc="lower left")
ax.set_ylim(-0.85, 0.85)
fig.tight_layout()
fig.savefig(OUT / "side_curve.png", dpi=200)
plt.close(fig)
print("wrote", OUT)
