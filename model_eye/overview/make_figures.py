"""
Makes the overview figures in this folder from the model-eye results.

  /opt/anaconda3/bin/python model_eye/overview/make_figures.py

Reads   model_eye/results/<block>/results.csv          (current analysis: reflex centring)
        model_eye/overview/data/disc_centre_<block>.csv (same photos, old dark-disc centring)
        model_eye/overview/data/sensitivity.json         (pupil-centre shift experiment)
        model_eye/sessions/<block>/<session>/             (photo crops, for the image strip)
Writes  fig1_accuracy.png  fig2_pupil_centring.png  fig3_physics.png  fig4_literature.png
        data/metrics.csv

All network numbers are the wide_m_only network. "Error" is network reading
minus the dial setting, with NO correction: the dial is calibrated in visible
light and the model eye's own 850 nm offset is unknown, so it stays in the error.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
NET = "checkpoints/wide_m_only/best.pt"

# Reference palette (dataviz skill), light surface.
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "axes.titlesize": 10, "axes.titleweight": "bold"})

BLOCKS = {  # block: (label, lens D)
    "A_37cm_plus4_7mm": ("A  37 cm, +4 lens, 7 mm", 4),
    "B_37cm_plus8_7mm": ("B  37 cm, +8 lens, 7 mm", 8),
    "D2_37cm_plus5_5mm": ("D2  37 cm, +5 lens, 5 mm", 5),
    "R_37cm_plus4_7mm_repeat": ("R  37 cm, +4 lens, re-placed", 4),
    "E2_30cm_plus5_7mm": ("E2  30 cm, +5 lens, 7 mm", 5),
    "F2_25cm_plus6_7mm": ("F2  25 cm, +6 lens, 7 mm", 6),
}


def rows(path):
    return [r for r in csv.DictReader(open(path)) if r["network"] == NET]


def arr(rs, key):
    return np.array([float(r[key]) for r in rs])


def seen(rs):
    return np.array([r["crescent_seen"] == "True" for r in rs])


# ---------------------------------------------------------------- metrics
metrics = []
for b in BLOCKS:
    now = rows(ROOT / "model_eye/results" / b / "results.csv")
    old = {r["session"]: r for r in rows(HERE / "data" / f"disc_centre_{b}.csv")}
    use = [r for r in now if r["crescent_seen"] == "True"]
    err = arr(use, "net_eye_ir") - arr(use, "dial_d")
    err_old = np.array([float(old[r["session"]]["net_eye_ir"]) - float(r["dial_d"]) for r in use])
    slope = np.polyfit(arr(use, "dial_d"), arr(use, "net_eye_ir"), 1)[0] if len(use) >= 3 else np.nan
    metrics.append(dict(block=b, dist_cm=float(use[0]["dist_cm"]), n=len(use), mae=np.abs(err).mean(),
                        bias=err.mean(), within05=(np.abs(err) <= 0.5).mean(), slope=slope,
                        mae_disc=np.abs(err_old).mean(), bias_disc=err_old.mean()))
with open(HERE / "data" / "metrics.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(metrics[0]))
    w.writeheader()
    for m in metrics:
        w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
M = {m["block"]: m for m in metrics}

# ---------------------------------------------------------------- fig 1: accuracy
fig, axes = plt.subplots(2, 3, figsize=(12, 7.6), sharex=False, sharey=False)
for ax, (b, (label, lens)) in zip(axes.flat, BLOCKS.items()):
    rs = rows(ROOT / "model_eye/results" / b / "results.csv")
    x, y, s = arr(rs, "dial_d"), arr(rs, "net_eye_ir"), seen(rs)
    lo, hi = min(x.min(), -5.5) - 0.3, max(x.max(), 0) + 0.3
    t = np.linspace(lo, hi, 2)
    ax.fill_between(t, t - 0.5, t + 0.5, color=S1, alpha=0.10, lw=0, label="within 0.5 D")
    ax.plot(t, t, color=INK2, lw=1, ls="--", label="perfect")
    ax.scatter(x[~s], y[~s], s=34, facecolors=SURF, edgecolors=INK2, lw=1, zorder=3,
               label="blind band (no crescent)")
    ax.scatter(x[s], y[s], s=40, color=S1, edgecolors=SURF, lw=1.5, zorder=4, label="crescent visible")
    m = M[b]
    ax.set_title(label, loc="left")
    ax.text(0.03, 0.97, f"crescent settings: {m['n']}\nmean |error| {m['mae']:.2f} D\nmean error {m['bias']:+.2f} D\n"
            f"within 0.5 D: {100 * m['within05']:.0f} %", transform=ax.transAxes, va="top", fontsize=8,
            color=INK, bbox=dict(boxstyle="round,pad=0.35", fc=SURF, ec=GRID))
    ax.set_xlim(lo, hi); ax.set_ylim(min(lo, y.min() - 0.5), max(hi, y.max() + 0.5))
    ax.set_xlabel("model eye dial setting (D)"); ax.set_ylabel("network reading of the eye (D)")
axes.flat[0].legend(loc="lower right", fontsize=7, frameon=False)
fig.suptitle("Fig. 1  Network reading vs the refraction set on the model eye (trial lens added back)\n"
             "A, D2, R (37 cm, +4/+5 lens): within 0.5 D, no bias. B (+8 lens) and the closer distances E2, F2 read too weak.",
             x=0.01, ha="left", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(HERE / "fig1_accuracy.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- fig 2: pupil centring
sens = json.loads((HERE / "data" / "sensitivity.json").read_text())
fig = plt.figure(figsize=(13, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.1, 1.1])
ax0 = fig.add_subplot(gs[0])
ax0.imshow(plt.imread(ROOT / "model_eye/results/F1_25cm_nolens_7mm/pupil_reflex.png")[:, :, :3])
ax0.axis("off")
ax0.set_title("a  25 cm, blocks F1+F2: all light that came back through\nthe aperture fills a disc of exactly the set 7 mm",
              loc="left", fontsize=9)

ax1 = fig.add_subplot(gs[1])
names = [BLOCKS[b][0].split()[0] for b in BLOCKS]
yy = np.arange(len(BLOCKS))[::-1]
for i, b in zip(yy, BLOCKS):
    ax1.plot([M[b]["mae_disc"], M[b]["mae"]], [i, i], color=GRID, lw=3, zorder=1)
ax1.scatter([M[b]["mae_disc"] for b in BLOCKS], yy, s=60, color=S2, edgecolors=SURF, lw=1.5, zorder=3,
            label="centre of dark disc (old)")
ax1.scatter([M[b]["mae"] for b in BLOCKS], yy, s=60, color=S1, edgecolors=SURF, lw=1.5, zorder=4,
            label="centre from the reflex (new)")
for i, b in zip(yy, BLOCKS):
    ax1.text(max(M[b]["mae_disc"], M[b]["mae"]) + 0.45, i, f"{M[b]['mae_disc']:.2f} → {M[b]['mae']:.2f} D",
             va="center", fontsize=8, color=INK)
ax1.set_yticks(yy); ax1.set_yticklabels([f"{n} ({M[b]['dist_cm']:.0f} cm)" for n, b in zip(names, BLOCKS)])
ax1.set_xlabel("mean |network − dial| over crescent settings (D)")
ax1.set_xlim(0, max(M[b]["mae_disc"] for b in BLOCKS) + 2.2)
ax1.legend(loc="lower right", fontsize=7, frameon=False)
ax1.set_title("b  Same photos, same network, only the pupil centre changed", loc="left", fontsize=9)

ax2 = fig.add_subplot(gs[2])
shifts = [-2, -1, -0.5, 0, 0.5, 1, 2]
for axis, col, lab in (("x", S1, "shift along the H LED axis"), ("y", S3, "shift across it")):
    curves = []
    for key in sorted({k.rsplit("|", 2)[0] for k in sens}):
        base = sens[f"{key}|{axis}|0"]
        curves.append([sens[f"{key}|{axis}|{s}"] - base for s in shifts])
    c = np.array(curves)
    for cv in c:
        ax2.plot(shifts, cv, color=col, lw=0.8, alpha=0.35)
    ax2.plot(shifts, np.median(c, 0), color=col, lw=2, marker="o", ms=5, label=f"{lab} (median of {len(c)})")
ax2.axhspan(-0.25, 0.25, color=INK2, alpha=0.08, lw=0)
ax2.set_ylim(-2.5, 2.5)
ax2.set_xlabel("pupil centre moved by (mm, at the eye)"); ax2.set_ylabel("change in reading (D)")
ax2.legend(loc="upper left", fontsize=7, frameon=False)
ax2.text(0.98, 0.03, "grey band: ±0.25 D (one prescription step)\n±2 mm: off the scale (+5 to +7 D)",
         transform=ax2.transAxes, ha="right", fontsize=7, color=INK2)
ax2.set_title("c  Cost of a centring error\n(6 sessions from blocks A and B)", loc="left", fontsize=9)
fig.suptitle("Fig. 2  The pupil centre decides the reading", x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(HERE / "fig2_pupil_centring.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- fig 3: physics
import measure_eye as me  # noqa: E402


def mean_frame(s, ch):
    meta = json.loads((s / "meta.json").read_text())
    acc = 0
    for k in range(meta["shots"]):
        d = me.load_frame(me.frame_path(s, k, "dark"), meta["white_level"], False, False)
        acc = acc + np.clip(me.load_frame(me.frame_path(s, k, ch), meta["white_level"], False, False) - d, 0, None)
    return acc / meta["shots"]


fig = plt.figure(figsize=(13, 7.4))
gs = fig.add_gridspec(2, 7, height_ratios=[1, 1.5])
blk = ROOT / "model_eye/sessions/A_37cm_plus4_7mm"
dials = ["+0", "-1", "-1.5", "-2", "-3", "-4", "-5"]
imgs = []
for d in dials:
    s = next(blk.glob(f"*true{d}-p*"))
    cx, cy, r = json.loads((s / "clicks.json").read_text())["pupil_xy_r"]
    h = int(1.6 * r)
    imgs.append((d, mean_frame(s, "H")[int(cy) - h:int(cy) + h, int(cx) - h:int(cx) + h], r, h))
vmax = max(np.percentile(im, 99.0) for _, im, _, _ in imgs)
lens_eff = 4 / (1 - 0.005 * 4)
for k, (d, im, r, h) in enumerate(imgs):
    ax = fig.add_subplot(gs[0, k])
    ax.imshow(im, cmap="gray", vmax=vmax)
    ax.add_patch(plt.Circle((h, h), r, fill=False, color=S1, lw=0.8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"dial {float(d):+.1f}\neye+lens {float(d) - lens_eff:+.1f} D", fontsize=8)
fig.text(0.01, 0.965, "a  Block A, H-LED photos (LED to the right), blue circle = 7 mm aperture. In the blind band the pupil is dark;\n"
         "    past it a crescent appears on the LED side and grows with myopia. That growth is what the network reads.",
         fontsize=9, fontweight="bold", va="top")

ax = fig.add_subplot(gs[1, :])
rs = rows(ROOT / "model_eye/results/A_37cm_plus4_7mm/results.csv") + \
    rows(ROOT / "model_eye/results/B_37cm_plus8_7mm/results.csv") + \
    rows(ROOT / "model_eye/results/C_37cm_nolens_7mm/results.csv")
sysM = np.array([float(r["dial_d"]) - float(r["lens_d"]) / (1 - 0.005 * float(r["lens_d"])) for r in rs])
side = np.array([0.5 * (float(r["side_H"]) + float(r["side_V"])) for r in rs])
Ms, cls = me.side_curve(370, 10.0, 2.6, 3.5)
ax.step(Ms, 0.35 * cls, where="mid", color=S2, lw=2, label="simulator: crescent side class (scaled)")
ax.scatter(sysM, side, s=26, color=S1, edgecolors=SURF, lw=1, zorder=3,
           label="real photos, blocks A, B, C (mean of H and V)")
ax.axhline(me.SIDE_MIN, color=INK2, lw=0.6, ls=":"); ax.axhline(-me.SIDE_MIN, color=INK2, lw=0.6, ls=":")
ax.axvline(-1000 / 370, color=INK2, lw=0.8, ls="--")
ax.text(-1000 / 370 + 0.15, 0.45, "null (eye focused\non the camera)", fontsize=7, color=INK2)
# Band edges. Simulator: where its class changes. Photos: the dark plateau is the
# median inside the simulated band; an edge is half-way between the last
# plateau point and the first point more than SIDE_MIN beyond it.
sim_lo = Ms[np.nonzero(cls == 0)[0][0]] - 0.125
sim_hi = Ms[np.nonzero(cls == 0)[0][-1]] + 0.125
o = np.argsort(sysM); xs, ys = sysM[o], side[o]
plateau = np.median(ys[(xs > sim_lo) & (xs < sim_hi)])
lit_m = np.nonzero((ys > plateau + me.SIDE_MIN) & (xs < -1000 / 370))[0][-1]
lit_h = np.nonzero((ys < plateau - me.SIDE_MIN) & (xs > -1000 / 370))[0][0]
real_lo, real_hi = 0.5 * (xs[lit_m] + xs[lit_m + 1]), 0.5 * (xs[lit_h - 1] + xs[lit_h])
null = -1000 / 370
EDGES = dict(sim=(sim_lo, sim_hi), real=(real_lo, real_hi))
ax.annotate("", xy=(sim_lo, -0.47), xytext=(sim_hi, -0.47), arrowprops=dict(arrowstyle="<->", color=S2, lw=1.2))
ax.text(0.5 * (sim_lo + sim_hi), -0.44, f"simulated blind band {sim_lo:+.1f} to {sim_hi:+.1f} D", color=INK, fontsize=8,
        ha="center")
ax.annotate("", xy=(real_lo, -0.57), xytext=(real_hi, -0.57), arrowprops=dict(arrowstyle="<->", color=S1, lw=1.2))
ax.text(0.5 * (real_lo + real_hi), -0.66, f"real blind band {real_lo:+.1f} to {real_hi:+.1f} D: myopic edge "
        f"{(null - real_lo) / (null - sim_lo):.1f}x as far from the null, hyperopic edge {real_hi - sim_hi:+.1f} D",
        color=INK, fontsize=8, ha="center")
ax.set_xlim(-14, 5); ax.set_ylim(-0.72, 0.55)
ax.set_xlabel("eye + trial lens refraction (D, dial minus effective lens power)")
ax.set_ylabel("crescent position (+ LED side, − far side)")
ax.legend(loc="upper right", fontsize=8, frameon=False)
ax.set_title("b  37 cm, 7 mm pupil: crescent side vs refraction. Same shape as the simulator; the myopic edge is further out.\n"
             "    Every point left of the band was taken through a trial lens, every point right of it without one.",
             loc="left", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(HERE / "fig3_physics.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- fig 4: literature
d37 = [m for m in metrics if m["dist_cm"] == 37.0]
ours37 = np.average([m["mae"] for m in d37], weights=[m["n"] for m in d37])
n37 = sum(m["n"] for m in d37)
near = [m for m in metrics if m["dist_cm"] < 37.0]
ours_near = np.average([m["mae"] for m in near], weights=[m["n"] for m in near])
n_near = sum(m["n"] for m in near)
entries = [  # label, MAE (D), ours?
    (f"This work, model eye, 37 cm\n({n37} settings, blocks A B D2 R)", ours37, True),
    (f"This work, model eye, 25-30 cm\n({n_near} settings, blocks E2 F2)", ours_near, True),
    ("Xu et al. 2022, human eyes\nNIR photorefraction, 6 LED meridians, 1 m", 0.1835, False),
    ("Varadarajan et al. 2018, human eyes\nretinal fundus photos, UK Biobank", 0.56, False),
    ("Varadarajan et al. 2018, human eyes\nretinal fundus photos, AREDS", 0.91, False),
]
fig, ax = plt.subplots(figsize=(10.5, 4.6))
yy = np.arange(len(entries))[::-1]
for i, (lab, v, ours) in zip(yy, entries):
    ax.barh(i, v, height=0.55, color=S1 if ours else "#b9b8b2", edgecolor=SURF)
    ax.text(v + 0.02, i, f"{v:.2f} D", va="center", fontsize=9, color=INK)
ax.set_yticks(yy); ax.set_yticklabels([e[0] for e in entries], fontsize=8)
ax.set_xlabel("mean absolute error of the spherical equivalent (D)")
ax.set_xlim(0, 1.15); ax.grid(axis="y", visible=False)
ax.set_title("Fig. 4  Published deep-learning refraction errors, for scale (not a like-for-like comparison)", loc="left")
fig.text(0.01, 0.01, "Different eyes, references and setups. Ours: a schematic eye against its dial (no 850 nm correction), "
         "network trained only on simulated photos.\nOthers: human eyes against autorefraction / subjective refraction, "
         "trained on thousands of real images. Classification-only studies (no MAE) are in README.md.", fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(HERE / "fig4_literature.png", dpi=150)
plt.close(fig)

for m in metrics:
    print(f"{m['block']:26s} {m['dist_cm']:.0f} cm  n {m['n']:2d}  MAE {m['mae']:.2f}  bias {m['bias']:+.2f}  "
          f"<=0.5 {100 * m['within05']:.0f}%  slope {m['slope']:.2f}  | disc centre MAE {m['mae_disc']:.2f}")
print(f"37 cm pooled MAE {ours37:.2f} D (n {n37}); 25-30 cm pooled {ours_near:.2f} D (n {n_near})")
print("band edges 37 cm 7 mm:", {k: tuple(round(float(x), 2) for x in v) for k, v in EDGES.items()})
