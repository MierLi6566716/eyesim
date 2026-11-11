"""
Makes the overview figures in this folder from the model-eye results.

  /opt/anaconda3/bin/python model_eye/overview/make_figures.py

Reads   model_eye/results/<block>/results.csv   (from scripts/model_eye_report.py)
        model_eye/sessions/<block>/<session>/    (the photos, for figs 2 and 3)
Writes  fig1_accuracy.png  fig2_real_vs_simulated.png  fig3_physics.png  fig4_literature.png
        tables/metrics.csv

Network: checkpoints/wide_m_only/best.pt, the best of the networks tested on these photos
(tables/network_comparison.csv). "Error" = network reading minus the dial setting, no corrections.
"""

import contextlib
import csv
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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
S1, S2 = "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "bold"})

BLOCKS = {  # block: title
    "A_37cm_plus4_7mm": "A: 37 cm, +4 lens",
    "D2_37cm_plus5_5mm": "D2: 37 cm, +5 lens, 5 mm pupil",
    "R_37cm_plus4_7mm_repeat": "R: A again, eye taken off and put back",
    "B_37cm_plus8_7mm": "B: 37 cm, +8 lens",
    "E2_30cm_plus5_7mm": "E2: 30 cm, +5 lens",
    "F2_25cm_plus6_7mm": "F2: 25 cm, +6 lens",
}


def rows(b, net=NET):
    return [r for r in csv.DictReader(open(ROOT / "model_eye/results" / b / "results.csv")) if r["network"] == net]


def arr(rs, key):
    return np.array([float(r[key]) for r in rs])


def lens_eff(P, vertex=5.0):
    return P / (1 - vertex * 1e-3 * P)


# ---------------------------------------------------------------- metrics
metrics = []
for b in BLOCKS:
    use = [r for r in rows(b) if r["crescent_seen"] == "True"]
    d, e = arr(use, "dial_d"), arr(use, "net_eye_ir") - arr(use, "dial_d")
    metrics.append(dict(block=b, dist_cm=float(use[0]["dist_cm"]), n=len(use), mae=float(np.abs(e).mean()),
                        bias=float(e.mean()), within05=float((np.abs(e) <= 0.5).mean()),
                        slope=float(np.polyfit(d, d + e, 1)[0])))
with open(HERE / "tables" / "metrics.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(metrics[0]))
    w.writeheader()
    for m in metrics:
        w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
M = {m["block"]: m for m in metrics}

# ---------------------------------------------------------------- fig 1: accuracy
fig, axes = plt.subplots(2, 3, figsize=(13, 8.4))
for ax, (b, title) in zip(axes.flat, BLOCKS.items()):
    rs = rows(b)
    x, y = arr(rs, "dial_d"), arr(rs, "net_eye_ir")
    s = np.array([r["crescent_seen"] == "True" for r in rs])
    t = np.array([-5.6, 0.3])
    ax.fill_between(t, t - 0.5, t + 0.5, color=S1, alpha=0.12, lw=0, label="within 0.5 D of the truth")
    ax.plot(t, t, color=INK2, lw=1, ls="--", label="perfect reading")
    ax.scatter(x[s], y[s], s=46, color=S1, edgecolors=SURF, lw=1.5, zorder=4, label="crescent in the photo")
    ax.scatter(x[~s], y[~s], s=40, facecolors=SURF, edgecolors="#9a998f", lw=1, zorder=3,
               label="no crescent (blind band):\nnothing to measure")
    m = M[b]
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.text(0.97, 0.04, f"average error {m['mae']:.2f} D\n{100 * m['within05']:.0f} % within 0.5 D",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc=SURF, ec=GRID))
    ax.set_xlim(-5.6, 0.3)
    ax.set_ylim(-6.5, 4.5)
    ax.set_xlabel("true refraction set on the model eye (D)")
    ax.set_ylabel("network reading (D)")
axes.flat[0].legend(loc="upper left", fontsize=8, frameon=False)
fig.suptitle("Fig. 1  What the network reads vs the true refraction (each dot = one setting, 20 photos)",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.text(0.01, 0.935, "Top row: 37 cm with a +4/+5 lens - readings sit on the perfect line.   "
         "Bottom row: the +8 lens and the closer distances read 0.6-1.0 D too weak.", fontsize=10, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(HERE / "fig1_accuracy.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- fig 2: real vs simulated photos
import measure_eye as me  # noqa: E402
import torch  # noqa: E402
from model_eye_report import load_net  # noqa: E402
from eyesim.cnn import normalize_frames  # noqa: E402
from eyesim.photorefraction import Eye, Geometry, Scene, render_pair  # noqa: E402

model, cfg = load_net(ROOT / NET)
margs = SimpleNamespace(hflip=False, vflip=False, brightness="glint", reclick=False, focal_mm=4.74, pixel_um=1.4,
                        refine=False, ecc_mm=None)
CASES = [  # block, dial
    ("C_37cm_nolens_7mm", "+4"), ("C_37cm_nolens_7mm", "+0"), ("A_37cm_plus4_7mm", "-2"),
    ("A_37cm_plus4_7mm", "-3"), ("A_37cm_plus4_7mm", "-5"), ("B_37cm_plus8_7mm", "-5"),
]
rng = np.random.default_rng(1)
fig, axes = plt.subplots(len(CASES), 4, figsize=(8.6, 2.25 * len(CASES)))
for row, (b, dial) in zip(axes, CASES):
    s = next((ROOT / "model_eye/sessions" / b).glob(f"*true{dial}-p*"))
    with contextlib.redirect_stdout(io.StringIO()):
        r = me.measure(s, model, cfg, margs)
    meta = r["meta"]
    true_sys = meta["true_rx"] - lens_eff(meta.get("lens_d", 0.0))
    sim = render_pair(Eye(M=true_sys, pupil_r_mm=meta["pupil_mm"] / 2, scatter_mrad=2.0, sa_d_per_mm2=0.12),
                      Geometry(d_mm=meta["dist_mm"], ecc_mm=meta["ecc_mm"], aperture_r_mm=1.3),
                      Scene(fill=me.FILL, glint_amp=1.0, glint_pos_frac=(-0.25, 0.25), red_reflex=0.0,
                            surround=0.0, blur_px=1.0, read_noise=0.004, shot_noise=0.008), rng).transpose(2, 0, 1)
    real = normalize_frames(torch.from_numpy(np.asarray(r["centred"], np.float32))[None])[0].numpy()
    simn = normalize_frames(torch.from_numpy(np.asarray(sim, np.float32))[None])[0].numpy()
    for k, (img, lab) in enumerate([(real[0], "photo, H LED"), (simn[0], "simulated, H LED"),
                                    (real[1], "photo, V LED"), (simn[1], "simulated, V LED")]):
        a = row[k]
        a.imshow(img, cmap="gray", vmin=0, vmax=1.2)
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_color(S1 if "photo" in lab else S2); sp.set_linewidth(2)
        if row is axes[0]:
            a.set_title(lab, fontsize=9.5, color=INK)
    lens = meta.get("lens_d", 0.0)
    row[0].set_ylabel(f"eye + lens {true_sys:+.1f} D\n(dial {meta['true_rx']:+g}" + (f", +{lens:g} lens)" if lens else ")")
                      + f"\nnetwork: {r['sys_ir'][0]:+.1f} D", fontsize=8.5)
fig.suptitle("Fig. 2  Real photos (blue frame) next to the simulator\nat the same refraction (orange frame)",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.text(0.01, 0.005, "Same crop and brightness scaling the network uses. H LED is right of the camera, V LED above it.\n"
         "Crescent side and growth match. Real crescents are softer, and shorter top-to-bottom than the simulated ones.",
         fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.035, 1, 0.94))
fig.savefig(HERE / "fig2_real_vs_simulated.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------- fig 3: physics
def mean_frame(s, ch):
    meta = json.loads((s / "meta.json").read_text())
    acc = 0
    for k in range(meta["shots"]):
        d = me.load_frame(me.frame_path(s, k, "dark"), meta["white_level"], False, False)
        acc = acc + np.clip(me.load_frame(me.frame_path(s, k, ch), meta["white_level"], False, False) - d, 0, None)
    return acc / meta["shots"]


fig = plt.figure(figsize=(13, 7.6))
gs = fig.add_gridspec(2, 7, height_ratios=[1, 1.55])
blk = ROOT / "model_eye/sessions/A_37cm_plus4_7mm"
imgs = []
for d in ["+0", "-1", "-1.5", "-2", "-3", "-4", "-5"]:
    s = next(blk.glob(f"*true{d}-p*"))
    cx, cy, rr = json.loads((s / "clicks.json").read_text())["pupil_xy_r"]
    h = int(1.6 * rr)
    imgs.append((float(d), mean_frame(s, "H")[int(cy) - h:int(cy) + h, int(cx) - h:int(cx) + h], rr, h))
vmax = max(np.percentile(im, 99.0) for _, im, _, _ in imgs)
for k, (d, im, rr, h) in enumerate(imgs):
    ax = fig.add_subplot(gs[0, k])
    ax.imshow(im, cmap="gray", vmax=vmax)
    ax.add_patch(plt.Circle((h, h), rr, fill=False, color=S1, lw=0.8))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(f"eye + lens\n{d - lens_eff(4):+.1f} D", fontsize=9)
fig.text(0.01, 0.975, "a  Block A photos, H LED (to the right). Blue circle = the 7 mm pupil. Close to the camera's focus the "
         "pupil is dark;\n    further away a crescent appears on the LED side and grows. The network reads that growth.",
         fontsize=10, fontweight="bold", va="top")

ax = fig.add_subplot(gs[1, :])
rs = rows("A_37cm_plus4_7mm") + rows("B_37cm_plus8_7mm") + rows("C_37cm_nolens_7mm")
sysM = np.array([float(r["dial_d"]) - lens_eff(float(r["lens_d"])) for r in rs])
side = np.array([0.5 * (float(r["side_H"]) + float(r["side_V"])) for r in rs])
Ms, cls = me.side_curve(370, 10.0, 2.6, 3.5)
ax.step(Ms, 0.35 * cls, where="mid", color=S2, lw=2, label="simulator")
ax.scatter(sysM, side, s=30, color=S1, edgecolors=SURF, lw=1, zorder=3, label="real photos (blocks A, B, C)")
null = -1000 / 370
ax.axvline(null, color=INK2, lw=0.8, ls="--")
ax.text(null + 0.15, 0.46, "eye focused exactly\non the camera", fontsize=8, color=INK2)
ax.text(-12.5, 0.43, "crescent on the LED side\n(eye more myopic than the camera focus)", fontsize=9, color=INK)
ax.text(2.2, -0.15, "crescent on the far side\n(more hyperopic)", fontsize=9, color=INK)
sim_lo = Ms[np.nonzero(cls == 0)[0][0]] - 0.125
sim_hi = Ms[np.nonzero(cls == 0)[0][-1]] + 0.125
ax.axvspan(sim_lo, sim_hi, color=S2, alpha=0.07, lw=0)
ax.text(0.5 * (sim_lo + sim_hi), -0.5, "blind band: no crescent, so the pupil carries no reading.\n"
        "A trial lens moves the eye out of it.", ha="center", fontsize=9, color=INK)
ax.set_xlim(-14, 5); ax.set_ylim(-0.62, 0.58)
ax.set_xlabel("refraction of eye + trial lens (D)")
ax.set_ylabel("where the light sits in the pupil\n(+ LED side,  − far side)")
ax.legend(loc="upper right", fontsize=9, frameon=False)
ax.set_title("b  37 cm, 7 mm pupil: same pattern as the simulator. Real edges are 0.3-0.8 D further out "
             "(probably: a faint first crescent is lost in real noise)", loc="left", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(HERE / "fig3_physics.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- fig 4: literature
good = [m for m in metrics if m["block"] in ("A_37cm_plus4_7mm", "D2_37cm_plus5_5mm", "R_37cm_plus4_7mm_repeat")]
all_lens = metrics
ours_good = np.average([m["mae"] for m in good], weights=[m["n"] for m in good])
ours_all = np.average([m["mae"] for m in all_lens], weights=[m["n"] for m in all_lens])
entries = [
    (f"This work: model eye, 37 cm, +4/+5 lens\n({sum(m['n'] for m in good)} settings)", ours_good, True),
    (f"This work: all lens blocks incl. +8 lens, 25-30 cm\n({sum(m['n'] for m in all_lens)} settings)", ours_all, True),
    ("Xu et al. 2022: human eyes, infrared\nphotorefraction, 6 LED directions, 1 m", 0.1835, False),
    ("Varadarajan et al. 2018: human eyes,\nretina photos (UK Biobank)", 0.56, False),
    ("Varadarajan et al. 2018: human eyes,\nretina photos (AREDS)", 0.91, False),
]
fig, ax = plt.subplots(figsize=(10.5, 4.8))
yy = np.arange(len(entries))[::-1]
for i, (lab, v, ours) in zip(yy, entries):
    ax.barh(i, v, height=0.55, color=S1 if ours else "#b9b8b2", edgecolor=SURF)
    ax.text(v + 0.02, i, f"{v:.2f} D", va="center", fontsize=10, color=INK)
ax.set_yticks(yy); ax.set_yticklabels([e[0] for e in entries], fontsize=9)
ax.set_xlabel("average error of the measured refraction (D)")
ax.set_xlim(0, 1.1); ax.grid(axis="y", visible=False)
ax.set_title("Fig. 4  Published deep-learning refraction results, for scale", loc="left")
fig.text(0.01, 0.01, "Not like-for-like: ours is one model eye against its dial, with a network trained only on simulated "
         "photos; the others are\nhuman eyes against clinical refraction, with networks trained on thousands of real "
         "photos. Sources in README.md.", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(HERE / "fig4_literature.png", dpi=150)
plt.close(fig)

for m in metrics:
    print(f"{m['block']:26s} n {m['n']:2d}  error {m['mae']:.2f}  bias {m['bias']:+.2f}  within 0.5 {100 * m['within05']:.0f}%")
print(f"A+D2+R {ours_good:.2f} D; all lens blocks {ours_all:.2f} D")
