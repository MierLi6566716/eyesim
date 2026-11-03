"""
One picture that shows how training went and where the network gets it wrong.

  top row     error and loss per epoch (several runs can be overlaid)
  middle row  how often the crescent vanishes vs true M, error vs true M,
              predicted vs true M
  bottom row  what the network sees: one eye swept across M

  python scripts/report.py                                    # checkpoints/
  python scripts/report.py --runs checkpoints/old checkpoints   # compare runs
  python scripts/report.py --curves-only --watch 60           # live while training
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_VIS, C_DARK, C_3 = "#2a78d6", "#eb6834", "#1baf7a"
RUN_COLORS = [C_VIS, C_DARK, C_3, "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TARGET_D = 0.25

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})


def read_log(run: Path):
    with open(run / "train_log.csv") as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else None


def plot_curves(ax_err, ax_loss, runs):
    for i, run in enumerate(runs):
        log = read_log(run)
        if log is None:
            continue
        c = RUN_COLORS[i % len(RUN_COLORS)]
        tag = f"{run.name}: " if len(runs) > 1 else ""
        ax_err.plot(log["epoch"], log["M_mae"], color=c, label=f"{tag}all frames")
        ax_err.plot(log["epoch"], log["M_mae_visible"], color=c, ls="--", label=f"{tag}crescent visible")
        ax_loss.plot(log["epoch"], log["train_loss"], color=c, ls=":", label=f"{tag}train")
        ax_loss.plot(log["epoch"], log["val_loss"], color=c, label=f"{tag}val")
        k = int(np.argmin(log["val_loss"]))
        ax_err.annotate(f"{log['M_mae_visible'][k]:.2f} D", (log["epoch"][k], log["M_mae_visible"][k]),
                        textcoords="offset points", xytext=(0, -12), ha="center", color=INK, fontsize=8)
    ax_err.axhline(TARGET_D, color=INK2, lw=0.8, ls="-.")
    ax_err.text(ax_err.get_xlim()[0], TARGET_D, " 0.25 D target", va="bottom", color=INK2, fontsize=8)
    ax_err.set(title="Validation M error per epoch", xlabel="epoch", ylabel="M mean abs error (D)")
    ax_err.set_ylim(bottom=0)
    ax_err.legend(fontsize=8)
    ax_loss.set(title="Loss (train ≈ val means under-fitting)", xlabel="epoch", ylabel="Huber loss")
    ax_loss.legend(fontsize=8)


def binned(x, y, edges, fn=np.median):
    idx = np.digitize(x, edges) - 1
    out = np.full(len(edges) - 1, np.nan)
    for b in range(len(out)):
        m = idx == b
        if m.sum() >= 20:
            out[b] = fn(y[m])
    return out


def plot_errors(axes, run, data, batch=1024):
    import torch
    from eyesim.cnn import PupilNet
    from train import VISIBLE_PEAK, Split, predict

    device = (torch.device("cuda") if torch.cuda.is_available() else
              torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    ck = torch.load(run / "best.pt", map_location=device, weights_only=False)
    model = PupilNet(**ck["model_kwargs"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    test = Split(data, "test")
    pred = predict(model, test, batch, device).numpy()
    true = test.labels.numpy()
    vis = test.meta[:, 7] >= VISIBLE_PEAK
    near = test.meta[:, 0] < 450
    err = np.abs(pred[:, 0] - true[:, 0])

    edges = np.arange(-9.0, 5.01, 0.5)
    mid = 0.5 * (edges[1:] + edges[:-1])

    ax = axes[0]
    for m, c, name in [(near, C_VIS, "d < 45 cm"), (~near, C_DARK, "d >= 45 cm")]:
        ax.plot(mid, 100 * binned(true[m, 0], (~vis[m]).astype(float), edges, np.mean), color=c, label=name)
    ax.set(title="Where the reflex disappears", xlabel="true M (D)", ylabel="dark-zone frames (%)", ylim=(0, 105))
    ax.legend(fontsize=8)

    ax = axes[1]
    for m, c, name in [(vis, C_VIS, "crescent visible"), (~vis, C_DARK, "dark zone")]:
        ax.plot(mid, binned(true[m, 0], err[m], edges), color=c, label=name)
        q1 = binned(true[m, 0], err[m], edges, lambda v: np.percentile(v, 25))
        q3 = binned(true[m, 0], err[m], edges, lambda v: np.percentile(v, 75))
        ax.fill_between(mid, q1, q3, color=c, alpha=0.15, lw=0)
    ax.axhline(TARGET_D, color=INK2, lw=0.8, ls="-.")
    ax.set(title="M error vs true M (median, IQR band)", xlabel="true M (D)", ylabel="|M error| (D)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.hexbin(true[vis, 0], pred[vis, 0], gridsize=45, cmap="Blues", mincnt=1, bins="log")
    lim = [-9.5, 5.5]
    ax.plot(lim, lim, color=INK, lw=0.8)
    ax.set(xlim=lim, ylim=lim, title=f"Predicted vs true M, visible frames (MAE {err[vis].mean():.2f} D)",
           xlabel="true M (D)", ylabel="predicted M (D)")
    ax.grid(False)
    return {"M_mae_all": float(err.mean()), "M_mae_visible": float(err[vis].mean()),
            "dark_share": float((~vis).mean())}


def plot_frames(fig, gs_row, d_mm=370.0, ecc_mm=10.0):
    from eyesim.photorefraction import Eye, Geometry, Scene, render_pair

    Ms = [-8, -6, -4, -3, -2.7, -2, -1, 0, 1, 2, 4]
    sub = gs_row.subgridspec(1, len(Ms), wspace=0.05)
    geom, scene = Geometry(d_mm=d_mm, ecc_mm=ecc_mm), Scene(glint_amp=1.0, red_reflex=0.03)
    rng = np.random.default_rng(0)
    for i, M in enumerate(Ms):
        ax = fig.add_subplot(sub[0, i])
        img, peak = render_pair(Eye(M=M, pupil_r_mm=3.0, scatter_mrad=2.0), geom, scene, rng, return_peak=True)
        ax.imshow(img[..., 0], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(C_DARK if peak < 0.15 else GRID); s.set_linewidth(2)
        ax.set_title(f"M={M:+g}", fontsize=8, fontweight="normal", loc="center", color=INK2)
        if i == 0:
            ax.set_ylabel("H LED", fontsize=8)
    fig.text(0.02, gs_row.get_position(fig).y1 + 0.012,
             f"What the CNN sees: one 6 mm pupil swept across M at d={d_mm:.0f} mm, ecc={ecc_mm:.0f} mm "
             "(orange border = dark zone)", fontsize=10, fontweight="bold", color=INK)


def render(runs, data, out, curves_only):
    if curves_only:
        fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))
        plot_curves(a, b, runs)
    else:
        fig = plt.figure(figsize=(15, 11))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.42], hspace=0.45, wspace=0.28,
                              left=0.05, right=0.98, top=0.93, bottom=0.03)
        plot_curves(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), runs)
        stats = plot_errors([fig.add_subplot(gs[1, k]) for k in range(3)], runs[-1], data)
        ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
        lines = [f"run: {runs[-1]}", f"test M MAE, all frames:   {stats['M_mae_all']:.2f} D",
                 f"test M MAE, visible:      {stats['M_mae_visible']:.2f} D",
                 f"frames in dark zone:      {100 * stats['dark_share']:.0f} %", "",
                 "Read it like this:", "· top-left flat → more epochs won't help",
                 "· train ≈ val loss → model or data limit, not overfitting",
                 "· orange curve high where dark-zone % is high",
                 "  → no light, no information: fix the instrument,", "    not the network"]
        ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9, color=INK)
        plot_frames(fig, gs[2, :])
        fig.suptitle("PupilNet training report", x=0.05, ha="left",
                     fontsize=13, fontweight="bold", color=INK)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["checkpoints"],
                    help="run folders; the last one is evaluated on the test set")
    ap.add_argument("--data", default="data/train_set")
    ap.add_argument("--out", default=None, help="default: <last run>/report.png")
    ap.add_argument("--curves-only", action="store_true", help="skip the test-set evaluation")
    ap.add_argument("--watch", type=float, default=0, metavar="SECONDS",
                    help="redraw every SECONDS (use with --curves-only while training)")
    args = ap.parse_args()
    runs = [Path(r) for r in args.runs]
    out = Path(args.out) if args.out else runs[-1] / ("curves.png" if args.curves_only else "report.png")
    while True:
        render(runs, Path(args.data), out, args.curves_only)
        print(f"wrote {out}", flush=True)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
