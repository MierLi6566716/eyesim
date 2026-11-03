"""
Shows the whole loop for one simulated eye in one picture, then runs a group
of random eyes through it and plots the leftover error after each stage.

For one eye (default: S=-2.25, C=-0.75, axis=30):

  Row 1  SEE     — test scene, the eye's PSF, what the uncorrected eye sees
  Row 2  SENSE   — photorefraction frames, CNN estimate vs ground truth
  Row 3  CORRECT — view after one-shot CNN correction, closed-loop
                   refinement convergence, final corrected view

Outputs
-------
  examples/output/pipeline_storyboard.png
  examples/output/pipeline_cohort.png

Usage
-----
  python examples/full_pipeline.py                         # defaults
  python examples/full_pipeline.py --S -3 --C -1 --axis 70
  python examples/full_pipeline.py --cohort 30             # cohort size
  python examples/full_pipeline.py --mock                  # no checkpoint/torch
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eyesim.optimize import power_vector_to_prescription
from eyesim.pipeline import run_full_pipeline

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)

DARK = "#1a1a2e"


def sample_prescription(rng):
    """A random everyday prescription (S, C, axis)."""
    return (float(np.clip(rng.normal(-1.0, 2.0), -6.0, 4.0)),
            float(-abs(rng.normal(0.0, 0.75))), float(rng.uniform(0.0, 180.0)))


def make_siemens_star(n: int = 256, spokes: int = 24) -> np.ndarray:
    """Radial Siemens star: isotropic edge content at every orientation,
    the classic focus target — ideal for the sharpness-based loop."""
    x = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(x, x)
    phi = np.arctan2(yy, xx)
    r = np.sqrt(xx ** 2 + yy ** 2)
    star = 0.5 + 0.5 * np.sign(np.sin(spokes * phi))
    star[r > 0.95] = 1.0
    star[r < 0.05] = 0.0
    return star.astype(np.float32)


def _get_estimator(args, rng):
    if args.mock:
        print("Using MOCK estimator (truth + noise) — no checkpoint needed.")
        state = {}

        def make(truth):
            def est(_frames):
                m, j0, j45 = truth
                return (m + rng.normal(0, 0.15),
                        j0 + rng.normal(0, 0.05),
                        j45 + rng.normal(0, 0.05))
            return est
        state["make"] = make
        return state
    from eyesim.pipeline import load_estimator
    est = load_estimator(args.checkpoint)
    return {"make": lambda truth: est}


def imshow(ax, img, title, cmap="gray", vmax=1.0):
    ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def psf_show(ax, psf, title, zoom=None):
    """sqrt-stretched PSF; zoom crops the centre so small kernels stay visible."""
    if zoom is not None and zoom < psf.shape[0]:
        c, h = psf.shape[0] // 2, zoom // 2
        psf = psf[c - h:c + h, c - h:c + h]
    p = psf / psf.max()
    ax.imshow(p ** 0.5, cmap="magma", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def storyboard(res, path):
    fig = plt.figure(figsize=(17, 10.5), facecolor="white")
    gs = GridSpec(3, 5, figure=fig, hspace=0.38, wspace=0.15,
                  left=0.055, right=0.985, top=0.90, bottom=0.05)

    S, C, ax_t = res["S_true"], res["C_true"], res["axis_true"]
    fig.suptitle(
        "eyesim — sense → estimate → correct (rows 1-2 + left of row 3: the wearable's "
        "real pipeline)  |  refine (right of row 3: benchtop-only, see caption)\n"
        f"eye under test:  S={S:+.2f} D   C={C:+.2f} D   axis={ax_t:.0f}°"
        f"   (power vector M={res['M_true']:+.2f}, J0={res['J0_true']:+.2f}, "
        f"J45={res['J45_true']:+.2f})",
        fontsize=12, fontweight="bold")

    row_labels = ["① SEE\nuncorrected eye", "② SENSE\nphotorefraction + CNN",
                  "③ CORRECT (wearable) +\n④ REFINE (benchtop only)"]
    for r, lab in enumerate(row_labels):
        fig.text(0.012, 0.80 - r * 0.315, lab, fontsize=10, fontweight="bold",
                 color=DARK, rotation=90, va="center", ha="center")

    # ---- Row 1 : SEE -------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    imshow(ax, res["scene"], "Test target (Siemens star)")
    ax = fig.add_subplot(gs[0, 1])
    psf_show(ax, res["psf_eye"], "Eye's blur kernel (PSF)")
    ax = fig.add_subplot(gs[0, 2])
    imshow(ax, res["seen_uncorrected"],
           f"What the eye sees — sharpness {res['sharpness_uncorrected']:.2e}")
    ax = fig.add_subplot(gs[0, 3:])
    ax.axis("off")
    ax.text(0.02, 0.85, "The eye's refractive error is unknown to the device.",
            fontsize=10, va="top")
    ax.text(0.02, 0.62,
            f"Uncorrected residual error: "
            f"{res['summary']['residual_D_uncorrected']:.2f} D (power-vector norm)\n\n"
            "The device must recover it from sensor data alone,\n"
            "then drive the tunable lens to cancel it.",
            fontsize=10, va="top")

    # ---- Row 2 : SENSE -----------------------------------------------------
    ph = res["photoref"]
    ax = fig.add_subplot(gs[1, 0])
    imshow(ax, ph[..., 0], "IR pupil image — horizontal LED", cmap="inferno")
    ax = fig.add_subplot(gs[1, 1])
    imshow(ax, ph[..., 1], "IR pupil image — vertical LED", cmap="inferno")

    ax = fig.add_subplot(gs[1, 2:4])
    names = ["M", "J0", "J45"]
    truth = [res["M_true"], res["J0_true"], res["J45_true"]]
    est = [res["M_cnn"], res["J0_cnn"], res["J45_cnn"]]
    xpos = np.arange(3)
    ax.bar(xpos - 0.17, truth, width=0.34, label="ground truth", color="#3a6ea5")
    ax.bar(xpos + 0.17, est, width=0.34, label="CNN estimate", color="#e07b39")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(xpos); ax.set_xticklabels(names)
    ax.set_ylabel("dioptres")
    ax.set_title(f"CNN estimate from {res['photoref_frames'].shape[0]} averaged frames",
                 fontsize=9)
    ax.legend(fontsize=8)
    for i, (t, e) in enumerate(zip(truth, est)):
        ax.text(i, max(t, e, 0) + 0.08, f"Δ{abs(t - e):.2f}", ha="center", fontsize=8)

    ax = fig.add_subplot(gs[1, 4])
    ax.axis("off")
    ax.text(0.02, 0.9,
            "CNN reads the crescent pattern:\n"
            f"  Ŝ = {res['S_cnn']:+.2f} D\n"
            f"  Ĉ = {res['C_cnn']:+.2f} D\n"
            f"  axis = {res['axis_cnn']:.0f}°\n\n"
            "Lens command = −estimate\n(open-loop, single shot)",
            fontsize=10, va="top", family="monospace")

    # ---- Row 3 : CORRECT & REFINE -----------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    r_cnn = res["summary"]["residual_D_after_cnn"]
    imshow(ax, res["seen_cnn"], f"After CNN correction\nresidual {r_cnn:.2f} D")
    ax = fig.add_subplot(gs[2, 1])
    psf_show(ax, res["psf_cnn"], "Residual PSF after CNN (zoomed)", zoom=32)

    ax = fig.add_subplot(gs[2, 2])
    hist = res["loop"]["history"]
    sharp = np.array([h[3] for h in hist])
    evals = np.arange(1, len(sharp) + 1)
    ax.plot(evals, sharp, ".", color="#e07b39", ms=3, alpha=0.45,
            label="each lens trial")
    ax.plot(evals, np.maximum.accumulate(sharp), "-", color="#c1440e", lw=1.6,
            label="best so far")
    ax.axhline(res["sharpness_cnn"], color="#3a6ea5", ls="--", lw=1,
               label="after CNN only")
    ax.set_xlabel("lens adjustment #", fontsize=8)
    ax.set_ylabel("sharpness", fontsize=8)
    ax.set_title(f"Benchtop-only refinement\n({len(sharp)} evals — needs a scene camera "
                 f"the wearable doesn't have)", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[2, 3])
    r_fin = res["summary"]["residual_D_after_loop"]
    imshow(ax, res["seen_final"], f"After benchtop refinement\nresidual {r_fin:.2f} D")
    ax = fig.add_subplot(gs[2, 4])
    psf_show(ax, res["psf_final"], "Final PSF ≈ diffraction spot (zoomed)", zoom=32)

    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Saved: {path}")


def cohort_figure(results, path):
    stages = ["uncorrected", "after CNN\n(wearable, real)", "after refinement\n(benchtop only)"]
    data = np.array([
        [r["summary"]["residual_D_uncorrected"],
         r["summary"]["residual_D_after_cnn"],
         r["summary"]["residual_D_after_loop"]] for r in results
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    fig.suptitle(f"Pipeline residual refractive error across {len(results)} random eyes",
                 fontsize=12, fontweight="bold")

    ax = axes[0]
    for row in data:
        ax.plot([0, 1, 2], row, "-", color="gray", alpha=0.35, lw=0.8, zorder=1)
    for j, c in enumerate(["#3a6ea5", "#e07b39", "#2a9d8f"]):
        ax.scatter(np.full(len(data), j), data[:, j], s=22, color=c, zorder=2)
    ax.set_yscale("log")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel("residual error  |ΔM,ΔJ0,ΔJ45|  (D, log scale)")
    ax.axhline(0.25, color="red", ls="--", lw=1)
    ax.text(2.35, 0.25, "0.25 D\nclinical step", color="red", fontsize=8, va="center")
    med = np.median(data, axis=0)
    for j, m in enumerate(med):
        ax.text(j, m, f"  med {m:.2f} D", fontsize=8, fontweight="bold")
    ax.set_title("Each line = one eye", fontsize=10)

    ax = axes[1]
    within = (data < 0.25).mean(axis=0) * 100
    bars = ax.bar(range(3), within, color=["#3a6ea5", "#e07b39", "#2a9d8f"], width=0.55)
    ax.bar_label(bars, fmt="%.0f%%", fontsize=10, fontweight="bold")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylim(0, 108)
    ax.set_ylabel("% of eyes within 0.25 D")
    ax.set_title("Clinical-step success rate", fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=float, default=-2.25)
    ap.add_argument("--C", type=float, default=-0.75)
    ap.add_argument("--axis", type=float, default=30.0)
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--cohort", type=int, default=20, help="eyes in cohort figure (0 = skip)")
    ap.add_argument("--frames", type=int, default=4, help="sensor frames averaged per eye")
    ap.add_argument("--mock", action="store_true",
                    help="use a mock estimator instead of the CNN checkpoint")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    est_factory = _get_estimator(args, rng)
    scene = make_siemens_star(256)

    from eyesim.optics import prescription_to_power_vector
    print(f"Running full pipeline for S={args.S} C={args.C} axis={args.axis} ...")
    truth = prescription_to_power_vector(args.S, args.C, args.axis)
    # refine=True here specifically because this script's storyboard is
    # demonstrating both capabilities side by side — the wearable's real
    # pipeline (stages 1-4) and the separate benchtop-only refinement
    # (stage 5). See eyesim/pipeline.py's module docstring.
    res = run_full_pipeline(args.S, args.C, args.axis,
                            est_factory["make"](truth), scene,
                            n_frames=args.frames, seed=args.seed, refine=True)
    res["scene"] = scene
    s = res["summary"]
    print(f"  residual: {s['residual_D_uncorrected']:.2f} D -> "
          f"{s['residual_D_after_cnn']:.2f} D (CNN) -> "
          f"{s['residual_D_after_loop']:.2f} D (loop, {s['loop_evals']} evals)")
    storyboard(res, os.path.join(OUT, "pipeline_storyboard.png"))

    if args.cohort > 0:
        print(f"Running cohort of {args.cohort} random eyes ...")
        results = []
        for i in range(args.cohort):
            S, C, axis = sample_prescription(rng)
            truth = prescription_to_power_vector(S, C, axis)
            r = run_full_pipeline(S, C, axis, est_factory["make"](truth), scene,
                                  n_frames=args.frames, seed=args.seed + i, refine=True)
            results.append(r)
            print(f"  eye {i + 1:2d}/{args.cohort}  S={S:+5.2f} C={C:+5.2f} ax={axis:5.1f}°"
                  f"   {r['summary']['residual_D_uncorrected']:5.2f} D -> "
                  f"{r['summary']['residual_D_after_cnn']:.2f} -> "
                  f"{r['summary']['residual_D_after_loop']:.2f}")
        cohort_figure(results, os.path.join(OUT, "pipeline_cohort.png"))


if __name__ == "__main__":
    main()
