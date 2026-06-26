"""
Visualize the full eyesim pipeline: Steps 1–3 + Step 4 preview.

Saves four PNG files to examples/output/:

  step1_eye_blur.png        — grid of what different prescriptions look like
  step2_predistort.png      — pre-distortion display chain
  step3_convergence.png     — optimizer sharpness convergence curve
  step4_photorefraction.png — synthetic photorefraction images at different Rx

Usage
-----
  python examples/visualize_pipeline.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eyesim.optics import psf_from_prescription, blur_image, simulate_vision
from eyesim.predistort import predistort, simulate_display_chain
from eyesim.optimize import run_closed_loop
from eyesim.photorefraction import render_single

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Shared test image: Snellen-like chart (reused from simulate.py)
# ─────────────────────────────────────────────────────────────────────────────

def make_chart(n=256):
    img = np.ones((n, n))
    y = 30
    for size in [32, 22, 16, 11, 8, 6]:
        x = 30
        while x + size < n - 30:
            img[y : y + size, x : x + size] = 0.0
            x += size * 2
        y += size + 14
        if y > n - 30:
            break
    return img.astype(np.float32)


CHART = make_chart(256)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — what different prescriptions look like
# ─────────────────────────────────────────────────────────────────────────────

def plot_step1():
    cases = [
        (0.0,  0.0,  0.0, "Emmetropic\n(no error)"),
        (-1.5, 0.0,  0.0, "−1.5 D sphere\n(mild myopia)"),
        (-3.0, 0.0,  0.0, "−3.0 D sphere\n(moderate myopia)"),
        (-1.5, -0.5, 0.0, "−1.5/−0.5/0°\n(myopia + astig)"),
        (-1.5, -0.5, 45.0,"−1.5/−0.5/45°\n(oblique astig)"),
        (-1.5, -0.5, 90.0,"−1.5/−0.5/90°\n(with-the-rule)"),
        (-5.0, 0.0,  0.0, "−5.0 D sphere\n(high myopia)"),
        (-1.5, -1.5, 30.0,"−1.5/−1.5/30°\n(mixed)"),
        (1.0,  0.0,  0.0, "+1.0 D sphere\n(hyperopia)"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle("Step 1 — How an eye with each prescription sees a visual chart",
                 fontsize=13, fontweight="bold")

    for ax, (S, C, axis, label) in zip(axes.flat, cases):
        seen, _ = simulate_vision(CHART, S, C, axis, pupil_radius_mm=2.0,
                                  grid=128, psf_crop=48)
        ax.imshow(seen, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(label, fontsize=8.5)
        ax.axis("off")

    fig.tight_layout()
    path = os.path.join(OUT, "step1_eye_blur.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — pre-distortion display chain
# ─────────────────────────────────────────────────────────────────────────────

def plot_step2():
    S, C, axis = -1.5, -0.5, 30.0
    chain = simulate_display_chain(CHART, S=S, C=C, theta_deg=axis,
                                   grid=128, psf_crop=48, noise_power=1e-3)

    panels = [
        (chain["original"],       "Original\n(what was intended)"),
        (chain["predistorted"],   "Pre-distorted display\n(looks garbled to a normal eye)"),
        (chain["seen_corrected"], f"Target eye sees\nS={S} C={C} axis={axis}°"),
        (chain["seen_raw"],       "Target eye without correction\n(blurred)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle("Step 2 — Pre-distortion display: the blurry eye sees a sharp image",
                 fontsize=12, fontweight="bold")
    for ax, (img, label) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    path = os.path.join(OUT, "step2_predistort.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — optimizer convergence
# ─────────────────────────────────────────────────────────────────────────────

def plot_step3():
    # Isotropic test image for the optimizer (avoids directional local optima)
    n = 128
    x = np.arange(n)
    xx, yy = np.meshgrid(x, x)
    img_iso = np.zeros((n, n))
    for angle in range(0, 180, 30):
        rad = np.radians(angle)
        img_iso += np.sin(2 * np.pi * (4 * np.cos(rad) * xx / n + 4 * np.sin(rad) * yy / n))
    img_iso = np.clip(img_iso / np.abs(img_iso).max() * 0.35 + 0.5, 0, 1).astype(np.float32)

    # Run optimizer for two prescriptions
    cases = [
        (-1.5,  0.0,  0.0, "−1.5 D sphere"),
        (-1.5, -0.5, 30.0, "−1.5/−0.5/30°"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Step 3 — Closed-loop optimizer convergence",
                 fontsize=12, fontweight="bold")

    colors_scan = "steelblue"
    colors_nm   = "darkorange"

    for ax, (S, C, axis, label) in zip(axes, cases):
        res = run_closed_loop(S, C, axis, img_iso, grid=64, psf_crop=32, max_iter=400)
        history = res["history"]

        # Stage 1: coarse scan (first 21 evaluations)
        n_scan = sum(1 for h in history if abs(h[0] - round(h[0] * 2) / 2) < 1e-9 and
                     abs(h[1] - round(history[0][1], 8)) < 1e-8)
        # Just split at first Nelder-Mead evaluation (first eval with non-half-step M)
        scan_s  = [h[3] for h in history if abs(h[0] % 0.5) < 1e-8][:21]
        nm_s    = [h[3] for h in history[len(scan_s):]]

        eval_scan = range(1, len(scan_s) + 1)
        eval_nm   = range(len(scan_s) + 1, len(scan_s) + 1 + len(nm_s))

        ax.scatter(eval_scan, scan_s, c=colors_scan, s=12, label="Coarse M scan", zorder=3)
        ax.scatter(eval_nm,   nm_s,   c=colors_nm,   s=12, label="Nelder-Mead refinement", zorder=3)
        ax.axvline(len(scan_s) + 0.5, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("Function evaluation")
        ax.set_ylabel("Sharpness (Laplacian energy)")
        ax.set_title(
            f"{label}\n"
            f"Converged to S={res['S_estimate']:.2f} C={res['C_estimate']:.2f} "
            f"ax={res['axis_estimate']:.0f}°  (+{res['sharpness_improvement_dB']:.1f} dB)",
            fontsize=9
        )
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUT, "step3_convergence.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — synthetic photorefraction images
# ─────────────────────────────────────────────────────────────────────────────

def plot_step4():
    # Rows: M values.  Columns: J0/J45 combinations
    Ms   = [-3.0, -1.5, 0.0, 1.5]
    J0s  = [ 0.0,  0.5, 0.0, -0.5]
    J45s = [ 0.0,  0.0, 0.5,  0.0]

    n_rows = len(Ms)
    n_cols = len(J0s)

    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(14, 8))
    fig.suptitle(
        "Step 4 — Synthetic eccentric photorefraction pupil images\n"
        "Left column per pair: horizontal source  |  Right: vertical source",
        fontsize=11, fontweight="bold"
    )

    rng = np.random.default_rng(0)
    for r, M in enumerate(Ms):
        for c, (J0, J45) in enumerate(zip(J0s, J45s)):
            # Convert power vector to prescription
            from eyesim.optimize import power_vector_to_prescription
            S, C, axis = power_vector_to_prescription(M, J0, J45)

            ch0 = render_single(S, C, axis, ex=10.0, ey=0.0,   noise_std=0.02, rng=rng)
            ch1 = render_single(S, C, axis, ex=0.0,  ey=10.0,  noise_std=0.02, rng=rng)

            ax_h = axes[r, c * 2]
            ax_v = axes[r, c * 2 + 1]

            ax_h.imshow(ch0, cmap="inferno", vmin=0, vmax=1)
            ax_v.imshow(ch1, cmap="inferno", vmin=0, vmax=1)

            if r == 0:
                ax_h.set_title(f"H  M={M:+.1f}\nJ0={J0:+.2f} J45={J45:+.2f}", fontsize=7.5)
                ax_v.set_title(f"V  M={M:+.1f}\nJ0={J0:+.2f} J45={J45:+.2f}", fontsize=7.5)
            else:
                if c == 0:
                    ax_h.set_ylabel(f"M={M:+.1f} D", fontsize=8)

            for ax in (ax_h, ax_v):
                ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    path = os.path.join(OUT, "step4_photorefraction.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating pipeline visualizations ...")
    print("  Step 1: eye blur grid ...")
    plot_step1()
    print("  Step 2: predistortion chain ...")
    plot_step2()
    print("  Step 3: optimizer convergence ...")
    plot_step3()
    print("  Step 4: photorefraction images ...")
    plot_step4()
    print(f"\nAll outputs saved to: {OUT}")
