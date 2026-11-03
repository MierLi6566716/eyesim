"""
Checks the simulator against a published paper (Chen, Tan & Lewis, Optics
Express 2003, Fig. 3) using that paper's setup: camera at 800 mm, flash 11 mm
off axis, pupils 7 to 3 mm, refraction +10 to -10 D.

Writes examples/output/paper_fig3_check.png and prints four pass/fail checks.

  python scripts/validate_photorefraction.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eyesim.photorefraction import Eye, Geometry, SA_NAVARRO, reflex

# The flash is 10 x 3 mm and sits 11 mm off axis; the aperture is cut off just
# inside it. One LED direction only (90 deg = flash above the lens).
PAPER = Geometry(d_mm=800.0, ecc_mm=11.0, aperture_r_mm=12.5, source_r_mm=1.5,
                 led_angles_deg=(90.0,), knife_edge_mm=9.5)
RX = [10, 7, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -7, -10]
PUPILS = [7, 6, 5, 4, 3]


def signal(rx, pupil_d, sa=SA_NAVARRO, n=48):
    eye = Eye(M=rx, pupil_r_mm=pupil_d / 2, sa_d_per_mm2=sa, scatter_mrad=0.3)
    I, mask, _ = reflex(eye, PAPER, image_size=n, fill=0.9, n_entry=500)
    return I[0], mask


def dark_zone_centre(pupil_d, sa):
    """Refraction at which the collected light is smallest (0.05 D steps)."""
    grid = np.arange(-4.0, 3.0, 0.05)
    tot = [signal(r, pupil_d, sa, n=24)[0].sum() for r in grid]
    return float(grid[int(np.argmin(tot))])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(PUPILS), len(RX), figsize=(15, 5.4))
    totals, peaks = {}, {}
    for i, pd in enumerate(PUPILS):
        row = [signal(r, pd) for r in RX]
        vmax = max(im.max() for im, _ in row)
        totals[pd] = [float(im.sum()) for im, _ in row]
        peaks[pd] = vmax * (pd / 2) ** 2      # pixel irradiance ~ fraction collected x entrance area
        for j, (im, mask) in enumerate(row):
            ax = axes[i, j]
            ax.imshow(np.where(mask, im, np.nan), cmap="jet", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if i == len(PUPILS) - 1:
                ax.set_xlabel(f"{RX[j]:+d}")
            if j == 0:
                ax.set_ylabel(f"{pd} mm")
    fig.suptitle("Simulated photorefraction, paper geometry (compare Chen et al. 2003, Fig. 3)")
    out = Path(__file__).resolve().parent.parent / "examples" / "output" / "paper_fig3_check.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}\n")

    print("Check 1: 3 mm pupil, no aberration: dark-zone centre should be -1/d = -1.25 D")
    print(f"         simulated: {dark_zone_centre(3, 0.0):+.2f} D")
    print("Check 2: 7 mm pupil, Navarro spherical aberration: centre shifts hyperopic (paper: about +0.25 D)")
    print(f"         simulated: {dark_zone_centre(7, SA_NAVARRO):+.2f} D")
    print("Check 3: peak irradiance per pupil size relative to 7 mm (paper: 100, 70, 50, 30, 16%)")
    print("         simulated: " + ", ".join(f"{100 * peaks[pd] / peaks[7]:.0f}%" for pd in PUPILS))
    t7 = dict(zip(RX, totals[7]))
    print("Check 4: 7 mm pupil, reflex grows out to about +/-4 D, then saturates or falls")
    print("         total light by refraction: " + "  ".join(f"{r:+d}:{t7[r] / max(t7.values()):.2f}" for r in RX))


if __name__ == "__main__":
    main()
