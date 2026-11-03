"""Figures for hardware/alvarez_doc/alvarez_glasses_hardware.pdf (run from repo root)."""
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from eyesim.photorefraction import Eye, Geometry, reflex

OUT = Path("hardware/alvarez_doc/figures")
C1, C2, C3, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "lines.linewidth": 2, "legend.frameon": False})

BIG = 1.0e5   # mm; a "camera at infinity" stand-in for the Badal (A = 0) case


def equiv(theta_e_mrad, theta_a_mrad, theta_s_mrad, angles=(0.0, 90.0)):
    """Equivalent free-space geometry for a relay whose camera stop sits at its focal plane."""
    k = BIG * 1e-3
    return Geometry(d_mm=BIG, ecc_mm=theta_e_mrad * k, aperture_r_mm=theta_a_mrad * k,
                    source_r_mm=theta_s_mrad * k, led_angles_deg=angles)


CONFIGS = {
    "A  bare camera on the frame (d = 25 mm, LED 2 mm off axis)":
        [Geometry(d_mm=25.0, ecc_mm=2.0, aperture_r_mm=0.3, source_r_mm=0.2)],
    "B  hot mirror + 50 mm relay, one LED 0.25 mm off the stop":
        [equiv(5.0, 1.5, 1.0)],
    "C  same relay, graded LED line at 0.12 / 0.25 / 0.5 mm":
        [equiv(2.5, 1.5, 1.0), equiv(5.0, 1.5, 1.0), equiv(10.0, 1.5, 1.0)],
}
MS = [-6, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4]


def frame(cfg, M, r=3.0, n=48):
    eye = Eye(M=M, pupil_r_mm=r, scatter_mrad=2.0)
    return sum(reflex(eye, g, n, 0.85, n_entry=200)[0][0] for g in cfg)


def asym(img):
    h = img.shape[1] // 2
    L, R = img[:, :h].sum(), img[:, h:].sum()
    return (R - L) / (R + L + 1e-9)


# ---------------------------------------------------------------- figure 1: frames
fig, axes = plt.subplots(len(CONFIGS), len(MS), figsize=(11, 3.9))
for row, (name, cfg) in enumerate(CONFIGS.items()):
    imgs = [frame(cfg, M) for M in MS]
    vmax = max(i.max() for i in imgs) + 1e-12
    for col, (M, img) in enumerate(zip(MS, imgs)):
        ax = axes[row, col]
        ax.imshow(img / vmax, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if row == 0:
            ax.set_title(f"M = {M:+g} D", fontsize=8, color=INK2)
    axes[row, 0].set_ylabel(name.split()[0], rotation=0, labelpad=12, fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "frames.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- figure 2: signal vs M
Mfine = np.linspace(-6, 4, 81)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.0))
for (name, cfg), c in zip(CONFIGS.items(), [C2, C1, C3]):
    imgs = [frame(cfg, M, n=32) for M in Mfine]
    tot = np.array([i.sum() for i in imgs])
    a1.plot(Mfine, [asym(i) for i in imgs], color=c, label=name.split("(")[0].strip())
    a2.plot(Mfine, tot / tot.max(), color=c, label=name.split()[0])
a1.axvline(0, color=INK2, lw=0.8, ls="-.")
a1.set(xlabel="eye refraction M (D)", ylabel="crescent asymmetry (R-L)/(R+L)")
a2.set(xlabel="eye refraction M (D)", ylabel="reflex brightness (normalised)")
a1.set_title("Signal the CNN reads: which side the light is on", loc="left", fontweight="bold", fontsize=10)
a2.set_title("Total returned light", loc="left", fontweight="bold", fontsize=10)
a1.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
fig.tight_layout()
fig.savefig(OUT / "signal.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- figure 3: closed loop
rng = np.random.default_rng(3)
N, K = 300, 8


def lens_actual(cmd, gain, offset, state):
    """Imprecise actuator: unknown gain/offset, 0.3 D backlash, 0.08 D jitter."""
    dirn = np.sign(cmd - state["last"]) if cmd != state["last"] else state["dir"]
    slack = 0.15 * dirn
    state["last"], state["dir"] = cmd, dirn
    return gain * (cmd - slack) + offset + rng.normal(0, 0.08)


def sensor(resid, frames=1):
    """CNN residual estimate: 0.10 D noise per frame + 10 % scale error, saturates at +/-5 D."""
    noise = rng.normal(0, 0.10 + 0.05 * abs(resid)) / np.sqrt(frames)
    return np.clip(resid * rng.uniform(0.9, 1.1) + noise, -5, 5)


closed = np.zeros((N, K + 1))
openloop = np.zeros(N)
for i in range(N):
    M = rng.uniform(-6, 4)
    gain, off = rng.uniform(0.8, 1.2), rng.normal(0, 0.25)
    st = {"last": 0.0, "dir": 1.0}
    cmd = 0.0
    resid = M + lens_actual(cmd, gain, off, st)
    closed[i, 0] = resid
    # open loop: one measurement, one command, trust the dial
    st_o = {"last": 0.0, "dir": 1.0}
    openloop[i] = M + lens_actual(-sensor(M), gain, off, st_o)
    for k in range(1, K + 1):
        est = sensor(resid, frames=4)          # average 4 frames per step
        if abs(est) > 0.12:                    # dead band: don't chase noise
            cmd = cmd - 0.8 * est
        resid = M + lens_actual(cmd, gain, off, st)
        closed[i, k] = resid

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.5, 1]})
for i in range(60):
    a1.plot(range(K + 1), np.abs(closed[i]), color=C1, alpha=0.18, lw=1)
a1.plot(range(K + 1), np.median(np.abs(closed), 0), color=INK, lw=2, label="median of 300 eyes")
a1.axhline(0.25, color=C2, lw=1, ls="-.")
a1.text(K, 0.27, "0.25 D", color=C2, ha="right", va="bottom", fontsize=8)
a1.set_yscale("log"); a1.set_ylim(0.01, 10)
a1.set(xlabel="measure-and-nudge iteration", ylabel="|residual refraction| (D)")
a1.set_title("Closed loop with a sloppy actuator", loc="left", fontweight="bold", fontsize=10)
a1.legend(fontsize=8)
bins = np.linspace(0, 2, 41)
a2.hist(np.abs(openloop), bins, color=C2, alpha=0.8, label="open loop (trust the dial)")
a2.hist(np.abs(closed[:, 5]), bins, color=C1, alpha=0.8, label="closed loop, 5 iterations")
a2.set(xlabel="|final residual| (D)", ylabel="eyes")
a2.set_title("Final error", loc="left", fontweight="bold", fontsize=10)
a2.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "closed_loop.png", dpi=170)
plt.close(fig)
print("open loop within 0.25 D:", np.mean(np.abs(openloop) <= 0.25),
      " closed (5 it):", np.mean(np.abs(closed[:, 5]) <= 0.25),
      " median open:", np.median(np.abs(openloop)), " median closed:", np.median(np.abs(closed[:, 5])))
