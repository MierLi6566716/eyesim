"""
Tests the whole system on a model eye, whose refraction you set on its dial.

For every session taken with `eye_capture.py --eye model --true-rx X` it
runs the same analysis as measure_eye.py with one or more networks and
writes, into --out:

  results.csv        one row per session and network: dial, lens, distance,
                     pupil (set vs measured), crescent side H/V, dark zone,
                     network reading of eye + lens and of the eye
  report.png         (1) network vs dial with the best-fit line,
                     (2) crescent side vs dial against the simulator's
                         prediction for the same rig (tests the blind band),
                     (3) measured vs set pupil size (tests the ruler scale)
  summary.txt        slope / offset / error per network

How to read it:
  * slope near 1 and a constant offset: the network works; the offset is the
    model eye's own colour error at 850 nm plus the dial's zero.
  * slope well below 1 (readings squashed toward the blind band): a
    rig/simulator mismatch. Check panel 3 (scale) and panel 2 (band) first.
  * measured pupil off from the set pupil by >10 %: redo the ruler photo.

The model eye has no focusing and cannot drift, so pupil clicks can be reused
across sessions taken without moving it (--reuse-clicks).

  python scripts/model_eye_report.py model_eye/sessions/*/ --reuse-clicks
        every block folder -> model_eye/results/<block>/, plus model_eye/results/SUMMARY.txt
  python scripts/model_eye_report.py model_eye/sessions/A_37cm_plus4_7mm --reuse-clicks
        one block
  python scripts/model_eye_report.py model_eye/sessions/A_*/*-true-2* --out model_eye/results/test
        any sessions you list, into one report
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import measure_eye as me  # noqa: E402
from eyesim.photorefraction import LCA_850_D  # noqa: E402

DEFAULT_NETS = ["checkpoints/best.pt", "checkpoints/wide_m_only/best.pt"]


def load_net(path):
    import torch
    from eyesim.cnn import PupilNet
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = PupilNet(**ck["model_kwargs"]).eval()
    model.load_state_dict(ck["model_state_dict"])
    model.m_only = bool(ck.get("m_only", False))
    return model, ck.get("dataset_config", {})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", help="session folders, or block folders that contain sessions")
    ap.add_argument("--checkpoints", nargs="+", default=DEFAULT_NETS)
    ap.add_argument("--out", default="model_eye/results")
    ap.add_argument("--reuse-clicks", action="store_true",
                    help="sessions without clicks.json borrow the previous session's (model eye not moved)")
    ap.add_argument("--reclick", action="store_true")
    ap.add_argument("--hflip", action="store_true")
    ap.add_argument("--vflip", action="store_true")
    ap.add_argument("--brightness", choices=["glint", "none"], default="glint")
    args = ap.parse_args()

    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    paths = [Path(s) for s in sorted(args.sessions)]
    blocks = [p for p in paths if not (p / "meta.json").exists() and any(p.glob("*/meta.json"))]
    if not blocks:
        report(paths, out, args)
        return
    summary = []
    for b in blocks:
        print(f"\n================ block {b.name}")
        try:
            summary.append(f"== {b.name}\n" + report(sorted(b.glob("*/")), out / b.name, args))
        except SystemExit as e:
            summary.append(f"== {b.name}\n  skipped: {e}\n")
    out.mkdir(parents=True, exist_ok=True)
    (out / "SUMMARY.txt").write_text("\n".join(summary))
    print(f"\nwrote {out / 'SUMMARY.txt'}")


def report(paths, out, args):
    reclick = args.reclick
    sessions = [s for s in paths if (s / "meta.json").exists()]
    sessions = [s for s in sessions if json.loads((s / "meta.json").read_text()).get("true_rx") is not None]
    if not sessions:
        raise SystemExit("no model-eye sessions (meta.json needs true_rx: take them with --eye model --true-rx X)")
    out.mkdir(parents=True, exist_ok=True)

    if args.reuse_clicks:
        last = None
        for s in sessions:
            if (s / "clicks.json").exists() and not reclick:
                last = s / "clicks.json"
            elif last is not None:
                shutil.copy(last, s / "clicks.json")
                print(f"{s.name}: using the clicks from {last.parent.name}")

    margs = SimpleNamespace(hflip=args.hflip, vflip=args.vflip, brightness=args.brightness, reclick=reclick,
                            focal_mm=4.74, pixel_um=1.4)
    rows = []
    for ck_path in args.checkpoints:
        model, cfg = load_net(ROOT / ck_path if not Path(ck_path).is_absolute() else ck_path)
        print(f"\n######## {ck_path}")
        for s in sessions:
            try:
                r = me.measure(s, model, cfg, margs)
            except Exception as e:                  # e.g. pupil outside the crop: skip, keep going
                print(f"{s.name}: SKIPPED ({type(e).__name__}: {e})")
                continue
            m = r["meta"]
            rows.append({
                "network": ck_path, "session": s.name, "dial_d": m["true_rx"], "lens_d": m.get("lens_d", 0.0),
                "dist_cm": r["d"] / 10, "pupil_set_mm": m.get("pupil_mm"), "pupil_measured_mm": round(2 * r["pupil_r"], 2),
                "side_H": round(float(r["side"][0]), 3), "side_V": round(float(r["side"][1]), 3),
                "dark_zone": bool(r["dark"]),           # the network's own reading lands in the blind band
                "crescent_seen": bool(max(abs(r["side"][0]), abs(r["side"][1])) >= me.SIDE_MIN),
                # Everything in IR so no human colour correction is assumed for a plastic eye.
                "net_system_ir": round(float(r["sys_ir"][0]), 3), "net_eye_ir": round(float(r["eye_ir"][0]), 3),
            })
            margs.reclick = False

    if not rows:
        raise SystemExit("every session failed (see SKIPPED above)")
    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    lines = []
    for k, ck_path in enumerate(args.checkpoints):
        rs = [r for r in rows if r["network"] == ck_path]
        use = [r for r in rs if r["crescent_seen"]]
        x = np.array([r["dial_d"] for r in rs]); y = np.array([r["net_eye_ir"] for r in rs])
        dark = ~np.array([r["crescent_seen"] for r in rs])
        ax[0].scatter(x[~dark], y[~dark], color=f"C{k}", s=22, label=f"{ck_path} (crescent)")
        ax[0].scatter(x[dark], y[dark], color=f"C{k}", s=22, facecolors="none", label=f"{ck_path} (no crescent in photo)")
        if len(use) >= 3:
            xu = np.array([r["dial_d"] for r in use]); yu = np.array([r["net_eye_ir"] for r in use])
            slope, off = np.polyfit(xu, yu, 1)
            resid = yu - (slope * xu + off)
            err1 = np.abs(yu - xu - np.median(yu - xu))     # error after removing one constant offset
            lines.append(f"{ck_path}\n  sessions with a crescent: {len(use)} of {len(rs)}\n"
                         f"  fit: network = {slope:.2f} x dial {off:+.2f} D   (ideal slope 1.00)\n"
                         f"  scatter around the fit: {resid.std():.2f} D\n"
                         f"  error after removing a constant offset ({np.median(yu - xu):+.2f} D): "
                         f"mean {err1.mean():.2f} D, within 0.5 D: {100 * (err1 <= 0.5).mean():.0f} %\n")
            xx = np.linspace(xu.min(), xu.max(), 2)
            ax[0].plot(xx, slope * xx + off, color=f"C{k}", lw=1)
        else:
            lines.append(f"{ck_path}\n  fewer than 3 sessions with a crescent: no fit\n")
    lim = [min(r["dial_d"] for r in rows) - 1, max(r["dial_d"] for r in rows) + 1]
    ax[0].plot(lim, lim, "k--", lw=0.8, label="perfect")
    ax[0].set_xlabel("model eye dial (D)"); ax[0].set_ylabel("network, eye (IR, D)")
    ax[0].set_title("network vs dial"); ax[0].legend(fontsize=7)

    # Crescent side vs dial, real against simulated (one curve per distance / pupil).
    first = [r for r in rows if r["network"] == args.checkpoints[0]]
    groups = {}
    for r in first:
        groups.setdefault((r["dist_cm"], r["pupil_measured_mm"] // 1), []).append(r)
    for g, ((dist, _), rs) in enumerate(sorted(groups.items())):
        x = np.array([r["dial_d"] - r["lens_d"] for r in rs])
        ax[1].scatter(x, [r["side_H"] for r in rs], color=f"C{g}", marker="o", s=20, label=f"H, {dist:.0f} cm")
        ax[1].scatter(x, [r["side_V"] for r in rs], color=f"C{g}", marker="^", s=20, label=f"V, {dist:.0f} cm")
        s0 = next(s for s in sessions if s.name == rs[0]["session"])
        meta = json.loads((s0 / "meta.json").read_text())
        Ms, cls = me.side_curve(dist * 10, meta["ecc_mm"], meta.get("aperture_mm", 2.6),
                                np.median([r["pupil_measured_mm"] for r in rs]) / 2)
        ax[1].step(Ms, 0.3 * cls, where="mid", color=f"C{g}", lw=1, alpha=0.7)
    ax[1].axhline(me.SIDE_MIN, color="0.7", lw=0.5); ax[1].axhline(-me.SIDE_MIN, color="0.7", lw=0.5)
    ax[1].set_xlabel("dial - lens (D)   [simulated: eye + lens in IR]")
    ax[1].set_ylabel("crescent side (+ LED side, - far side)")
    ax[1].set_title("crescent side: dots real, steps simulated"); ax[1].legend(fontsize=7)

    ps = [(r["pupil_set_mm"], r["pupil_measured_mm"]) for r in first if r["pupil_set_mm"]]
    if ps:
        a = np.array(ps, float)
        ax[2].scatter(a[:, 0], a[:, 1], s=22)
        m = [a.min() - 0.5, a.max() + 0.5]
        ax[2].plot(m, m, "k--", lw=0.8)
        ratio = float(np.median(a[:, 1] / a[:, 0]))
        lines.append(f"pupil: measured / set = {ratio:.2f} (median). "
                     + ("OK." if abs(ratio - 1) < 0.1 else "More than 10 % off: redo the ruler photo (px_per_mm)."))
    ax[2].set_xlabel("pupil set on model eye (mm)"); ax[2].set_ylabel("pupil measured (mm)")
    ax[2].set_title("pixel scale check")
    fig.tight_layout()
    fig.savefig(out / "report.png", dpi=120)

    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n")
    print("\n" + text)
    print(f"wrote {out / 'results.csv'}, {out / 'report.png'}, {out / 'summary.txt'}")
    return text + "\n"


if __name__ == "__main__":
    main()
