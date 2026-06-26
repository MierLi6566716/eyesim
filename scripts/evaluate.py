"""
Evaluate a trained PupilNet checkpoint on the held-out test set.

Reports per-component MAE and RMS error in power-vector space (M, J0, J45)
and in clinical space (S, C, axis). Saves a prediction scatter plot.

The 0.25 D clinical threshold: if M_MAE < 0.25 D and J0/J45 MAE < 0.12 D,
the estimator is within one optometry step (0.25 D) for most patients.
That is the minimum bar for a production deployment.

Usage
-----
  python scripts/evaluate.py --checkpoint checkpoints/best.pt --data data
"""

import argparse
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eyesim.cnn import PupilNet, PrescriptionDataset
from eyesim.optimize import power_vector_to_prescription


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    preds, targets = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        preds.append(model(imgs).cpu().numpy())
        targets.append(lbls.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def clinical_errors(preds_pv, targets_pv):
    """Convert (N,3) power-vector arrays to clinical errors."""
    S_pred, C_pred, ax_pred, S_true, C_true, ax_true = [], [], [], [], [], []
    for p, t in zip(preds_pv, targets_pv):
        sp, cp, axp = power_vector_to_prescription(*p)
        st, ct, axt = power_vector_to_prescription(*t)
        S_pred.append(sp); C_pred.append(cp); ax_pred.append(axp)
        S_true.append(st); C_true.append(ct); ax_true.append(axt)

    S_err  = np.abs(np.array(S_pred)  - np.array(S_true))
    C_err  = np.abs(np.array(C_pred)  - np.array(C_true))
    # Axis error wraps at 180° — use the smaller of |Δaxis| and 180 - |Δaxis|
    ax_err = np.abs(np.array(ax_pred) - np.array(ax_true))
    ax_err = np.minimum(ax_err, 180.0 - ax_err)
    return S_err, C_err, ax_err


def report(preds, targets):
    diff   = preds - targets
    abs_d  = np.abs(diff)
    rms    = np.sqrt((diff ** 2).mean(axis=0))
    mae    = abs_d.mean(axis=0)
    p95    = np.percentile(abs_d, 95, axis=0)

    print("\n=== Power-vector errors (dioptres) ===")
    print(f"{'':6}  {'MAE':>7}  {'RMS':>7}  {'P95':>7}")
    for i, name in enumerate(["M", "J0", "J45"]):
        print(f"  {name:4}  {mae[i]:7.4f}  {rms[i]:7.4f}  {p95[i]:7.4f}")
    print(f"  {'ALL':4}  {mae.mean():7.4f}  {rms.mean():7.4f}  {p95.mean():7.4f}")

    S_err, C_err, ax_err = clinical_errors(preds, targets)
    print("\n=== Clinical errors ===")
    print(f"  S    MAE={S_err.mean():.4f} D   P95={np.percentile(S_err, 95):.4f} D")
    print(f"  C    MAE={C_err.mean():.4f} D   P95={np.percentile(C_err, 95):.4f} D")
    print(f"  axis MAE={ax_err.mean():.2f}°   P95={np.percentile(ax_err, 95):.2f}°")

    threshold_M   = 0.25
    threshold_J   = 0.12
    pass_M   = (abs_d[:, 0] < threshold_M).mean() * 100
    pass_J0  = (abs_d[:, 1] < threshold_J).mean()  * 100
    pass_J45 = (abs_d[:, 2] < threshold_J).mean()  * 100
    print(f"\n=== Clinical threshold (M<{threshold_M}D, J<{threshold_J}D) ===")
    print(f"  Within threshold: M={pass_M:.1f}%  J0={pass_J0:.1f}%  J45={pass_J45:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data",       default="data")
    ap.add_argument("--batch",      type=int, default=512)
    ap.add_argument("--workers",    type=int, default=4)
    ap.add_argument("--plot",       action="store_true", help="save scatter plots")
    args = ap.parse_args()

    device = (
        torch.device("cuda") if torch.cuda.is_available()  else
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = PupilNet().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.6f})")

    test_ds     = PrescriptionDataset(Path(args.data) / "test.npz", augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch, num_workers=args.workers)

    preds, targets = run_inference(model, test_loader, device)
    report(preds, targets)

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            names = ["M (D)", "J0 (D)", "J45 (D)"]
            lims  = [(-6.5, 4.5), (-2.2, 2.2), (-2.2, 2.2)]
            for ax, n, lim, i in zip(axes, names, lims, range(3)):
                ax.scatter(targets[:, i], preds[:, i], s=1, alpha=0.2, color="steelblue")
                ax.plot(lim, lim, "r--", lw=1)
                ax.set_xlim(lim); ax.set_ylim(lim)
                ax.set_xlabel(f"True {n}"); ax.set_ylabel(f"Predicted {n}")
                mae = np.abs(preds[:, i] - targets[:, i]).mean()
                ax.set_title(f"{n}   MAE={mae:.3f}")
            fig.tight_layout()
            plot_path = Path(args.checkpoint).parent / "test_scatter.png"
            fig.savefig(plot_path, dpi=150)
            print(f"Saved scatter plot: {plot_path}")
        except ImportError:
            print("matplotlib not installed, skipping plot.")


if __name__ == "__main__":
    main()
