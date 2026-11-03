"""
Trains the network (PupilNet) on the fake photos from generate_dataset.py.

The network gets the two LED frames plus the rig numbers it can't see in the
picture (distance, LED offset, pupil size, lens opening) and predicts the
eye's M, J0, J45. Photos in the blind band have no crescent, so a robust loss
keeps them from dominating. About 3 minutes per epoch on an M-series Mac.

Writes checkpoints/best.pt, train_log.csv, test_report.json, test_scatter.png.

  python scripts/train.py
  python scripts/train.py --data data/train_set --epochs 30 --out checkpoints
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eyesim.cnn import PupilNet, count_parameters

VISIBLE_PEAK = 0.15      # brightest crescent pixel (fraction of full scale) needed to count as visible


class Split:
    def __init__(self, root: Path, name: str):
        self.images = np.load(root / f"{name}_images.npy", mmap_mode="r")
        self.labels = torch.from_numpy(np.load(root / f"{name}_labels.npy"))
        self.aux = torch.from_numpy(np.load(root / f"{name}_aux.npy"))
        self.meta = np.load(root / f"{name}_meta.npy")
        self.n = len(self.labels)

    def batches(self, batch, shuffle, device):
        order = np.random.permutation(self.n) if shuffle else np.arange(self.n)
        for s in range(0, self.n, batch):
            idx = np.sort(order[s:s + batch])          # sorted reads are kinder to the memmap
            # HWC -> CHW on the CPU. Permuting on an MPS device leaves a
            # non-contiguous tensor that the convolutions silently mis-read.
            x = np.ascontiguousarray(self.images[idx].transpose(0, 3, 1, 2))
            x = torch.from_numpy(x).to(device).float() / 255.0
            yield x, self.aux[idx].to(device), self.labels[idx].to(device), idx


def augment(x):
    """Small framing and sensor jitter on top of what the simulator already varies."""
    b = x.shape[0]
    dx, dy = np.random.randint(-2, 3, 2)
    x = torch.roll(x, shifts=(int(dy), int(dx)), dims=(2, 3))
    x = x * torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.85, 1.15)
    x = x + 0.01 * torch.randn_like(x)
    return x.clamp(0.0, 1.0)


@torch.no_grad()
def predict(model, split, batch, device):
    model.eval()
    out = torch.empty(split.n, 3)
    for x, aux, _, idx in split.batches(batch, False, device):
        out[idx] = model(x, aux).cpu()
    return out


def mae_table(pred, split):
    err = (pred - split.labels).abs().numpy()
    peak, d = split.meta[:, 7], split.meta[:, 0]
    vis = peak >= VISIBLE_PEAK
    rows = {
        "all": np.ones(split.n, bool),
        "crescent visible": vis,
        "dark zone (no usable reflex)": ~vis,
        "crescent visible, d < 45 cm": vis & (d < 450),
        "crescent visible, d >= 45 cm": vis & (d >= 450),
    }
    table = {}
    for name, m in rows.items():
        if m.sum() == 0:
            continue
        e = err[m]
        table[name] = {"n": int(m.sum()), "share": float(m.mean()),
                       "M_mae": float(e[:, 0].mean()), "J0_mae": float(e[:, 1].mean()),
                       "J45_mae": float(e[:, 2].mean()),
                       "M_within_0.25": float((e[:, 0] <= 0.25).mean()),
                       "M_within_0.50": float((e[:, 0] <= 0.50).mean())}
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_set")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    device = (torch.device("cuda") if torch.cuda.is_available() else
              torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    print(f"Device: {device}")
    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train, val, test = Split(root, "train"), Split(root, "val"), Split(root, "test")
    size = train.images.shape[1]
    print(f"train {train.n:,}  val {val.n:,}  test {test.n:,}  image {size}x{size}")

    aux_dim = train.aux.shape[1]
    model = PupilNet(image_size=size, aux_dim=aux_dim, input_norm=True).to(device)
    print(f"PupilNet  {count_parameters(model):,} parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * ((train.n + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    loss_fn = nn.SmoothL1Loss(beta=0.5)

    log = out / "train_log.csv"
    log.write_text("epoch,train_loss,val_loss,M_mae,J0_mae,J45_mae,M_mae_visible,elapsed_s\n")
    best, t0 = float("inf"), time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for x, aux, y, _ in train.batches(args.batch, True, device):
            opt.zero_grad()
            loss = loss_fn(model(augment(x), aux), y)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
        pred = predict(model, val, args.batch * 2, device)
        vloss = loss_fn(pred, val.labels).item()
        t = mae_table(pred, val)
        a, v = t["all"], t["crescent visible"]
        el = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs}  train={tot / train.n:.4f}  val={vloss:.4f}  "
              f"M={a['M_mae']:.3f}  J0={a['J0_mae']:.3f}  J45={a['J45_mae']:.3f}  "
              f"M(visible)={v['M_mae']:.3f}  {el:.0f}s", flush=True)
        with open(log, "a") as fh:
            fh.write(f"{epoch},{tot / train.n:.6f},{vloss:.6f},{a['M_mae']:.6f},{a['J0_mae']:.6f},"
                     f"{a['J45_mae']:.6f},{v['M_mae']:.6f},{el:.1f}\n")
        if vloss < best:
            best = vloss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": vloss,
                        "mae": a, "model_kwargs": {"image_size": size, "aux_dim": aux_dim, "input_norm": True},
                        "dataset_config": json.loads((root / "config.json").read_text())},
                       out / "best.pt")

    ck = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    pred = predict(model, test, args.batch * 2, device)
    report = mae_table(pred, test)
    (out / "test_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nTest set (best epoch {ck['epoch']}):")
    for name, r in report.items():
        print(f"  {name:<30} n={r['n']:>6} ({100 * r['share']:.0f}%)  M={r['M_mae']:.3f}  J0={r['J0_mae']:.3f}  "
              f"J45={r['J45_mae']:.3f}  M within 0.25 D: {100 * r['M_within_0.25']:.0f}%  0.50 D: {100 * r['M_within_0.50']:.0f}%")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vis = test.meta[:, 7] >= VISIBLE_PEAK
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for k, (ax, name) in enumerate(zip(axes, ["M", "J0", "J45"])):
        t_, p_ = test.labels[:, k].numpy(), pred[:, k].numpy()
        ax.scatter(t_[~vis], p_[~vis], s=2, alpha=0.25, color="0.6", label="dark zone")
        ax.scatter(t_[vis], p_[vis], s=2, alpha=0.25, color="C0", label="crescent visible")
        lim = [t_.min(), t_.max()]
        ax.plot(lim, lim, "k-", lw=0.8)
        ax.set_xlabel(f"true {name} (D)"); ax.set_ylabel(f"predicted {name} (D)")
        ax.legend(markerscale=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "test_scatter.png", dpi=120)
    print(f"\nCheckpoint: {out / 'best.pt'}")


if __name__ == "__main__":
    main()
