"""Train the reflex SNN on window shards.

    pixi run train                                  # data/synth -> out/models/snn.pt
    pixi run train --data data/synth --epochs 20

Validation is split by episode, never by window, so windows of one ball never sit on
both sides. Reported: accuracy per class, false-alarm rate per kind of negative, and
firing rate per layer (sparsity).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from reflex.models.snn import ReflexSNN, SNNConfig, count_parameters


def load_shards(data_dir: str | Path) -> tuple[dict, dict]:
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "meta.json").read_text())
    parts = [np.load(p) for p in sorted(data_dir.glob("shard_*.npz"))]
    if not parts:
        raise FileNotFoundError(f"no shards in {data_dir}; run `pixi run gen-data` first")
    keys = ("x", "y", "episode", "kind") + (("session",) if all("session" in p for p in parts) else ())
    data = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    return data, meta


def split_by_episode(episode: np.ndarray, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.unique(episode)
    np.random.default_rng(seed).shuffle(ids)
    val_ids = ids[: max(1, int(len(ids) * val_frac))]
    val = np.isin(episode, val_ids)
    return ~val, val


@torch.no_grad()
def evaluate(model, x, y, kind, labels, kinds, device, batch=1024) -> dict:
    model.eval()
    preds, rate_sum, n_batches = [], None, 0
    for i in range(0, len(x), batch):
        logits, rates = model(x[i:i + batch].to(device), return_stats=True)
        preds.append(logits.argmax(1).cpu())
        rate_sum = np.array(rates) if rate_sum is None else rate_sum + np.array(rates)
        n_batches += 1
    pred = torch.cat(preds).numpy()
    y, kind = y.numpy(), kind.numpy()
    out = {"accuracy": float((pred == y).mean())}
    out["per_class"] = {lab: float((pred[y == i] == i).mean()) for i, lab in enumerate(labels) if (y == i).any()}
    # False alarm: a "none" window classified as a threat, broken down by where it came from.
    neg = y == 0
    out["false_alarm"] = {k: float((pred[neg & (kind == i)] != 0).mean())
                          for i, k in enumerate(kinds) if (neg & (kind == i)).any()}
    out["false_alarm_all"] = float((pred[neg] != 0).mean())
    out["missed_threat"] = float((pred[~neg] == 0).mean())
    out["wrong_side"] = float(((pred[~neg] != 0) & (pred[~neg] != y[~neg])).mean())
    out["firing_rate"] = [float(r) for r in rate_sum / n_batches]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--out", default="out/models/snn.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-session", default=None,
                    help="recorded data: hold out this whole session for validation instead of random clips")
    ap.add_argument("--init", default=None, help="checkpoint to start from (fine-tuning)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    data, meta = load_shards(args.data)
    labels, kinds = meta["labels"], meta["kinds"]
    tr, va = split_by_episode(data["episode"], args.val_frac, args.seed)
    held_out = "held-out SIMULATED episodes; says nothing about the real camera"
    if args.val_session:
        va = data["session"] == meta["sessions"].index(args.val_session)
        tr = ~va
        held_out = f"recorded session '{args.val_session}', never trained on"
    elif "sessions" in meta:
        held_out = "held-out recorded clips from the same sessions as training (optimistic)"
    x = torch.from_numpy(data["x"])
    y = torch.from_numpy(data["y"])
    kind = torch.from_numpy(data["kind"])
    print(f"{len(y)} windows, shape {tuple(x.shape[1:])}: train {int(tr.sum())}, val {int(va.sum())} "
          f"(split by episode); classes {dict(zip(labels, np.bincount(data['y'], minlength=len(labels)).tolist()))}")

    cfg = SNNConfig(n_classes=len(labels))
    model = ReflexSNN(cfg)
    if args.init:
        ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
        cfg = SNNConfig.from_dict(ckpt["model_cfg"])
        model = ReflexSNN(cfg)
        model.load_state_dict(ckpt["state_dict"])
    model.to(args.device)
    print(f"model: {count_parameters(model)} parameters, channels {cfg.channels}, {x.shape[1]} time steps")

    counts = np.bincount(data["y"][tr], minlength=len(labels)).astype(np.float32)
    weight = torch.tensor(counts.sum() / (len(labels) * np.maximum(counts, 1)), device=args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * (int(tr.sum()) // args.batch + 1))
    xtr, ytr = x[tr], y[tr]
    xva, yva, kva = x[va], y[va], kind[va]
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(ytr))
        total = 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            xb, yb = xtr[idx].to(args.device), ytr[idx].to(args.device)
            if torch.rand(()) < 0.5:   # mirror left/right: flip the image and swap the labels
                xb = xb.flip(-1)
                yb = torch.where(yb == 1, 3, torch.where(yb == 3, 1, yb))
            loss = F.cross_entropy(model(xb), yb, weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        res = evaluate(model, xva, yva, kva, labels, kinds, args.device)
        print(f"epoch {epoch:2d}  loss {total / len(perm):.3f}  val acc {res['accuracy']:.3f}  "
              f"false alarm {res['false_alarm_all']:.3f}  missed {res['missed_threat']:.3f}  "
              f"wrong side {res['wrong_side']:.3f}")
    res = evaluate(model, xva, yva, kva, labels, kinds, args.device)
    res.update(train_seconds=time.time() - t0, parameters=count_parameters(model),
               validation=held_out)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_cfg": cfg.to_dict(), "labels": labels,
                "window": meta["window"], "metrics": res, "data": str(args.data)}, out)
    out.with_suffix(".json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
