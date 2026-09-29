"""finetune_cloud.py -- train the T-GCN + GAT on graph_model/processed/dataset.pt.

Run from the project root:
  python -m graph_model.finetune_cloud --mode scratch --seeds 5
  python -m graph_model.finetune_cloud --mode head    --seeds 5   (needs tgcn_gat_core.pt)
  python -m graph_model.finetune_cloud --mode full    --seeds 5   (needs tgcn_gat_core.pt)

Modes
  scratch : random init, train everything                         (your reference result)
  head    : load pretrained core, FREEZE it, train new adapter + head only
  full    : head phase first, then unfreeze the core at a low learning rate

Task: classify the newest window of each 5-window sequence (y_last) as LOW/MEDIUM/HIGH.
Early stopping on validation macro-F1. Test metrics are averaged over seeds, because
the test set is small (MEDIUM has only a handful of sequences).
Edge input is log1p(request count); the is_attack edge flag is deliberately NOT used
(it is derived from the same volume thresholds as the label).
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from graph_model.models import RiskModel, to_device

BASE_DIR = Path(__file__).resolve().parent.parent
CLASSES = ["LOW", "MEDIUM", "HIGH"]


def make_items(split):
    snaps = split["snapshots"]
    return [([snaps[i] for i in q["idx"]], q["y_last"]) for q in split["sequences"]]


def collate(items):
    T = len(items[0][0])
    xs, eis, ews = [], [], []
    for t in range(T):
        xp, ep, wp, off = [], [], [], 0
        for snaps, _ in items:
            s = snaps[t]
            xp.append(s["x"]); ep.append(s["edge_index"] + off); wp.append(s["edge_attr"][:, 0])
            off += s["x"].size(0)
        xs.append(torch.cat(xp)); eis.append(torch.cat(ep, 1)); ews.append(torch.cat(wp))
    sizes = torch.tensor([snaps[0]["x"].size(0) for snaps, _ in items])
    batch = torch.repeat_interleave(torch.arange(len(items)), sizes)
    y = torch.tensor([y for _, y in items])
    return {"x": xs, "edge_index": eis, "edge_weight": ews, "batch": batch,
            "num_graphs": len(items)}, y


@torch.no_grad()
def evaluate(model, items, class_w, device, bs=32):
    model.eval()
    preds, ys, loss = [], [], 0.0
    for i in range(0, len(items), bs):
        b, y = collate(items[i:i + bs])
        logits = model(to_device(b, device))
        loss += F.cross_entropy(logits, y.to(device), weight=class_w, reduction="sum").item()
        preds.append(logits.argmax(-1).cpu()); ys.append(y)
    p, y = torch.cat(preds).numpy(), torch.cat(ys).numpy()
    return f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0), loss / len(items), y, p


def fit(model, params, lr, max_epochs, patience, train_items, val_items, class_w,
        bs, device, seed, core_frozen):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    best_key, best_state, bad = (-1.0, -float("inf")), None, 0
    for _ in range(max_epochs):
        model.train()
        if core_frozen:
            model.core.eval()                 # frozen core: no dropout either
        perm = torch.randperm(len(train_items), generator=g).tolist()
        for i in range(0, len(perm), bs):
            b, y = collate([train_items[j] for j in perm[i:i + bs]])
            loss = F.cross_entropy(model(to_device(b, device)), y.to(device), weight=class_w)
            opt.zero_grad(); loss.backward(); opt.step()
        f1, vloss, _, _ = evaluate(model, val_items, class_w, device)
        key = (f1, -vloss)
        if key > best_key:
            best_key, best_state, bad = key, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return best_key[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["scratch", "head", "full"], required=True)
    ap.add_argument("--data", type=Path, default=BASE_DIR / "graph_model" / "processed" / "dataset.pt")
    ap.add_argument("--ckpt", type=Path, default=BASE_DIR / "graph_model" / "checkpoints" / "tgcn_gat_core.pt")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--head-epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ft-lr", type=float, default=1e-4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(args.data, weights_only=True)
    in_dim = len(data["meta"]["feature_names"])
    train_items, val_items, test_items = (make_items(data["splits"][k]) for k in ("train", "val", "test"))
    print(f"mode={args.mode} device={device} in_dim={in_dim} "
          f"sequences train/val/test = {len(train_items)}/{len(val_items)}/{len(test_items)}")

    counts = np.bincount([y for _, y in train_items], minlength=3).clip(min=1)
    class_w = torch.tensor(counts.sum() / (3.0 * counts), dtype=torch.float32, device=device)
    print("class weights (LOW, MEDIUM, HIGH):", [round(float(w), 2) for w in class_w])

    hidden, heads, core_sd = args.hidden, args.heads, None
    if args.mode != "scratch":
        if not args.ckpt.exists():
            raise SystemExit(f"Pretrained core not found at {args.ckpt}. Run pretrain_dgraph first.")
        ck = torch.load(args.ckpt, weights_only=True)
        hidden, heads, core_sd = ck["hidden"], ck["heads"], ck["core"]

    results = []
    for seed in range(args.seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        model = RiskModel(in_dim, 3, "graph", hidden, heads).to(device)
        if core_sd is not None:
            model.load_core(core_sd)

        if args.mode == "scratch":
            val_f1 = fit(model, model.parameters(), args.lr, args.epochs, args.patience,
                         train_items, val_items, class_w, args.batch_size, device, seed, False)
        else:
            model.set_core_trainable(False)
            head_params = list(model.adapter.parameters()) + list(model.head.parameters())
            val_f1 = fit(model, head_params, args.lr, args.head_epochs, args.patience,
                         train_items, val_items, class_w, args.batch_size, device, seed, True)
            if args.mode == "full":
                model.set_core_trainable(True)
                val_f1 = fit(model, model.parameters(), args.ft_lr, args.epochs, args.patience,
                             train_items, val_items, class_w, args.batch_size, device, seed, False)

        f1, _, y, p = evaluate(model, test_items, class_w, device)
        rec = recall_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)
        cm = confusion_matrix(y, p, labels=[0, 1, 2])
        results.append({"seed": seed, "val_f1": float(val_f1), "test_f1": float(f1),
                        "recall": rec.tolist(), "confusion": cm.tolist()})
        print(f"seed {seed}: val macro-F1 {val_f1:.3f} | test macro-F1 {f1:.3f} | "
              f"recall LOW/MED/HIGH {np.round(rec, 2).tolist()}")

    f1s = np.array([r["test_f1"] for r in results])
    rec = np.mean([r["recall"] for r in results], axis=0)
    cm = np.sum([r["confusion"] for r in results], axis=0)
    print(f"\n[{args.mode}] test macro-F1 over {args.seeds} seeds: {f1s.mean():.3f} +/- {f1s.std():.3f}")
    print("mean per-class recall:", {c: round(float(r), 3) for c, r in zip(CLASSES, rec)})
    print("confusion matrix summed over seeds (rows=true, cols=pred; LOW/MEDIUM/HIGH):")
    print(cm)

    out = BASE_DIR / "graph_model" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"finetune_{args.mode}.json", "w") as f:
        json.dump({"mode": args.mode, "test_f1_mean": float(f1s.mean()), "test_f1_std": float(f1s.std()),
                   "per_seed": results}, f, indent=2)


if __name__ == "__main__":
    main()