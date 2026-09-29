"""pretrain_dgraph.py -- pretrain the T-GCN + GAT core on DGraph-Fin (fraud vs normal).

Run from the project root:
  python -m graph_model.pretrain_dgraph
  python -m graph_model.pretrain_dgraph --epochs 30 --steps-per-epoch 50

Saves graph_model/checkpoints/tgcn_gat_core.pt (core weights only; the 17-dim
input adapter and the fraud head are NOT saved -- fine-tuning uses new ones).
Validation is on a FIXED set of sampled subgraphs so epochs are comparable.
Seeds are sampled 50/50 fraud/normal, so AUC is the meaningful metric here.
"""
import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from graph_model.dgraph_loader import DGraphData, collate
from graph_model.models import RiskModel, to_device

BASE_DIR = Path(__file__).resolve().parent.parent


def evaluate(model, graphs, device, bs=4):
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for i in range(0, len(graphs), bs):
            b, seed, y = collate(graphs[i:i + bs])
            logits = model(to_device(b, device))[seed.to(device)]
            ps.append(torch.softmax(logits, -1)[:, 1].cpu()); ys.append(y)
    p, y = torch.cat(ps).numpy(), torch.cat(ys).numpy()
    return roc_auc_score(y, p), average_precision_score(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, default=BASE_DIR / "data" / "DGraphFin" / "dgraphfin.npz")
    ap.add_argument("--out", type=Path, default=BASE_DIR / "graph_model" / "checkpoints" / "tgcn_gat_core.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=50)
    ap.add_argument("--batch-graphs", type=int, default=4)
    ap.add_argument("--seeds-per-graph", type=int, default=32)
    ap.add_argument("--val-graphs", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    print(f"device={device}; loading {args.npz} ...")
    data = DGraphData(args.npz)

    val_rng, test_rng = np.random.default_rng(1234), np.random.default_rng(4321)
    val_graphs = [data.sample("valid", val_rng, args.seeds_per_graph) for _ in range(args.val_graphs)]
    test_graphs = [data.sample("test", test_rng, args.seeds_per_graph) for _ in range(args.val_graphs)]
    print(f"fixed val/test subgraphs built (avg {np.mean([g['x'].size(0) for g in val_graphs]):.0f} nodes each)")

    model = RiskModel(17, 2, "node", args.hidden, args.heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    best_auc, best_core, bad = -1.0, None, 0

    for ep in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for _ in range(args.steps_per_epoch):
            items = [data.sample("train", rng, args.seeds_per_graph) for _ in range(args.batch_graphs)]
            b, seed, y = collate(items)
            logits = model(to_device(b, device))[seed.to(device)]
            loss = F.cross_entropy(logits, y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        auc, ap_ = evaluate(model, val_graphs, device)
        print(f"epoch {ep:3d} | loss {np.mean(losses):.4f} | val AUC {auc:.4f} AP {ap_:.4f} | {time.time() - t0:.0f}s")
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_core = copy.deepcopy(model.core.state_dict())
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"core": best_core, "hidden": args.hidden, "heads": args.heads,
                        "val_auc": float(best_auc), "pretrained_on": "DGraph-Fin"}, args.out)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping"); break

    model.core.load_state_dict(best_core)
    auc, ap_ = evaluate(model, test_graphs, device)
    print(f"best val AUC {best_auc:.4f} | test AUC {auc:.4f} AP {ap_:.4f}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()