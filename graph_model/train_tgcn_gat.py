"""train_tgcn_gat.py -- scratch-trained dense GAT + T-GCN (graph-conv GRU) classifier.

Pure PyTorch (no PyG needed): the graph has only 25 nodes, so dense ops are fast.

    python -m graph_model.train_tgcn_gat --seeds 0 1 2 3 4
    python -m graph_model.train_tgcn_gat --variant nograph --tag nograph   # graph ablation

Input per sample: X [T,25,F], E [T,2,25,25]; output: LOW/MEDIUM/HIGH of last window.
"""
import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, recall_score

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA = BASE_DIR / "graph_model" / "dataset_dyn.pt"
CKPT_DIR = BASE_DIR / "graph_model" / "checkpoints"
RESULTS_DIR = BASE_DIR / "graph_model" / "results"


# ----------------------------------------------------------------- data utils
def load_dataset(path, device):
    d = torch.load(path, map_location="cpu", weights_only=False)
    for s in ("train", "val", "test"):
        for k in ("X", "E", "y", "rule_label"):
            d[s][k] = d[s][k].to(device)
    d["class_weights"] = d["class_weights"].to(device)
    return d


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gcn_norm(E0):
    """E0 [B,N,N] weighted src->dst. Symmetrised, self-loops, D^-1/2 A D^-1/2."""
    n = E0.shape[-1]
    A = E0 + E0.transpose(1, 2) + torch.eye(n, device=E0.device)
    dinv = A.sum(-1).clamp(min=1e-6).pow(-0.5)
    return dinv.unsqueeze(-1) * A * dinv.unsqueeze(-2)


# --------------------------------------------------------------------- layers
class DenseGAT(nn.Module):
    """GAT layer on dense adjacency with edge-feature bias on attention logits."""

    def __init__(self, in_dim, head_dim, heads, edge_dim, dropout):
        super().__init__()
        self.h, self.d = heads, head_dim
        self.W = nn.Linear(in_dim, heads * head_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(1, 1, heads, head_dim))
        self.a_dst = nn.Parameter(torch.empty(1, 1, heads, head_dim))
        self.e_proj = nn.Linear(edge_dim, heads, bias=False)
        self.bias = nn.Parameter(torch.zeros(heads * head_dim))
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x, E):                       # x [B,N,F], E [B,C,N,N] (src,dst)
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.h, self.d)
        s = (h * self.a_src).sum(-1)               # [B,N,H]
        t = (h * self.a_dst).sum(-1)
        logit = t.unsqueeze(2) + s.unsqueeze(1)    # [B,i(dst),j(src),H]
        logit = logit + self.e_proj(E.permute(0, 3, 2, 1))   # edge feats as [dst,src,C]
        logit = F.leaky_relu(logit, 0.2)
        mask = (E[:, 0].transpose(1, 2) > 0) | torch.eye(N, dtype=torch.bool, device=x.device)
        logit = logit.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        alpha = self.drop(torch.softmax(logit, dim=2))
        out = torch.einsum("bijh,bjhd->bihd", alpha, h).reshape(B, N, self.h * self.d)
        return out + self.bias


class TGCNCell(nn.Module):
    """T-GCN cell: GRU whose gates use graph convolution (A_hat (X W))."""

    def __init__(self, in_dim, hid):
        super().__init__()
        self.gates = nn.Linear(in_dim + hid, 2 * hid)
        self.cand = nn.Linear(in_dim + hid, hid)

    def forward(self, x, h, Ahat):
        g = torch.sigmoid(Ahat @ self.gates(torch.cat([x, h], -1)))
        r, u = g.chunk(2, -1)
        c = torch.tanh(Ahat @ self.cand(torch.cat([x, r * h], -1)))
        return u * h + (1 - u) * c


def readout_head(hidden, n_classes, dropout):
    return nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                         nn.Dropout(dropout), nn.Linear(hidden, n_classes))


# --------------------------------------------------------------------- models
class TGCNGAT(nn.Module):
    def __init__(self, in_dim, hidden=64, heads=4, edge_dim=2, n_classes=3, dropout=0.3):
        super().__init__()
        self.p = dropout
        self.gat1 = DenseGAT(in_dim, hidden // heads, heads, edge_dim, dropout)
        self.gat2 = DenseGAT(hidden, hidden // heads, heads, edge_dim, dropout)
        self.skip = nn.Linear(in_dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.cell = TGCNCell(hidden, hidden)
        self.head = readout_head(hidden, n_classes, dropout)
        self.hidden = hidden

    def forward(self, X, E):                       # X [B,T,N,F], E [B,T,C,N,N]
        B, T, N, Fdim = X.shape
        x, e = X.reshape(B * T, N, Fdim), E.reshape(B * T, *E.shape[2:])
        z = F.dropout(F.elu(self.gat1(x, e)), self.p, self.training)
        z = F.elu(self.gat2(z, e)) + self.skip(x)
        z = self.norm(z).reshape(B, T, N, -1)
        Ahat = gcn_norm(e[:, 0]).reshape(B, T, N, N)
        h = torch.zeros(B, N, self.hidden, device=X.device)
        for t in range(T):
            h = self.cell(z[:, t], h, Ahat[:, t])
        return self.head(torch.cat([h.mean(1), h.max(1).values], -1))


class NoGraphTGCNGAT(TGCNGAT):
    """Graph ablation: same model with every edge removed, so each node only sees itself."""

    def forward(self, X, E):
        return super().forward(X, torch.zeros_like(E))


class StaticGCN(nn.Module):
    """Baseline: last window only, 2 GCN layers, no temporal part, no attention."""

    def __init__(self, in_dim, hidden=64, n_classes=3, dropout=0.3):
        super().__init__()
        self.l1, self.l2 = nn.Linear(in_dim, hidden), nn.Linear(hidden, hidden)
        self.p = dropout
        self.head = readout_head(hidden, n_classes, dropout)

    def forward(self, X, E):
        A = gcn_norm(E[:, -1, 0])
        h = F.dropout(F.relu(A @ self.l1(X[:, -1])), self.p, self.training)
        h = F.relu(A @ self.l2(h))
        return self.head(torch.cat([h.mean(1), h.max(1).values], -1))


# ----------------------------------------------------------------- train/eval
@torch.no_grad()
def predict_logits(model, split, bs=256):
    model.eval()
    return torch.cat([model(split["X"][i:i + bs], split["E"][i:i + bs])
                      for i in range(0, len(split["y"]), bs)])


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "acc": float((y_true == y_pred).mean()),
        "f1_per_class": [float(v) for v in f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)],
        "high_recall": float(recall_score(y_true, y_pred, labels=[2], average="macro", zero_division=0)),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def fit_and_eval(ctor, data, seed, device, epochs=150, patience=25, lr=2e-3,
                 wd=1e-4, bs=64, verbose=True):
    set_seed(seed)
    model = ctor().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=8)
    crit = nn.CrossEntropyLoss(weight=data["class_weights"], label_smoothing=0.05)
    tr, va, te = data["train"], data["val"], data["test"]
    n = len(tr["y"])
    best_key, best_state, best_ep, bad = (-1.0, 0.0), None, 0, 0

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = crit(model(tr["X"][b], tr["E"][b]), tr["y"][b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        vl = predict_logits(model, va)
        vf1 = metrics(va["y"].cpu(), vl.argmax(1).cpu())["macro_f1"]
        vloss = F.cross_entropy(vl, va["y"]).item()
        sched.step(vf1)
        if (vf1, -vloss) > best_key:
            best_key, best_ep, bad = (vf1, -vloss), ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if verbose and ep % 10 == 0:
            print(f"  seed {seed} ep {ep:3d} val_macroF1={vf1:.3f} val_loss={vloss:.3f}")
        if bad >= patience:
            break

    model.load_state_dict(best_state)
    res = {"seed": seed, "best_epoch": best_ep,
           "val": metrics(va["y"].cpu(), predict_logits(model, va).argmax(1).cpu()),
           "test": metrics(te["y"].cpu(), predict_logits(model, te).argmax(1).cpu())}
    return model, res


def summarize(results):
    f1 = np.array([r["test"]["macro_f1"] for r in results])
    vf1 = np.array([r["val"]["macro_f1"] for r in results])
    hr = np.array([r["test"]["high_recall"] for r in results])
    pc = np.array([r["test"]["f1_per_class"] for r in results]).mean(0)
    return {"macro_f1_mean": float(f1.mean()), "macro_f1_std": float(f1.std()),
            "val_macro_f1_mean": float(vf1.mean()),
            "high_recall_mean": float(hr.mean()), "f1_per_class_mean": pc.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--tag", default="scratch")
    ap.add_argument("--variant", choices=["full", "nograph"], default="full",
                    help="nograph = ablation with all edges removed")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_dataset(args.data, device)
    in_dim = data["train"]["X"].shape[-1]
    cfg = dict(in_dim=in_dim, hidden=args.hidden, heads=args.heads, edge_dim=2,
               n_classes=3, dropout=args.dropout)
    model_cls = NoGraphTGCNGAT if args.variant == "nograph" else TGCNGAT
    print(f"device={device} variant={args.variant} features={data['feature_names']} "
          f"seq_len={data['seq_len']}")

    results, best = [], (-1.0, None, None)
    for sd in args.seeds:
        model, r = fit_and_eval(lambda: model_cls(**cfg), data, sd, device, args.epochs,
                                args.patience, args.lr, args.wd, args.bs)
        results.append(r)
        print(f"seed {sd}: best_ep={r['best_epoch']} val_F1={r['val']['macro_f1']:.3f} "
              f"TEST macroF1={r['test']['macro_f1']:.3f} HIGH_recall={r['test']['high_recall']:.3f}")
        if r["val"]["macro_f1"] > best[0]:
            best = (r["val"]["macro_f1"], copy.deepcopy(model.state_dict()), r)

    summ = summarize(results)
    print(f"\nT-GCN+GAT ({args.tag}, {args.variant}) val macro-F1 = {summ['val_macro_f1_mean']:.3f} | "
          f"test macro-F1 = {summ['macro_f1_mean']:.3f} +/- {summ['macro_f1_std']:.3f}"
          f" | HIGH recall = {summ['high_recall_mean']:.3f} | per-class F1 (L/M/H) = "
          f"{[round(x, 3) for x in summ['f1_per_class_mean']]}")
    print("confusion (best-val seed, rows=true, cols=pred, L/M/H):")
    for row in best[2]["test"]["confusion"]:
        print("  ", row)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best[1], "config": cfg, "variant": args.variant,
                "feature_names": data["feature_names"],
                "norm": data["norm"], "label_names": data["label_names"],
                "seq_len": data["seq_len"], "node_order": data["node_order"],
                "val_macro_f1": best[0]}, CKPT_DIR / f"tgcn_gat_{args.tag}.pt")
    with open(RESULTS_DIR / f"tgcn_gat_{args.tag}.json", "w") as f:
        json.dump({"summary": summ, "runs": results, "args": vars(args)}, f, indent=2)
    print(f"checkpoint -> {CKPT_DIR / f'tgcn_gat_{args.tag}.pt'}")


if __name__ == "__main__":
    main()