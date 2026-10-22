"""train_node_risk.py -- node-level risk scores, localization metrics, attack-path ranking.

    python -m graph_model.build_node_dataset --horizon 0 --seq-len 3 --fix-run-boundary --out graph_model\\ds_node_h0.pt
    python -m graph_model.train_node_risk --data graph_model\\ds_node_h0.pt --tag node_h0

Model: the same dense GAT + T-GCN encoder as train_tgcn_gat.py with TWO heads
  * window head : LOW / MEDIUM / HIGH (weighted cross-entropy)
  * node head   : one score per node; softmax over the 25 nodes, trained with
                  cross-entropy on the attacker's compromised node
                  (only windows where a compromised node exists)
Localization metrics (test windows with a known compromised node):
  top-1 / top-3 accuracy and MRR of the true compromised node in the node ranking.
Compared against: random, a traffic heuristic, a per-node Random Forest (no graph),
and the same network with edges removed (no-graph ablation).

Dashboard output: node_risk (0..1) = softmax-node-score / max * (1 - P(LOW)), so quiet
windows give low risk everywhere. Attack paths are a transparent HEURISTIC (mean node
risk along directed paths starting at the 3 riskiest nodes) and are NOT validated.
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier

try:
    from graph_model.train_tgcn_gat import (CKPT_DIR, RESULTS_DIR, DenseGAT, TGCNCell, gcn_norm,
                                            load_dataset, metrics, readout_head, set_seed)
except ImportError:
    from train_tgcn_gat import (CKPT_DIR, RESULTS_DIR, DenseGAT, TGCNCell, gcn_norm,
                                load_dataset, metrics, readout_head, set_seed)


# ---------------------------------------------------------------------- model
class NodeRiskNet(nn.Module):
    def __init__(self, in_dim, hidden=32, heads=4, edge_dim=2, n_classes=3, dropout=0.5, use_graph=True):
        super().__init__()
        self.p, self.hidden, self.use_graph = dropout, hidden, use_graph
        self.gat1 = DenseGAT(in_dim, hidden // heads, heads, edge_dim, dropout)
        self.gat2 = DenseGAT(hidden, hidden // heads, heads, edge_dim, dropout)
        self.skip = nn.Linear(in_dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.cell = TGCNCell(hidden, hidden)
        self.win_head = readout_head(hidden, n_classes, dropout)
        self.node_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                       nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, X, E):                       # X [B,T,N,F], E [B,T,C,N,N]
        if not self.use_graph:
            E = torch.zeros_like(E)
        B, T, N, Fd = X.shape
        x, e = X.reshape(B * T, N, Fd), E.reshape(B * T, *E.shape[2:])
        z = F.dropout(F.elu(self.gat1(x, e)), self.p, self.training)
        z = F.elu(self.gat2(z, e)) + self.skip(x)
        z = self.norm(z).reshape(B, T, N, -1)
        Ahat = gcn_norm(e[:, 0]).reshape(B, T, N, N)
        h = torch.zeros(B, N, self.hidden, device=X.device)
        for t in range(T):
            h = self.cell(z[:, t], h, Ahat[:, t])
        win = self.win_head(torch.cat([h.mean(1), h.max(1).values], -1))
        node = self.node_head(h).squeeze(-1)       # [B,N]
        return win, node


# -------------------------------------------------------------------- metrics
def loc_metrics(scores, comp):
    """scores [S,N] (higher = riskier); comp [S] true node index, -1 = none (ignored)."""
    scores = torch.as_tensor(scores).detach().float().cpu()
    comp = torch.as_tensor(comp).cpu()
    m = comp >= 0
    n = int(m.sum())
    if n == 0:
        return {"top1": 0.0, "top3": 0.0, "mrr": 0.0, "n": 0}
    s, c = scores[m], comp[m]
    true_s = s.gather(1, c.unsqueeze(1))
    greater = (s > true_s).sum(1).float()
    equal = (s == true_s).sum(1).float()
    rank = greater + (equal - 1) / 2 + 1           # ties share the average rank
    return {"top1": float((rank <= 1).float().mean()), "top3": float((rank <= 3).float().mean()),
            "mrr": float((1.0 / rank).mean()), "n": n}


def loc_by_phase(scores, comp, phases):
    scores = torch.as_tensor(scores).detach().float().cpu()
    comp = torch.as_tensor(comp).cpu()
    phases = np.asarray(phases)
    out = {}
    for ph in sorted(set(phases[(comp >= 0).numpy()])):
        sel = torch.from_numpy(phases == ph)
        out[str(ph)] = loc_metrics(scores[sel], comp[sel])
    return out


@torch.no_grad()
def predict(model, split, bs=256):
    model.eval()
    wl, nl = [], []
    for i in range(0, len(split["y"]), bs):
        w, n = model(split["X"][i:i + bs], split["E"][i:i + bs])
        wl.append(w)
        nl.append(n)
    return torch.cat(wl), torch.cat(nl)


# ------------------------------------------------------------------- training
def train_one(use_graph, data, seed, device, args):
    set_seed(seed)
    tr, va, te = data["train"], data["val"], data["test"]
    model = NodeRiskNet(tr["X"].shape[-1], args.hidden, args.heads, 2, 3, args.dropout, use_graph).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=8)
    crit = nn.CrossEntropyLoss(weight=data["class_weights"], label_smoothing=0.05)
    n = len(tr["y"])
    best_key, best_state, best_ep, bad = (-1.0, -1.0), None, 0, 0

    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, args.bs):
            b = perm[i:i + args.bs]
            wl, nl = model(tr["X"][b], tr["E"][b])
            loss = crit(wl, tr["y"][b])
            c = tr["comp"][b]
            m = c >= 0
            if m.any():
                loss = loss + args.lam * F.cross_entropy(nl[m], c[m])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        wl, nl = predict(model, va)
        vmrr = loc_metrics(nl, va["comp"])["mrr"]
        vf1 = metrics(va["y"].cpu(), wl.argmax(1).cpu())["macro_f1"]
        sched.step(vmrr)
        if (vmrr, vf1) > best_key:
            best_key, best_ep, bad = (vmrr, vf1), ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= args.patience:
            break

    model.load_state_dict(best_state)
    wl, nl = predict(model, te)
    res = {"seed": seed, "best_epoch": best_ep, "val_mrr": best_key[0],
           "loc": loc_metrics(nl, te["comp"]),
           "window": metrics(te["y"].cpu(), wl.argmax(1).cpu())}
    return model, res, nl


def rf_per_node(data, seed):
    """Tabular baseline: each (window, node) is a row of that node's own features over the
    T input windows; label = is this the compromised node. No neighbours, no graph."""
    def feats(split):
        X = split["X"].cpu().numpy()
        S, T, N, Fd = X.shape
        return X.transpose(0, 2, 1, 3).reshape(S, N, T * Fd)

    Xtr, Xte = feats(data["train"]), feats(data["test"])
    comp = data["train"]["comp"].cpu().numpy()
    keep = comp >= 0
    N = Xtr.shape[1]
    Xf = Xtr[keep].reshape(-1, Xtr.shape[-1])
    yf = (np.arange(N)[None, :] == comp[keep][:, None]).reshape(-1).astype(int)
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
                                n_jobs=-1, random_state=seed).fit(Xf, yf)
    return rf.predict_proba(Xte.reshape(-1, Xte.shape[-1]))[:, 1].reshape(Xte.shape[0], N)


# ------------------------------------------------------------ attack paths
def rank_paths(node_risk, adj, node_order, n_starts=3, max_nodes=4, top=5):
    """HEURISTIC (not validated): directed paths of 3..max_nodes nodes starting at one of the
    n_starts riskiest nodes; score = mean node risk along the path."""
    import networkx as nx
    G = nx.DiGraph()
    src, dst = np.nonzero(adj)
    G.add_edges_from(zip(src.tolist(), dst.tolist()))
    seen, paths = set(), []
    for s in np.argsort(-node_risk)[:n_starts]:
        s = int(s)
        if s not in G:
            continue
        for t in list(G.nodes):
            if t == s:
                continue
            for p in nx.all_simple_paths(G, s, t, cutoff=max_nodes - 1):
                if len(p) < 3 or tuple(p) in seen:
                    continue
                seen.add(tuple(p))
                paths.append((float(np.mean(node_risk[p])), p))
    paths.sort(key=lambda x: (-x[0], -len(x[1])))
    return [{"nodes": [node_order[i] for i in p], "score": round(sc, 3)} for sc, p in paths[:top]]


@torch.no_grad()
def export_demo(model, data, path, n_loc=8, n_other=4):
    te, names = data["test"], data["node_order"]
    comp = te["comp"].cpu().numpy()
    loc_idx, oth_idx = np.nonzero(comp >= 0)[0], np.nonzero(comp < 0)[0]

    def spread(a, k):
        return [] if len(a) == 0 else list(a[np.linspace(0, len(a) - 1, min(k, len(a))).astype(int)])

    model.eval()
    items = []
    for i in spread(loc_idx, n_loc) + spread(oth_idx, n_other):
        i = int(i)
        wl, nl = model(te["X"][i:i + 1], te["E"][i:i + 1])
        pw, pn = torch.softmax(wl, -1)[0].cpu(), torch.softmax(nl, -1)[0].cpu()
        risk = ((pn / pn.max()) * (1.0 - pw[0])).numpy()
        adj = (te["E"][i, -1, 0] > 0).cpu().numpy()
        items.append({
            "run_id": te["run_id"][i], "phase": str(te["phase"][i]),
            "window_label": data["label_names"][int(te["y"][i])],
            "true_compromised": names[comp[i]] if comp[i] >= 0 else None,
            "risk_class": data["label_names"][int(pw.argmax())],
            "probabilities": {data["label_names"][k]: round(float(pw[k]), 3) for k in range(3)},
            "node_risk": {names[k]: round(float(risk[k]), 3) for k in range(len(names))},
            "top_paths": rank_paths(risk, adj, names),
        })
    with open(path, "w") as f:
        json.dump({"top_paths_method": "heuristic: mean node risk along directed paths from the 3 "
                                       "riskiest nodes (not validated)", "items": items}, f, indent=2)


# ----------------------------------------------------------------------- main
def fmt(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    if not v:
        return "      -      "
    return f"{np.mean(v):.3f}+/-{np.std(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}        "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lam", type=float, default=1.0, help="weight of the node-localization loss")
    ap.add_argument("--graph-only", action="store_true",
                    help="train only the full graph model (skip the no-graph ablation and the per-node RF)")
    ap.add_argument("--tag", default="node_h0")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_dataset(args.data, device)
    for s in ("train", "val", "test"):
        data[s]["comp"] = data[s]["comp"].to(device)
    te = data["test"]
    N = len(data["node_order"])
    print(f"horizon={data.get('horizon', 0)} seq_len={data['seq_len']} features={data['feature_names']}")
    print(f"windows with a compromised node: train={int((data['train']['comp'] >= 0).sum())} "
          f"val={int((data['val']['comp'] >= 0).sum())} test={int((te['comp'] >= 0).sum())}")

    rows = {}
    rows["Random (chance)"] = [{"top1": 1 / N, "top3": 3 / N,
                                "mrr": sum(1 / r for r in range(1, N + 1)) / N, "f1": None}]
    fn, mean, std = data["feature_names"], data["norm"]["mean"], data["norm"]["std"]
    last = te["X"][:, -1].cpu()
    raw = lambda i: last[:, :, i] * std[i] + mean[i]          # undo z-score
    heur = raw(fn.index("in_strength")) + raw(fn.index("out_strength"))
    rows["Traffic heuristic (in+out strength)"] = [{**loc_metrics(heur, te["comp"]), "f1": None}]

    best = {"model": None, "val": -1.0, "nl": None}
    for sd in args.seeds:
        if not args.graph_only:
            p = rf_per_node(data, sd)
            rows.setdefault("RF per node (no graph)", []).append({**loc_metrics(p, te["comp"]), "f1": None})
        for use_graph, name in ((False, "T-GCN+GAT node model (no graph)"), (True, "T-GCN+GAT node model (full)")):
            if args.graph_only and not use_graph:
                continue
            model, r, nl = train_one(use_graph, data, sd, device, args)
            rows.setdefault(name, []).append({**r["loc"], "f1": r["window"]["macro_f1"]})
            print(f"seed {sd} {name:34s} top1={r['loc']['top1']:.3f} top3={r['loc']['top3']:.3f} "
                  f"mrr={r['loc']['mrr']:.3f} windowF1={r['window']['macro_f1']:.3f}")
            if use_graph and r["val_mrr"] > best["val"]:
                best = {"model": model, "val": r["val_mrr"], "nl": nl}

    n_te = rows["T-GCN+GAT node model (full)"][0]["n"]
    print(f"\nNode localization on TEST windows with a known compromised node (n={n_te}), "
          f"{N} candidate nodes")
    print(f"{'Method':38s} {'top-1':>13s} {'top-3':>13s} {'MRR':>13s} {'window F1':>13s}")
    for name, rs in rows.items():
        print(f"{name:38s} {fmt(rs, 'top1'):>13s} {fmt(rs, 'top3'):>13s} "
              f"{fmt(rs, 'mrr'):>13s} {fmt(rs, 'f1'):>13s}")

    byp = loc_by_phase(best["nl"], te["comp"], te["phase"])
    print("\nFull model, best-validation seed, by attack phase (top-1 / top-3 / n):")
    for ph, r in byp.items():
        print(f"  {ph:8s} {r['top1']:.3f} / {r['top3']:.3f} / {r['n']}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best["model"].state_dict(),
                "config": dict(in_dim=te["X"].shape[-1], hidden=args.hidden, heads=args.heads,
                               edge_dim=2, n_classes=3, dropout=args.dropout),
                "feature_names": data["feature_names"], "norm": data["norm"],
                "node_order": data["node_order"], "seq_len": data["seq_len"],
                "horizon": data.get("horizon", 0), "val_mrr": best["val"]},
               CKPT_DIR / f"node_risk_{args.tag}.pt")
    with open(RESULTS_DIR / f"node_risk_{args.tag}.json", "w") as f:
        json.dump({"rows": rows, "by_phase": byp, "args": vars(args)}, f, indent=2)
    export_demo(best["model"], data, RESULTS_DIR / f"node_risk_demo_{args.tag}.json")
    print(f"\ncheckpoint -> {CKPT_DIR / f'node_risk_{args.tag}.pt'}")
    print(f"dashboard sample -> {RESULTS_DIR / f'node_risk_demo_{args.tag}.json'}")


if __name__ == "__main__":
    main()