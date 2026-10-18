"""inference.py -- the ONLY file the backend needs to call the trained node-risk model.

    from graph_model.inference import RiskPredictor, BadInput
    predictor = RiskPredictor()                       # once at startup (loads the checkpoint)
    result = predictor.predict(last_windows)          # POST /predict   (list of snapshot dicts)
    graph  = predictor.graph_view(last_windows[-1], result)   # GET /graph/latest
    meta   = predictor.meta()                         # GET /meta

Everything the model needs (feature list, normalisation, node order, sequence length,
architecture config) is read from the checkpoint, so retraining and overwriting
graph_model/checkpoints/node_risk_node_h0.pt needs NO backend code change.
It reproduces the preprocessing of build_node_dataset.py exactly.

Requirements: torch, numpy, networkx, scikit-learn (train_node_risk imports it).

Quick test / mock-data generator:
    python -m graph_model.inference --limit 12 --out graph_model\\results\\replay_sample.json
"""
import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from graph_model.snapshot_builder import BASE_EDGES, FEATURE_NAMES as RAW_FEATURE_NAMES  # noqa: E402
from graph_model.train_node_risk import NodeRiskNet, rank_paths  # noqa: E402

DEFAULT_CKPT = BASE_DIR / "graph_model" / "checkpoints" / "node_risk_node_h0.pt"
LABELS = ["LOW", "MEDIUM", "HIGH"]


class BadInput(ValueError):
    """Bad request data -> HTTP 400 with contract error code BAD_INPUT."""


class RiskPredictor:
    def __init__(self, ckpt_path=DEFAULT_CKPT, device="cpu"):
        path = Path(ckpt_path)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.device = device
        self.feature_names = list(ck["feature_names"])
        self.node_order = list(ck["node_order"])
        self.seq_len = int(ck["seq_len"])
        self.horizon = int(ck.get("horizon", 0))
        self.mean = torch.as_tensor(ck["norm"]["mean"]).float().numpy()
        self.std = torch.as_tensor(ck["norm"]["std"]).float().numpy()
        self.node_idx = {n: i for i, n in enumerate(self.node_order)}
        self.base = {(self.node_idx[s], self.node_idx[t]) for s, t in BASE_EDGES
                     if s in self.node_idx and t in self.node_idx}
        self.model = NodeRiskNet(**ck["config"]).to(device)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.model_name = path.stem
        self.model_version = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%dT%H:%M")

    # ------------------------------------------------------------ preprocessing
    def _edges(self, w):
        n = len(self.node_order)
        E = np.zeros((2, n, n), dtype=np.float32)
        for e in w.get("edges", []):
            i, j = self.node_idx.get(e.get("source")), self.node_idx.get(e.get("target"))
            if i is None or j is None or i == j:
                continue
            E[0, i, j] = math.log1p(float(e.get("weight", 1))) / 5.0
            E[1, i, j] = 0.0 if (i, j) in self.base else 1.0
        return E

    def _features(self, w, E):
        if "feature_matrix" not in w:
            raise BadInput("window is missing feature_matrix")
        fm = np.asarray(w["feature_matrix"], dtype=np.float32)
        nodes = w.get("nodes")
        if nodes and list(nodes) != self.node_order:
            try:
                fm = fm[[list(nodes).index(n) for n in self.node_order]]
            except ValueError:
                raise BadInput("window nodes do not match the model's node_order")
        if fm.shape[0] != len(self.node_order):
            raise BadInput(f"feature_matrix has {fm.shape[0]} rows, expected {len(self.node_order)}")
        col = {n: k for k, n in enumerate(w.get("feature_names") or RAW_FEATURE_NAMES)}
        derived = {"in_strength": E[0].sum(0), "out_strength": E[0].sum(1),
                   "new_conn": E[1].sum(0) + E[1].sum(1)}
        cols = []
        for name in self.feature_names:
            if name in derived:
                cols.append(derived[name])
            elif name in col:
                cols.append(fm[:, col[name]])
            else:
                raise BadInput(f"window is missing feature '{name}'")
        X = np.stack(cols, axis=-1).astype(np.float32)
        return (X - self.mean) / self.std

    # --------------------------------------------------------------- public API
    def predict(self, windows, pad_short=False):
        """windows: list of snapshot dicts, oldest first. Returns the /predict response."""
        if not windows:
            raise BadInput("no windows given")
        if len(windows) < self.seq_len and not pad_short:
            raise BadInput(f"need {self.seq_len} windows, got {len(windows)}")
        ws = list(windows)[-self.seq_len:]
        ws = [ws[0]] * (self.seq_len - len(ws)) + ws          # same left-padding as training
        Es = [self._edges(w) for w in ws]
        Xs = [self._features(w, E) for w, E in zip(ws, Es)]
        X = torch.from_numpy(np.stack(Xs))[None].to(self.device)    # [1,T,N,F]
        E = torch.from_numpy(np.stack(Es))[None].to(self.device)    # [1,T,2,N,N]
        with torch.no_grad():
            wl, nl = self.model(X, E)
        pw = torch.softmax(wl, -1)[0].cpu()
        pn = torch.softmax(nl, -1)[0].cpu()
        risk = ((pn / pn.max()) * (1.0 - pw[0])).numpy()            # quiet windows -> low risk everywhere
        adj = E[0, -1, 0].cpu().numpy() > 0
        return {
            "window_end": ws[-1].get("window_end"),
            "risk_class": LABELS[int(pw.argmax())],
            "probabilities": {LABELS[k]: round(float(pw[k]), 3) for k in range(3)},
            "horizon": self.horizon,
            "node_risk": {n: round(float(risk[i]), 3) for i, n in enumerate(self.node_order)},
            "top_paths": rank_paths(risk, adj, self.node_order),
            "placeholder": [],
            "heuristic": ["top_paths"],            # additive field: computed by a heuristic, not validated
            "model": {"name": self.model_name, "version": self.model_version},
        }

    def graph_view(self, window, result):
        """/graph/latest: nodes with risk + edges flagged when they are not in the base topology."""
        edges = []
        for e in window.get("edges", []):
            i, j = self.node_idx.get(e.get("source")), self.node_idx.get(e.get("target"))
            if i is None or j is None or i == j:
                continue
            edges.append({"source": e["source"], "target": e["target"], "weight": e.get("weight", 1),
                          "type": e.get("type", "base"), "is_new": (i, j) not in self.base})
        return {"window_end": result["window_end"], "risk_class": result["risk_class"],
                "nodes": [{"id": n, "risk": result["node_risk"][n], "type": "unknown"} for n in self.node_order],
                "edges": edges}

    def meta(self):
        return {"model": {"name": self.model_name, "version": self.model_version},
                "seq_len": self.seq_len, "horizon": self.horizon,
                "feature_names": self.feature_names, "node_order": self.node_order,
                "window_seconds": 30, "labels": LABELS}


# ------------------------------------------------------------------ CLI / demo
def _load_dir(d):
    ws = []
    for f in Path(d).glob("window_*.json"):
        with open(f) as fh:
            ws.append(json.load(fh))
    ws.sort(key=lambda w: w["window_start"])
    return ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--dir", default=str(BASE_DIR / "graph_model" / "snapshots" / "snapshots_dynamic"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--out", default=None, help="write the responses as JSON (mock data for the frontend)")
    args = ap.parse_args()

    pred = RiskPredictor(args.ckpt)
    print(f"model={pred.model_name} version={pred.model_version} seq_len={pred.seq_len} "
          f"horizon={pred.horizon} features={len(pred.feature_names)}")
    ws = _load_dir(args.dir)
    items = []
    for k in range(args.start, min(args.start + args.limit, len(ws))):
        res = pred.predict(ws[max(0, k - pred.seq_len + 1):k + 1], pad_short=True)
        top3 = sorted(res["node_risk"], key=res["node_risk"].get, reverse=True)[:3]
        print(f"window {k:4d} | true {ws[k].get('label', '?'):6s} node={str(ws[k].get('compromised_node')):18s} "
              f"| pred {res['risk_class']:6s} top3={top3}")
        items.append(res)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"meta": pred.meta(), "items": items}, f, indent=2)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()