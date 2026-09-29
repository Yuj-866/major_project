"""build_dataset.py -- steps 1-4 of the pipeline.

  (anomaly_score is dropped from the inputs by default; see --drop-features)
  1. Check/clean   graph_model/snapshots/window_*.json
  2. JSON -> tensors (x, edge_index, edge_attr, y), PyG-ready via to_pyg()
  3. Temporal sequences, built strictly WITHIN a run_id (never across runs)
  4. Split by whole runs -> train / val / test (scaler fit on train only)

Output (one shareable file, small):
  graph_model/processed/dataset.pt
  graph_model/processed/dataset_report.json

Place this file in the project root (next to split_dataset.py) and run:
  python build_dataset.py
  python build_dataset.py --seq-len 5 --train-ratio 0.7 --val-ratio 0.15

Load it anywhere (only torch needed):
  data = torch.load("dataset.pt", weights_only=True)
  train = data["splits"]["train"]      # {"snapshots": [...], "sequences": [...], "runs": [...]}

Each sequence = seq_len consecutive windows from ONE run:
  idx     -> indices into that split's "snapshots" list (oldest -> newest)
  y_last  -> label of the newest window in the sequence (classify current state)
  y_next  -> label of the following window (-1 if none), for forecasting
"""
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
SNAP_DIR = BASE_DIR / "graph_model" / "snapshots"
OUT_DIR = BASE_DIR / "graph_model" / "processed"

LABELS = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
REQUIRED = ["window", "label", "run_id", "nodes", "feature_names",
            "feature_matrix", "edges"]


# ---------------------------------------------------------------- step 1
def load_snapshots(snap_dir):
    files = sorted(snap_dir.glob("window_*.json"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        raise SystemExit(f"No snapshots found in {snap_dir}")
    out = []
    for f in files:
        try:
            with open(f) as fh:
                out.append((f.name, json.load(fh)))
        except Exception as e:
            out.append((f.name, {"_load_error": str(e)}))
    return out


def check_snapshot(s, ref_nodes, ref_features):
    """Return a list of problems (empty list = snapshot is clean)."""
    if "_load_error" in s:
        return [f"unreadable json: {s['_load_error']}"]
    missing = [k for k in REQUIRED if k not in s]
    if missing:
        return [f"missing keys {missing}"]
    problems = []
    if s["label"] not in LABELS:
        problems.append(f"bad label {s['label']!r}")
    if not s["run_id"] or s["run_id"] == "unknown":
        problems.append("run_id unknown (pre-fix / pipeline-test snapshot)")
    m = np.array(s["feature_matrix"], dtype=float)
    if m.ndim != 2 or m.shape[0] != len(s["nodes"]):
        problems.append(f"feature_matrix shape {m.shape} vs {len(s['nodes'])} nodes")
    elif m.shape[1] != len(s["feature_names"]):
        problems.append("feature count != feature_names")
    elif not np.isfinite(m).all():
        problems.append("NaN/inf in features")
    if ref_nodes is not None and s["nodes"] != ref_nodes:
        problems.append("node order differs from first snapshot")
    if ref_features is not None and s["feature_names"] != ref_features:
        problems.append("feature_names differ from first snapshot")
    return problems


# ---------------------------------------------------------------- step 2
def tensorize(s, node_idx, mean, std, keep):
    raw = np.array(s["feature_matrix"], dtype=np.float32)[:, keep]
    x = torch.tensor((raw - mean) / std, dtype=torch.float32)
    src, dst, attr, dropped = [], [], [], 0
    for e in s["edges"]:
        if e["source"] in node_idx and e["target"] in node_idx:
            src.append(node_idx[e["source"]])
            dst.append(node_idx[e["target"]])
            attr.append([math.log1p(float(e["weight"])),
                         1.0 if e.get("type") == "dynamic_attack" else 0.0])
        else:
            dropped += 1
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(attr, dtype=torch.float32)
    return {
        "run_id": s["run_id"], "window": int(s["window"]),
        "cycle_window": s.get("cycle_window"),
        "scheduled_phase": s.get("scheduled_phase"),   # metadata only, NOT a feature
        "x": x, "edge_index": edge_index, "edge_attr": edge_attr,
        "y": LABELS[s["label"]],
        "event_count": int(s.get("event_count", 0)),
    }, dropped


def to_pyg(snap):
    """Convert one stored snapshot dict to a torch_geometric Data object."""
    from torch_geometric.data import Data
    return Data(x=snap["x"], edge_index=snap["edge_index"],
                edge_attr=snap["edge_attr"], y=torch.tensor([snap["y"]]))


# ---------------------------------------------------------------- step 3
def make_sequences(snaps, seq_len):
    by_run = defaultdict(list)
    for i, s in enumerate(snaps):
        by_run[s["run_id"]].append(i)
    seqs = []
    for run_id, idxs in by_run.items():
        for j in range(seq_len - 1, len(idxs)):
            win = idxs[j - seq_len + 1: j + 1]
            # windows must be consecutive (no gap from deleted/missing files)
            if snaps[win[-1]]["window"] - snaps[win[0]]["window"] != seq_len - 1:
                continue
            y_next = -1
            if j + 1 < len(idxs) and snaps[idxs[j + 1]]["window"] == snaps[win[-1]]["window"] + 1:
                y_next = snaps[idxs[j + 1]]["y"]
            seqs.append({"idx": win, "run_id": run_id,
                         "y_last": snaps[win[-1]]["y"], "y_next": y_next})
    return seqs


# ---------------------------------------------------------------- step 4
def split_runs(run_ids, train_ratio, val_ratio, shuffle, seed):
    run_ids = list(run_ids)
    if shuffle:
        random.Random(seed).shuffle(run_ids)
    n = len(run_ids)
    if n < 3:
        raise SystemExit(f"Need at least 3 complete runs to split, found {n}.")
    n_train = max(1, round(n * train_ratio))
    n_val = max(1, round(n * val_ratio))
    if n_train + n_val >= n:               # always leave >=1 run for test
        n_train = max(1, n - n_val - 1)
        n_val = max(1, n - n_train - 1)
    return (run_ids[:n_train], run_ids[n_train:n_train + n_val],
            run_ids[n_train + n_val:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap-dir", type=Path, default=SNAP_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--seq-len", type=int, default=5)
    ap.add_argument("--min-run-windows", type=int, default=13,
                    help="Drop runs with fewer windows (partial last cycle). "
                         "Use your --cycle-windows value.")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--shuffle", action="store_true",
                    help="Random run-level split instead of chronological.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop-features", nargs="*", default=["anomaly_score"],
                    help="Feature columns to remove from the model input. Default drops "
                         "anomaly_score because it is computed from the same counts as the "
                         "label (circular). Pass '--drop-features' with no names to keep all.")
    args = ap.parse_args()

    # ---- step 1: check / clean
    raw = load_snapshots(args.snap_dir)
    ref_nodes = ref_features = None
    for _, s in raw:
        if "nodes" in s and "feature_names" in s:
            ref_nodes, ref_features = s["nodes"], s["feature_names"]
            break

    clean, rejected = [], []
    for name, s in raw:
        probs = check_snapshot(s, ref_nodes, ref_features)
        (rejected if probs else clean).append((name, s, probs))
    print(f"[1] snapshots: {len(raw)} found, {len(clean)} clean, {len(rejected)} rejected")
    for name, _, probs in rejected[:10]:
        print(f"    rejected {name}: {'; '.join(probs)}")
    if len(rejected) > 10:
        print(f"    ... and {len(rejected) - 10} more")

    by_run = defaultdict(list)
    for name, s, _ in clean:
        by_run[s["run_id"]].append(s)
    kept_runs, dropped_runs = {}, {}
    for run_id in sorted(by_run):
        snaps = sorted(by_run[run_id], key=lambda s: s["window"])
        (kept_runs if len(snaps) >= args.min_run_windows else dropped_runs)[run_id] = snaps
    print(f"    runs: {len(kept_runs)} kept, {len(dropped_runs)} dropped "
          f"(< {args.min_run_windows} windows): "
          f"{ {r: len(v) for r, v in dropped_runs.items()} }")
    if not kept_runs:
        raise SystemExit("No complete runs left. Check --min-run-windows.")

    unknown = [f for f in args.drop_features if f not in ref_features]
    if unknown:
        raise SystemExit(f"--drop-features names not found in snapshots: {unknown}")
    keep = [i for i, n in enumerate(ref_features) if n not in args.drop_features]
    kept_features = [ref_features[i] for i in keep]
    print(f"    input features: {len(kept_features)} kept, dropped {args.drop_features}")

    all_kept = [s for r in kept_runs.values() for s in r]
    # label vs scheduled phase (label comes from real traffic, not the schedule)
    xt = Counter((s.get("scheduled_phase") or "idle", s["label"]) for s in all_kept)
    print("    scheduled_phase x label (kept snapshots):")
    for ph in ["idle", "recon", "brute", "exploit", "exfil"]:
        row = {lb: xt.get((ph, lb), 0) for lb in LABELS}
        print(f"      {ph:8s} {row}")

    # ---- step 4a: split runs (before scaling -> no leakage)
    train_runs, val_runs, test_runs = split_runs(
        sorted(kept_runs), args.train_ratio, args.val_ratio, args.shuffle, args.seed)
    print(f"[4] runs -> train {len(train_runs)}, val {len(val_runs)}, test {len(test_runs)}")

    # ---- scaler fit on TRAIN only
    train_mat = np.concatenate(
        [np.array(s["feature_matrix"], dtype=np.float32)[:, keep]
         for r in train_runs for s in kept_runs[r]], axis=0)
    mean = train_mat.mean(axis=0)
    std = train_mat.std(axis=0)
    const_cols = [kept_features[i] for i in range(len(std)) if std[i] < 1e-8]
    std[std < 1e-8] = 1.0
    if const_cols:
        print(f"    constant (zero-variance) features in train: {const_cols}")

    # ---- step 2 + 3 per split
    node_idx = {n: i for i, n in enumerate(ref_nodes)}
    splits, dropped_edges = {}, 0
    for name, runs in [("train", train_runs), ("val", val_runs), ("test", test_runs)]:
        snaps = []
        for r in runs:
            for s in kept_runs[r]:
                t, d = tensorize(s, node_idx, mean, std, keep)
                snaps.append(t)
                dropped_edges += d
        seqs = make_sequences(snaps, args.seq_len)
        dist = Counter(q["y_last"] for q in seqs)
        inv = {v: k for k, v in LABELS.items()}
        print(f"[2-3] {name:5s}: {len(runs)} runs, {len(snaps)} snapshots, "
              f"{len(seqs)} sequences, class counts "
              f"{ {inv[k]: dist.get(k, 0) for k in LABELS.values()} }")
        if name != "train" and any(dist.get(k, 0) == 0 for k in LABELS.values()):
            print(f"      [WARN] {name} split is missing a class -- consider --shuffle "
                  f"or collecting more runs")
        splits[name] = {"runs": runs, "snapshots": snaps, "sequences": seqs}
    if dropped_edges:
        print(f"    dropped {dropped_edges} edges pointing to nodes outside the 25-node set")

    meta = {
        "nodes": ref_nodes, "feature_names": kept_features,
        "dropped_features": args.drop_features,
        "edge_feature_names": ["log1p_weight", "is_attack_edge"],
        "label_map": LABELS, "seq_len": args.seq_len,
        "scaler_mean": mean.tolist(), "scaler_std": std.tolist(),
        "split_strategy": "shuffle" if args.shuffle else "chronological",
        "seed": args.seed if args.shuffle else None,
        "constant_features": const_cols,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "splits": splits}, args.out_dir / "dataset.pt")

    report = {
        "clean": len(clean), "rejected": len(rejected),
        "dropped_runs": {r: len(v) for r, v in dropped_runs.items()},
        "runs": {"train": train_runs, "val": val_runs, "test": test_runs},
        "sequences": {k: len(v["sequences"]) for k, v in splits.items()},
        "phase_x_label": {f"{p}|{l}": c for (p, l), c in xt.items()},
    }
    with open(args.out_dir / "dataset_report.json", "w") as f:
        json.dump(report, f, indent=2)
    size_mb = (args.out_dir / "dataset.pt").stat().st_size / 1e6
    print(f"Saved {args.out_dir / 'dataset.pt'} ({size_mb:.1f} MB) + dataset_report.json")


if __name__ == "__main__":
    main()