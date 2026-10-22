"""build_dataset.py -- dynamic v2 snapshots -> run-level-split sequence dataset.

Run from project root:
    python -m graph_model.build_dataset --seq-len 3 --horizon 1 --edge-features

New in this version
  --horizon H        label = ground-truth label of the window H steps AFTER the last
                     input window (same run_id). H=0 -> classify current window,
                     H=1 -> early warning (predict the NEXT window).
  --edge-features    add 3 per-node features computed from the edges:
                     in_strength, out_strength, new_conn (non-base edges touching node)

The run-level split depends only on runs + seed, so datasets built with different
horizon / feature settings share the SAME train/val/test runs (comparable results).
"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from graph_model.snapshot_builder import NODE_ORDER, BASE_EDGES, FEATURE_NAMES  # noqa: E402

LABELS = ["LOW", "MEDIUM", "HIGH"]
L2I = {l: i for i, l in enumerate(LABELS)}
N = len(NODE_ORDER)
NODE_IDX = {n: i for i, n in enumerate(NODE_ORDER)}
BASE_SET = {(NODE_IDX[s], NODE_IDX[t]) for s, t in BASE_EDGES}
EDGE_FEATURE_NAMES = ["in_strength", "out_strength", "new_conn"]


def load_windows(snap_dirs, min_events):
    rows, dropped = [], Counter()
    for d in snap_dirs:
        for f in sorted(Path(d).glob("window_*.json")):
            try:
                with open(f) as fh:
                    s = json.load(fh)
            except Exception:
                dropped["unreadable"] += 1
                continue
            if s.get("feature_names") != FEATURE_NAMES:
                dropped["old_feature_schema"] += 1
            elif s.get("label_source") != "ground_truth":
                dropped["no_ground_truth_match"] += 1
            elif s.get("run_id", "unknown") == "unknown":
                dropped["unknown_run_id"] += 1
            elif int(s.get("event_count", 0)) < min_events:
                dropped["empty_window"] += 1
            elif s.get("label") not in L2I:
                dropped["bad_label"] += 1
            else:
                s["_t"] = datetime.fromisoformat(s["window_start"])
                rows.append(s)
    rows.sort(key=lambda s: s["_t"])
    return rows, dropped


def frame_arrays(s, keep_idx):
    X = np.asarray(s["feature_matrix"], dtype=np.float32)[:, keep_idx]
    E = np.zeros((2, N, N), dtype=np.float32)
    for e in s["edges"]:
        i, j = NODE_IDX.get(e["source"]), NODE_IDX.get(e["target"])
        if i is None or j is None or i == j:
            continue
        E[0, i, j] = math.log1p(float(e["weight"])) / 5.0
        E[1, i, j] = 0.0 if (i, j) in BASE_SET else 1.0
    return X, E


def make_sequences(rows, seq_len, horizon):
    """idx [S,T] input windows, tgt [S] target window; never crosses a run_id."""
    runs = defaultdict(list)
    for i, s in enumerate(rows):
        runs[s["run_id"]].append(i)
    idx, tgt, seq_run = [], [], []
    for rid, ids in runs.items():
        for p in range(len(ids) - horizon):
            win = ids[max(0, p - seq_len + 1): p + 1]
            win = [win[0]] * (seq_len - len(win)) + win
            idx.append(win)
            tgt.append(ids[p + horizon])
            seq_run.append(rid)
    return np.asarray(idx), np.asarray(tgt), seq_run, runs


def split_runs(runs, rows, seed, frac=0.15, min_per_class=6, tries=5000):
    rids = sorted(runs)
    cnt = {r: Counter(rows[i]["label"] for i in runs[r]) for r in rids}
    tot = sum(sum(c.values()) for c in cnt.values())
    nv = nt = max(1, round(frac * len(rids)))
    rng = np.random.RandomState(seed)
    best, best_score = None, 1e9
    for _ in range(tries):
        p = list(rng.permutation(len(rids)))
        val = [rids[i] for i in p[:nv]]
        test = [rids[i] for i in p[nv:nv + nt]]
        train = [rids[i] for i in p[nv + nt:]]
        ok = True
        for part in (val, test):
            c = Counter()
            for r in part:
                c.update(cnt[r])
            if min(c[l] for l in LABELS) < min_per_class:
                ok = False
        if not ok:
            continue
        fv = sum(sum(cnt[r].values()) for r in val) / tot
        ft = sum(sum(cnt[r].values()) for r in test) / tot
        score = abs(fv - frac) + abs(ft - frac)
        if score < best_score:
            best, best_score = (train, val, test), score
            if score < 0.02:
                break
    if best is None:
        raise SystemExit("Could not find a split with all classes in val/test. "
                         "Lower --min-per-class or collect more data.")
    return best


def boundary_candidates(rows):
    """Windows that look like the LAST tick of an episode but carry the NEXT run_id
    (simulator logs attacker.run_id after step() rotated it). Signature: first window
    of a run is non-idle and the next window of that run is idle."""
    first = {}
    for i, s in enumerate(rows):
        first.setdefault(s["run_id"], i)
    cands = []
    for rid, i in first.items():
        if i == 0 or rows[i - 1]["run_id"] == rid:
            continue
        nxt = rows[i + 1] if i + 1 < len(rows) and rows[i + 1]["run_id"] == rid else None
        if rows[i].get("scheduled_phase") != "idle" and nxt is not None and nxt.get("scheduled_phase") == "idle":
            cands.append(i)
    return cands, len(first)


def chrono_split(runs, rows, min_per_class):
    """Rule 2: earliest 70% of runs = train, next 15% = val, latest 15% = test."""
    rids = list(runs)                      # insertion order == chronological order
    n = len(rids)
    a, b = int(n * 0.70), int(n * 0.85)
    train, val, test = rids[:a], rids[a:b], rids[b:]
    for name, part in (("val", val), ("test", test)):
        c = Counter(rows[i]["label"] for r in part for i in runs[r])
        if min(c[l] for l in LABELS) < min_per_class:
            counts = {l: c[l] for l in LABELS}
            print("[WARN] chronological %s split has few windows of some class: %s "
                  "-> metrics for that class will be unstable" % (name, counts))
    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap-dirs", nargs="+", default=[str(BASE_DIR / "graph_model" / "snapshots" / "snapshots_dynamic")])
    ap.add_argument("--out", default=str(BASE_DIR / "graph_model" / "dataset_dyn.pt"))
    ap.add_argument("--seq-len", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=0, help="0 = current window, 1 = next window")
    ap.add_argument("--edge-features", action="store_true",
                    help="add in_strength, out_strength, new_conn node features")
    ap.add_argument("--drop-features", nargs="*", default=["anomaly_score", "time_of_day"])
    ap.add_argument("--aug-copies", type=int, default=3, help="total copies of each TRAIN seq (1 = no aug)")
    ap.add_argument("--min-events", type=int, default=1)
    ap.add_argument("--min-per-class", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-mode", choices=["shuffle", "chrono"], default="shuffle",
                    help="shuffle = seeded random run-level split; chrono = earliest runs train, latest test")
    ap.add_argument("--fix-run-boundary", action="store_true",
                    help="move each episode's last tick back to its own run_id")
    args = ap.parse_args()

    rows, dropped = load_windows(args.snap_dirs, args.min_events)
    print(f"Loaded {len(rows)} usable windows; dropped: {dict(dropped)}")
    if len(rows) < 100:
        raise SystemExit("Too few windows.")

    cands, n_runs0 = boundary_candidates(rows)
    print(f"Run-boundary check: {len(cands)} of {n_runs0} runs start with a non-idle window "
          f"followed by idle (episode's last tick logged under the next run_id)")
    if args.fix_run_boundary:
        for i in cands:
            rows[i]["run_id"] = rows[i - 1]["run_id"]
        print(f"  -> fixed: moved {len(cands)} windows back to their own run")
    elif cands:
        print("  -> not fixed (use --fix-run-boundary to move them back)")

    keep_names = [f for f in FEATURE_NAMES if f not in args.drop_features]
    keep_idx = [FEATURE_NAMES.index(f) for f in keep_names]

    Xw, Ew = zip(*[frame_arrays(s, keep_idx) for s in rows])
    Xw, Ew = np.stack(Xw), np.stack(Ew)                      # [W,N,F], [W,2,N,N]

    if args.edge_features:
        E0, E1 = Ew[:, 0], Ew[:, 1]                          # [W,src,dst]
        ef = np.stack([E0.sum(1), E0.sum(2), E1.sum(1) + E1.sum(2)], axis=-1).astype(np.float32)
        Xw = np.concatenate([Xw, ef], axis=-1)
        keep_names = keep_names + EDGE_FEATURE_NAMES
    print(f"Features ({len(keep_names)}): {keep_names}")
    print(f"Task: predict label of window t+{args.horizon} from the last {args.seq_len} windows")

    yw = np.array([L2I[s["label"]] for s in rows])
    rule_w = np.array([L2I.get(s.get("label_traffic_rule"), 0) for s in rows])

    print("\nphase x label (all windows):")
    ct = Counter((s["scheduled_phase"], s["label"]) for s in rows)
    for ph in sorted({k[0] for k in ct}):
        print(f"  {ph:8s}", {l: ct[(ph, l)] for l in LABELS})

    idx, tgt, seq_run, runs = make_sequences(rows, args.seq_len, args.horizon)
    print(f"\n{len(runs)} runs, {len(idx)} sequences (seq_len={args.seq_len}, horizon={args.horizon})")
    if args.split_mode == "chrono":
        train_r, val_r, test_r = chrono_split(runs, rows, args.min_per_class)
    else:
        train_r, val_r, test_r = split_runs(runs, rows, args.seed, min_per_class=args.min_per_class)
    print(f"split mode: {args.split_mode}")
    which = {r: "train" for r in train_r}
    which.update({r: "val" for r in val_r})
    which.update({r: "test" for r in test_r})
    print(f"split runs: train={len(train_r)} val={len(val_r)} test={len(test_r)}")

    # z-score using TRAIN input windows only
    tr_w = sorted({w for sq, r in zip(idx, seq_run) if which[r] == "train" for w in sq})
    mean = Xw[tr_w].reshape(-1, Xw.shape[-1]).mean(0)
    std = Xw[tr_w].reshape(-1, Xw.shape[-1]).std(0)
    std = np.where(std < 1e-6, 1.0, std)
    Xn = (Xw - mean) / std
    dyn_mask = (Xw[tr_w].std(axis=0).mean(0) > 1e-6).astype(np.float32)  # features that vary over time
    print("time-varying features:", [n for n, m in zip(keep_names, dyn_mask) if m])

    g = torch.Generator().manual_seed(args.seed)
    out = {}
    for split in ("train", "val", "test"):
        sel = [i for i, r in enumerate(seq_run) if which[r] == split]
        ii = idx[sel]
        X = torch.from_numpy(Xn[ii]).float()                  # [S,T,N,F]
        E = torch.from_numpy(Ew[ii]).float()                  # [S,T,2,N,N]
        y = torch.from_numpy(yw[tgt[sel]]).long()             # label of target window
        cy = torch.from_numpy(yw[ii[:, -1]]).long()           # true label of last input window
        rl = torch.from_numpy(rule_w[ii[:, -1]]).long()       # traffic-rule label of last input window
        runs_s = [seq_run[i] for i in sel]
        if split == "train" and args.aug_copies > 1:
            Xs, Es, ys, cs, rs, rr = [X], [E], [y], [cy], [rl], [runs_s]
            dm = torch.from_numpy(dyn_mask)
            for _ in range(args.aug_copies - 1):
                Xs.append(X + 0.05 * torch.randn(X.shape, generator=g) * dm)
                jit = torch.exp(0.1 * torch.randn(E[:, :, 0].shape, generator=g))
                Ec = E.clone()
                Ec[:, :, 0] = Ec[:, :, 0] * jit
                Es.append(Ec)
                ys.append(y); cs.append(cy); rs.append(rl); rr.append(runs_s)
            X, E, y, cy, rl = torch.cat(Xs), torch.cat(Es), torch.cat(ys), torch.cat(cs), torch.cat(rs)
            runs_s = [r for part in rr for r in part]
        out[split] = {"X": X, "E": E, "y": y, "cur_y": cy, "rule_label": rl, "run_id": runs_s}
        c = Counter(y.tolist())
        print(f"{split:5s}: {len(y):5d} seqs  " + "  ".join(f"{LABELS[k]}={c[k]}" for k in range(3)))

    ytr = out["train"]["y"]
    cnt = torch.bincount(ytr, minlength=3).float()
    cw = (cnt.sum() / (3 * cnt.clamp(min=1)))
    cw = cw / cw.mean()
    out.update({"class_weights": cw, "label_names": LABELS, "feature_names": keep_names,
                "norm": {"mean": torch.from_numpy(mean), "std": torch.from_numpy(std)},
                "seq_len": args.seq_len, "horizon": args.horizon, "node_order": NODE_ORDER,
                "split_runs": {"train": train_r, "val": val_r, "test": test_r}})
    print("class weights:", [round(x, 3) for x in cw.tolist()])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()