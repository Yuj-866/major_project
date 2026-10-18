"""build_node_dataset.py -- same pipeline as build_dataset.py, plus node-level labels.

    python -m graph_model.build_node_dataset --horizon 0 --seq-len 3 --fix-run-boundary --out graph_model\\ds_node_h0.pt

Extra keys per split (on top of build_dataset.py's):
    comp  [S]  index (0..24) of the attacker's compromised node in the TARGET window, -1 if none
    phase [S]  scheduled_phase string of the TARGET window (idle/recon/brute/exploit/exfil)

Uses all 14 features (static + centralities + in/out_strength + new_conn) by default.
Imports helpers from build_dataset.py (read-only), so nothing existing is modified.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from graph_model.build_dataset import (EDGE_FEATURE_NAMES, L2I, LABELS, NODE_IDX,  # noqa: E402
                                       boundary_candidates, chrono_split, frame_arrays,
                                       load_windows, make_sequences, split_runs)
from graph_model.snapshot_builder import FEATURE_NAMES, NODE_ORDER  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap-dirs", nargs="+",
                    default=[str(BASE_DIR / "graph_model" / "snapshots" / "snapshots_dynamic")])
    ap.add_argument("--out", default=str(BASE_DIR / "graph_model" / "ds_node_h0.pt"))
    ap.add_argument("--seq-len", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=0, help="0 = current foothold, 1 = next window")
    ap.add_argument("--drop-features", nargs="*", default=["anomaly_score", "time_of_day"])
    ap.add_argument("--aug-copies", type=int, default=3)
    ap.add_argument("--min-per-class", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-mode", choices=["shuffle", "chrono"], default="shuffle")
    ap.add_argument("--fix-run-boundary", action="store_true")
    args = ap.parse_args()

    rows, dropped = load_windows(args.snap_dirs, 1)
    print(f"Loaded {len(rows)} usable windows; dropped: {dict(dropped)}")
    cands, _ = boundary_candidates(rows)
    if args.fix_run_boundary:
        for i in cands:
            rows[i]["run_id"] = rows[i - 1]["run_id"]
        print(f"run-boundary fix: moved {len(cands)} windows back to their own run")

    keep_names = [f for f in FEATURE_NAMES if f not in args.drop_features]
    keep_idx = [FEATURE_NAMES.index(f) for f in keep_names]
    Xw, Ew = zip(*[frame_arrays(s, keep_idx) for s in rows])
    Xw, Ew = np.stack(Xw), np.stack(Ew)
    E0, E1 = Ew[:, 0], Ew[:, 1]
    ef = np.stack([E0.sum(1), E0.sum(2), E1.sum(1) + E1.sum(2)], axis=-1).astype(np.float32)
    Xw = np.concatenate([Xw, ef], axis=-1)
    keep_names = keep_names + EDGE_FEATURE_NAMES
    print(f"Features ({len(keep_names)}): {keep_names}")

    yw = np.array([L2I[s["label"]] for s in rows])
    rule_w = np.array([L2I.get(s.get("label_traffic_rule"), 0) for s in rows])
    comp_w = np.array([NODE_IDX.get(s.get("compromised_node"), -1) for s in rows])
    phase_w = np.array([s.get("scheduled_phase", "idle") for s in rows])
    print(f"windows with a known compromised node: {(comp_w >= 0).sum()} of {len(rows)}")

    idx, tgt, seq_run, runs = make_sequences(rows, args.seq_len, args.horizon)
    if args.split_mode == "chrono":
        train_r, val_r, test_r = chrono_split(runs, rows, args.min_per_class)
    else:
        train_r, val_r, test_r = split_runs(runs, rows, args.seed, min_per_class=args.min_per_class)
    which = {r: "train" for r in train_r}
    which.update({r: "val" for r in val_r})
    which.update({r: "test" for r in test_r})
    print(f"{len(runs)} runs, {len(idx)} sequences | split={args.split_mode} "
          f"train/val/test runs = {len(train_r)}/{len(val_r)}/{len(test_r)}")

    tr_w = sorted({w for sq, r in zip(idx, seq_run) if which[r] == "train" for w in sq})
    flat = Xw[tr_w].reshape(-1, Xw.shape[-1])
    mean, std = flat.mean(0), flat.std(0)
    std = np.where(std < 1e-6, 1.0, std)
    Xn = (Xw - mean) / std
    dyn_mask = (Xw[tr_w].std(axis=0).mean(0) > 1e-6).astype(np.float32)

    g = torch.Generator().manual_seed(args.seed)
    out = {}
    for split in ("train", "val", "test"):
        sel = [i for i, r in enumerate(seq_run) if which[r] == split]
        ii = idx[sel]
        X = torch.from_numpy(Xn[ii]).float()
        E = torch.from_numpy(Ew[ii]).float()
        y = torch.from_numpy(yw[tgt[sel]]).long()
        cy = torch.from_numpy(yw[ii[:, -1]]).long()
        rl = torch.from_numpy(rule_w[ii[:, -1]]).long()
        cp = torch.from_numpy(comp_w[tgt[sel]]).long()
        ph = list(phase_w[tgt[sel]])
        runs_s = [seq_run[i] for i in sel]
        if split == "train" and args.aug_copies > 1:
            Xs, Es, ys, cs, rs, cps, phs, rr = [X], [E], [y], [cy], [rl], [cp], [ph], [runs_s]
            dm = torch.from_numpy(dyn_mask)
            for _ in range(args.aug_copies - 1):
                Xs.append(X + 0.05 * torch.randn(X.shape, generator=g) * dm)
                jit = torch.exp(0.1 * torch.randn(E[:, :, 0].shape, generator=g))
                Ec = E.clone()
                Ec[:, :, 0] = Ec[:, :, 0] * jit
                Es.append(Ec)
                ys.append(y); cs.append(cy); rs.append(rl); cps.append(cp); phs.append(ph); rr.append(runs_s)
            X, E, y, cy = torch.cat(Xs), torch.cat(Es), torch.cat(ys), torch.cat(cs)
            rl, cp = torch.cat(rs), torch.cat(cps)
            ph = [p for part in phs for p in part]
            runs_s = [r for part in rr for r in part]
        out[split] = {"X": X, "E": E, "y": y, "cur_y": cy, "rule_label": rl,
                      "run_id": runs_s, "comp": cp, "phase": ph}
        c = Counter(y.tolist())
        n_loc = int((cp >= 0).sum())
        print(f"{split:5s}: {len(y):5d} seqs  " + "  ".join(f"{LABELS[k]}={c[k]}" for k in range(3))
              + f"  | with compromised node: {n_loc}")

    cnt = torch.bincount(out["train"]["y"], minlength=3).float()
    cw = cnt.sum() / (3 * cnt.clamp(min=1))
    cw = cw / cw.mean()
    out.update({"class_weights": cw, "label_names": LABELS, "feature_names": keep_names,
                "norm": {"mean": torch.from_numpy(mean), "std": torch.from_numpy(std)},
                "seq_len": args.seq_len, "horizon": args.horizon, "node_order": NODE_ORDER,
                "split_runs": {"train": train_r, "val": val_r, "test": test_r}})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()