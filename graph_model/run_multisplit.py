"""run_multisplit.py -- evaluate every model over several run-level splits.

For each split seed it rebuilds the dataset (build_dataset.py --seed s), trains
T-GCN+GAT (full + no-graph), static GCN, RF, XGBoost, then reports mean +/- std
across splits and the PAIRED difference of every model vs the full T-GCN+GAT
(mean difference with a 95% confidence interval).

    python -m graph_model.run_multisplit --horizon 1 --feature-set allfeat --splits 10 --seeds 3 --tag h1_allfeat

Results are saved after every split to graph_model/results/multisplit_<tag>.json
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

try:
    from graph_model.train_tgcn_gat import (RESULTS_DIR, NoGraphTGCNGAT, StaticGCN, TGCNGAT,
                                            fit_and_eval, load_dataset, metrics)
except ImportError:
    from train_tgcn_gat import (RESULTS_DIR, NoGraphTGCNGAT, StaticGCN, TGCNGAT,
                                fit_and_eval, load_dataset, metrics)

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
    print("[WARN] xgboost not installed -> skipping XGBoost")

BASE_DIR = Path(__file__).resolve().parent.parent
CENT = ["pagerank", "betweenness", "degree_centrality", "in_degree", "out_degree"]
FULL = "T-GCN+GAT (full)"


def tab(split, mode):
    X = split["X"].cpu().numpy()
    return X[:, -1].reshape(len(X), -1) if mode == "last" else X.reshape(len(X), -1)


def mean_over_seeds(rows):
    return {"macro_f1": float(np.mean([r["macro_f1"] for r in rows])),
            "high_recall": float(np.mean([r["high_recall"] for r in rows]))}


def evaluate_split(data, seeds, device, epochs):
    out = {}
    tr, te = data["train"], data["test"]
    ytr, yte = tr["y"].cpu().numpy(), te["y"].cpu().numpy()
    cfg = dict(in_dim=tr["X"].shape[-1], hidden=32, heads=4, edge_dim=2, n_classes=3, dropout=0.5)

    out["Traffic-rule heuristic"] = mean_over_seeds([metrics(yte, te["rule_label"].cpu().numpy())])
    if data.get("horizon", 0) > 0:
        out["Persistence (ORACLE)"] = mean_over_seeds([metrics(yte, te["cur_y"].cpu().numpy())])

    sw = compute_sample_weight("balanced", ytr)
    for mode in ("last", "seq"):
        Xtr, Xte = tab(tr, mode), tab(te, mode)
        rf_rows, xgb_rows = [], []
        for sd in seeds:
            rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                        random_state=sd, n_jobs=-1).fit(Xtr, ytr)
            rf_rows.append(metrics(yte, rf.predict(Xte)))
            if XGBClassifier is not None:
                xg = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.9,
                                   colsample_bytree=0.8, random_state=sd, eval_metric="mlogloss", n_jobs=-1)
                xg.fit(Xtr, ytr, sample_weight=sw)
                xgb_rows.append(metrics(yte, xg.predict(Xte)))
        out[f"Random Forest ({mode})"] = mean_over_seeds(rf_rows)
        if xgb_rows:
            out[f"XGBoost ({mode})"] = mean_over_seeds(xgb_rows)

    for name, ctor in [("Static GCN (last window)", lambda: StaticGCN(cfg["in_dim"], 32, 3, 0.5)),
                       ("T-GCN+GAT (no graph)", lambda: NoGraphTGCNGAT(**cfg)),
                       (FULL, lambda: TGCNGAT(**cfg))]:
        rows = []
        for sd in seeds:
            _, r = fit_and_eval(ctor, data, sd, device, epochs, verbose=False)
            rows.append(r["test"])
        out[name] = mean_over_seeds(rows)
    return out


def summarize(per_split):
    names = list(per_split[0].keys())
    summ = {}
    full_f1 = np.array([s[FULL]["macro_f1"] for s in per_split])
    for n in names:
        f1 = np.array([s[n]["macro_f1"] for s in per_split])
        hr = np.array([s[n]["high_recall"] for s in per_split])
        d = f1 - full_f1
        row = {"macro_f1_mean": float(f1.mean()), "macro_f1_std": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
               "high_recall_mean": float(hr.mean()), "n_splits": len(f1)}
        if n != FULL and len(d) > 1:
            se = d.std(ddof=1) / np.sqrt(len(d))
            ci = stats.t.ppf(0.975, len(d) - 1) * se
            row.update({"diff_vs_full_mean": float(d.mean()), "diff_ci95": float(ci),
                        "splits_better_than_full": int((d > 0).sum())})
        summ[n] = row
    return summ


def print_table(summ):
    print(f"\n{'Model':30s} {'macro-F1 (mean +/- sd)':>24s} {'HIGH rec':>9s}   diff vs full [95% CI]   wins")
    for n, r in sorted(summ.items(), key=lambda kv: -kv[1]["macro_f1_mean"]):
        diff = ""
        if "diff_vs_full_mean" in r:
            m, c = r["diff_vs_full_mean"], r["diff_ci95"]
            diff = f"{m:+.3f} [{m - c:+.3f}, {m + c:+.3f}]   {r['splits_better_than_full']}/{r['n_splits']}"
        print(f"{n:30s} {r['macro_f1_mean']:8.3f} +/- {r['macro_f1_std']:.3f}      "
              f"{r['high_recall_mean']:7.3f}   {diff}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap-dirs", nargs="+", default=[str(BASE_DIR / "graph_model" / "snapshots" / "snapshots_dynamic")])
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--feature-set", choices=["nocent", "allfeat"], default="allfeat")
    ap.add_argument("--seq-len", type=int, default=3)
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3, help="model seeds per split")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--tag", default="multisplit")
    ap.add_argument("--fix-run-boundary", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    drop = ["anomaly_score", "time_of_day"] + (CENT if args.feature_set == "nocent" else [])
    seeds = list(range(args.seeds))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / f"multisplit_{args.tag}.json"
    tmp = BASE_DIR / "graph_model" / f"_tmp_ds_{args.tag}.pt"

    per_split, used = [], []
    t0 = time.time()
    for s in range(args.splits):
        cmd = [sys.executable, "-m", "graph_model.build_dataset", "--snap-dirs", *args.snap_dirs,
               "--seq-len", str(args.seq_len), "--horizon", str(args.horizon), "--edge-features",
               "--drop-features", *drop, "--seed", str(s), "--out", str(tmp)]
        if args.fix_run_boundary:
            cmd.append("--fix-run-boundary")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=BASE_DIR)
        except subprocess.CalledProcessError as e:
            print(f"[split {s}] build failed, skipping: {e.stderr.strip().splitlines()[-1] if e.stderr else e}")
            continue
        data = load_dataset(tmp, device)
        per_split.append(evaluate_split(data, seeds, device, args.epochs))
        used.append(s)
        n_high = int((data["test"]["y"] == 2).sum())
        print(f"[split {s}] done in {(time.time() - t0) / 60:.1f} min total | test HIGH windows={n_high} | "
              f"full={per_split[-1][FULL]['macro_f1']:.3f} nograph={per_split[-1]['T-GCN+GAT (no graph)']['macro_f1']:.3f}")
        with open(out_json, "w") as f:
            json.dump({"args": vars(args), "split_seeds": used, "per_split": per_split,
                       "summary": summarize(per_split) if len(per_split) > 1 else {}}, f, indent=2)

    if tmp.exists():
        tmp.unlink()
    if len(per_split) < 2:
        raise SystemExit("Need at least 2 successful splits to summarize.")
    summ = summarize(per_split)
    print(f"\n=== {args.tag}: horizon={args.horizon} features={args.feature_set} "
          f"splits={len(per_split)} model-seeds={len(seeds)} ===")
    print_table(summ)
    print(f"\nsaved -> {out_json}")


if __name__ == "__main__":
    main()