"""run_baselines.py -- baselines on the SAME run-level split / train set as T-GCN+GAT.

    python -m graph_model.run_baselines --data graph_model\\ds_h1_nocent.pt --tag h1_nocent ^
        --gnn-results h1_nocent_full h1_nocent_nograph

Rows: traffic-rule heuristic, persistence (oracle, only when horizon>0), RF, XGBoost
(last window / whole sequence, flattened), static GCN, plus any trained T-GCN+GAT
results named via --gnn-results (tags used with train_tgcn_gat --tag).
"""
import argparse
import json

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

try:
    from graph_model.train_tgcn_gat import (DEFAULT_DATA, RESULTS_DIR, StaticGCN,
                                            fit_and_eval, load_dataset, metrics)
except ImportError:
    from train_tgcn_gat import (DEFAULT_DATA, RESULTS_DIR, StaticGCN,
                                fit_and_eval, load_dataset, metrics)

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
    print("[WARN] xgboost not installed -> skipping XGBoost (pip install xgboost)")


def tab(split, mode):
    X = split["X"].cpu().numpy()
    return X[:, -1].reshape(len(X), -1) if mode == "last" else X.reshape(len(X), -1)


def agg(rows):
    f1 = np.array([r["macro_f1"] for r in rows])
    hr = np.array([r["high_recall"] for r in rows])
    pc = np.array([r["f1_per_class"] for r in rows]).mean(0)
    return {"macro_f1_mean": float(f1.mean()), "macro_f1_std": float(f1.std()),
            "high_recall_mean": float(hr.mean()), "f1_per_class_mean": pc.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--tag", default="default", help="name used for results/baselines_<tag>.json")
    ap.add_argument("--gnn-results", nargs="*", default=["scratch"],
                    help="tags of train_tgcn_gat runs to include in the table")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_dataset(args.data, device)
    tr, te = data["train"], data["test"]
    ytr, yte = tr["y"].cpu().numpy(), te["y"].cpu().numpy()
    in_dim = tr["X"].shape[-1]
    horizon = data.get("horizon", 0)
    print(f"dataset: horizon={horizon} seq_len={data['seq_len']} features={data['feature_names']}")
    out = {}

    # reference rows (not trained)
    out["Traffic-rule heuristic (last window)"] = agg([metrics(yte, te["rule_label"].cpu().numpy())])
    if horizon > 0 and "cur_y" in te:
        out["Persistence (ORACLE: true current label)"] = agg([metrics(yte, te["cur_y"].cpu().numpy())])

    # tabular models
    sw = compute_sample_weight("balanced", ytr)
    for mode in ("last", "seq"):
        Xtr, Xte = tab(tr, mode), tab(te, mode)
        rf_rows, xgb_rows = [], []
        for sd in args.seeds:
            rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                        random_state=sd, n_jobs=-1).fit(Xtr, ytr)
            rf_rows.append(metrics(yte, rf.predict(Xte)))
            if XGBClassifier is not None:
                xg = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                                   subsample=0.9, colsample_bytree=0.8, random_state=sd,
                                   eval_metric="mlogloss", n_jobs=-1)
                xg.fit(Xtr, ytr, sample_weight=sw)
                xgb_rows.append(metrics(yte, xg.predict(Xte)))
        out[f"Random Forest ({mode})"] = agg(rf_rows)
        if xgb_rows:
            out[f"XGBoost ({mode})"] = agg(xgb_rows)
        print(f"done tabular: {mode}")

    # static GCN (single window, no time, no attention)
    rows = []
    for sd in args.seeds:
        _, r = fit_and_eval(lambda: StaticGCN(in_dim), data, sd, device, args.epochs, verbose=False)
        rows.append(r["test"])
        print(f"static GCN seed {sd}: test macroF1={r['test']['macro_f1']:.3f}")
    out["Static GCN (last window)"] = agg(rows)

    # T-GCN+GAT runs from train_tgcn_gat.py
    for t in args.gnn_results:
        p = RESULTS_DIR / f"tgcn_gat_{t}.json"
        if p.exists():
            s = json.load(open(p))["summary"]
            out[f"T-GCN+GAT [{t}]"] = {k: s[k] for k in
                                       ("macro_f1_mean", "macro_f1_std", "high_recall_mean", "f1_per_class_mean")}
        else:
            print(f"[INFO] {p.name} not found -> run train_tgcn_gat with --tag {t} first")

    print(f"\n{'Model':44s} {'test macro-F1':>16s} {'HIGH recall':>12s}   F1 L/M/H")
    for name, a in out.items():
        print(f"{name:44s} {a['macro_f1_mean']:7.3f} +/- {a['macro_f1_std']:.3f} "
              f"{a['high_recall_mean']:12.3f}   {[round(x, 2) for x in a['f1_per_class_mean']]}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / f"baselines_{args.tag}.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()