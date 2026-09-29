"""Preprocess PaySim -> same window JSON format as snapshot_builder.
Input: ml_model/paysim.csv (Kaggle PaySim1, ~470MB). Use --sample 100000 for demo.
Groups rows into 6h windows via `step`, labels HIGH if fraud else LOW.
Output: ml_model/paysim_windows/window_paysim_{i}.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "paysim_windows"
FEATURE_NAMES = ["cvss", "exposure", "privilege", "patch_status", "auth_strength",
                 "asset_type", "pagerank", "betweenness", "degree_centrality",
                 "in_degree", "out_degree", "anomaly_score"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(BASE / "paysim.csv"))
    ap.add_argument("--sample", type=int, default=100000)
    ap.add_argument("--window-steps", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import os
    if not os.path.exists(args.csv):
        print(f"CSV not found: {args.csv}")
        print("Generating 120 synthetic PaySim-style windows instead (demo mode).")
        print("To use real data: kaggle datasets download -d ealaxi/paysim1 -p ml_model/ --unzip")
        print("  then re-run with --csv ml_model/PS_20174392719_1491204439457_log.csv")
        import random
        for i in range(120):
            fraud = 1 if i % 6 == 5 else 0
            total = random.randint(800, 5000)
            label = "HIGH" if fraud else "LOW"
            nodes = [f"ACC-{random.randint(1000,9999)}" for _ in range(25)]
            matrix = []
            for k in range(25):
                cnt = random.randint(5, 80)
                anomaly = round(min(cnt / 50.0 + (1.0 if fraud and k < 3 else 0.0), 1.0), 4)
                matrix.append([0.5, 0.5, 0.5, 0.0, 0.0, 3.0, 0.04, 0.0, 0.1,
                               float(cnt), float(cnt), anomaly])
            edges = [{"source": nodes[0], "target": nodes[1], "weight": 100.0, "type": "base"}]
            with open(OUT / f"window_paysim_{i}.json", "w") as f:
                json.dump({"window": f"paysim_{i}", "label": label, "event_count": total,
                           "max_failed_login": 0, "nodes": nodes,
                           "feature_names": FEATURE_NAMES, "feature_matrix": matrix,
                           "edges": edges}, f)
        print(f"Done. 120 synthetic files in {OUT}")
        return
    print(f"Reading {args.csv} (sample={args.sample})...")
    df = pd.read_csv(args.csv, nrows=args.sample if args.sample else None,
                     usecols=lambda c: c in {"step", "nameOrig", "nameDest", "amount", "isFraud", "type"})
    df["window"] = df["step"] // args.window_steps
    for i, (w, g) in enumerate(df.groupby("window")):
        fraud = int((g["isFraud"] == 1).sum())
        total = len(g)
        label = "HIGH" if fraud > 0 else "LOW"
        # Aggregate per-account as pseudo-nodes (top 25 by volume to match 25-node shape)
        top = list(g["nameDest"].value_counts().head(25).index)
        nodes = top + [f"pad-{k}" for k in range(25 - len(top))]
        matrix = []
        for n in nodes[:25]:
            sub = g[g["nameDest"] == n] if not n.startswith("pad-") else None
            amt = float(sub["amount"].sum()) if sub is not None and len(sub) else 0.0
            cnt = len(sub) if sub is not None else 0
            anomaly = round(min(cnt / 50.0 + (1.0 if fraud and n in top[:3] else 0.0), 1.0), 4)
            matrix.append([0.5, 0.5, 0.5, 0.0, 0.0, 3.0, 0.04, 0.0, 0.1,
                           float(cnt), float(cnt), anomaly])
        edges = [{"source": r["nameOrig"][:12], "target": r["nameDest"][:12],
                  "weight": float(r["amount"]), "type": "base"}
                 for _, r in g.head(100).iterrows()]
        with open(OUT / f"window_paysim_{i}.json", "w") as f:
            json.dump({"window": f"paysim_{i}", "label": label, "event_count": total,
                       "max_failed_login": 0, "nodes": nodes[:25],
                       "feature_names": FEATURE_NAMES, "feature_matrix": matrix,
                       "edges": edges}, f)
        print(f"window_paysim_{i}.json: label={label} events={total} fraud={fraud}")
    print(f"Done. Files in {OUT}")


if __name__ == "__main__":
    main()
