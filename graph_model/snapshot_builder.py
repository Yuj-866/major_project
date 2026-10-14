"""Snapshot builder: logs -> temporal graph windows.
Reads cloud_simulation/logs/requests.jsonl (shared volume), slices WINDOW_SEC,
builds 25-node graph, 13-feature vectors, LOW/MEDIUM/HIGH label.

=============================================================================
DYNAMIC VERSION v2 -- fixes applied after review of v1:
=============================================================================
  1. LABEL IS NOW GROUND-TRUTH, NOT TRAFFIC-RULE-DERIVED.
     simulator.py knows exactly what it's doing each tick (recon, an
     ongoing/failed/succeeded credential attack, which node it's moving
     into and how sensitive that node is, exfiltration) and writes that,
     with precise window start/end timestamps, to
     cloud_simulation/logs/attacker_windows.jsonl. This file reads that and
     matches each snapshot to the attacker-window record whose
     [start, end] interval overlaps it the most -- NOT by polling a single
     mutable run_state.json, which was racy and could lag by a window.
     The old traffic-counter-based label (failed logins / volume spikes /
     new edges / time_of_day) is kept as a SECONDARY field,
     `label_traffic_rule`, for comparison/debugging only. It is NOT what
     build_dataset.py should train on.
  2. Fixed failed-login double counting. A single login attempt produces
     TWO 401 log lines (api-server's own hook, plus auth-service's hook
     when api-server internally proxies to /verify) against two different
     `destination` values. Summing failed_401 across all destinations
     therefore counted every real attempt twice. total_failed_logins is now
     counted directly off endpoint == "/api/login" lines only.
  3. scheduled_phase is now mapped to the short names build_dataset.py's
     phase x label crosstab already expects (recon/brute/exploit/exfil/
     idle) via PHASE_COMPAT_MAP, so that table is no longer all zeros. The
     full descriptive state name is kept separately as `attacker_state`.
  4. A feature_names consistency check at startup warns (loudly) if
     pre-existing snapshot files in SNAP_DIR don't match the current
     13-feature schema, so 12-feature and 13-feature snapshots never get
     silently mixed into the same training run.
  5. time_of_day is still computed and stored (it's useful metadata / for
     the live dashboard) but MUST be excluded from the model's input
     features at train time -- see the reminder printed at startup and in
     build_dataset.py's --drop-features.

Run: python -m graph_model.snapshot_builder [--window-sec 30]
Output: graph_model/snapshots/window_{N}.json
"""
import argparse
import json
import time
from collections import Counter, deque
from datetime import datetime, timezone, timedelta
from pathlib import Path

import networkx as nx

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = BASE_DIR / "cloud_simulation" / "logs" / "requests.jsonl"
ATTACKER_WINDOWS_FILE = BASE_DIR / "cloud_simulation" / "logs" / "attacker_windows.jsonl"
SNAP_DIR = BASE_DIR / "graph_model" / "snapshots"
ASSETS_FILE = BASE_DIR / "asset_discovery" / "assets.json"

NODE_ORDER = [
    "api-server", "auth-service", "payment-service", "worker-service",
    "user-docs", "kyc-files", "audit-logs", "backups",
    "db-access-role", "lambda-exec-role", "readonly-role", "payment-role",
    "admin-role", "payment-lambda", "kyc-lambda", "notify-lambda",
    "fintech-api-gateway", "tx-queue", "notify-queue", "mysql-db",
    "redis-cache", "waf-firewall", "load-balancer", "cloudtrail-logs", "kms-keys",
]

BASE_EDGES = [
    ("waf-firewall", "load-balancer"), ("load-balancer", "fintech-api-gateway"),
    ("load-balancer", "api-server"), ("fintech-api-gateway", "api-server"),
    ("api-server", "auth-service"), ("api-server", "payment-service"),
    ("api-server", "mysql-db"), ("auth-service", "mysql-db"),
    ("payment-service", "mysql-db"), ("payment-service", "redis-cache"),
    ("api-server", "user-docs"), ("api-server", "audit-logs"),
    ("payment-service", "kyc-files"), ("payment-service", "backups"),
    ("api-server", "readonly-role"), ("auth-service", "db-access-role"),
    ("payment-service", "payment-role"), ("db-access-role", "admin-role"),
    ("payment-role", "admin-role"), ("tx-queue", "payment-lambda"),
    ("notify-queue", "notify-lambda"), ("kyc-files", "kyc-lambda"),
    ("lambda-exec-role", "payment-lambda"), ("lambda-exec-role", "kyc-lambda"),
    ("lambda-exec-role", "notify-lambda"), ("admin-role", "kms-keys"),
    ("admin-role", "cloudtrail-logs"), ("admin-role", "mysql-db"),
    ("worker-service", "tx-queue"), ("worker-service", "notify-queue"),
    ("api-server", "worker-service"),
]

EXPOSURE_MAP = {"public": 1.0, "internal": 0.5, "private": 0.0}
PRIVILEGE_MAP = {"limited": 0.25, "storage": 0.5, "compute": 0.6, "gateway": 0.7,
                 "api": 0.75, "auth": 0.8, "database": 0.9, "payment": 0.9,
                 "admin": 1.0, "infra": 0.5}
TYPE_MAP = {"DockerService": 1.0, "EC2": 2.0, "S3Bucket": 3.0, "IAMRole": 4.0,
            "Lambda": 5.0, "APIGateway": 6.0, "SQS": 7.0, "ElastiCache": 8.0,
            "WAF": 9.0, "ALB": 10.0, "CloudTrail": 11.0, "KMS": 12.0}

# Per-node auth strength (0 = trivially bypassable, 1 = well protected).
# Single source of truth, imported by simulator.py as well.
AUTH_STRENGTH_MAP = {
    "api-server": 0.40, "auth-service": 0.20, "payment-service": 0.15,
    "worker-service": 0.50, "user-docs": 0.10, "kyc-files": 0.10,
    "audit-logs": 0.20, "backups": 0.15, "db-access-role": 0.65,
    "lambda-exec-role": 0.60, "readonly-role": 0.70, "payment-role": 0.60,
    "admin-role": 0.30, "payment-lambda": 0.55, "kyc-lambda": 0.55,
    "notify-lambda": 0.60, "fintech-api-gateway": 0.65, "tx-queue": 0.60,
    "notify-queue": 0.60, "mysql-db": 0.55, "redis-cache": 0.45,
    "waf-firewall": 0.75, "load-balancer": 0.70, "cloudtrail-logs": 0.80,
    "kms-keys": 0.90,
}

FEATURE_NAMES = ["cvss", "exposure", "privilege", "patch_status", "auth_strength",
                 "asset_type", "pagerank", "betweenness", "degree_centrality",
                 "in_degree", "out_degree", "anomaly_score", "time_of_day"]

# NOTE: anomaly_score AND time_of_day are both derived from / correlated
# with the same signals that drive the label. build_dataset.py must drop
# BOTH at train time:
#   python build_dataset.py --drop-features anomaly_score time_of_day
TRAIN_TIME_FEATURES_TO_DROP = ["anomaly_score", "time_of_day"]

# Maps the simulator's full state names to the short names
# build_dataset.py's phase x label crosstab already expects.
PHASE_COMPAT_MAP = {
    "idle": "idle",
    "reconnaissance": "recon",
    "credential_attack": "brute",
    "lateral_movement": "exploit",
    "exfiltration": "exfil",
    None: "idle",
}

ROLLING_WINDOW = 20
MIN_HISTORY_FOR_RATIO_RULES = 5


def load_assets():
    with open(ASSETS_FILE) as f:
        assets = json.load(f)
    lookup = {}
    for a in assets:
        lookup[a["name"]] = a  # last wins (Docker entries)
    return lookup


def read_window(window_sec):
    """Returns (events, window_start, window_end) as timezone-aware UTC
    datetimes, so callers can match this interval against other
    timestamped logs (e.g. attacker_windows.jsonl)."""
    window_end = datetime.now(timezone.utc)
    window_start = window_end - timedelta(seconds=window_sec)
    if not LOG_FILE.exists():
        return [], window_start, window_end
    out = []
    with open(LOG_FILE) as f:
        for line in f:
            try:
                e = json.loads(line)
                ts = datetime.fromisoformat(e["timestamp"])
                if ts >= window_start:
                    out.append(e)
            except Exception:
                continue
    return out, window_start, window_end


def _load_attacker_window_records(max_lines=4000):
    """Tail the attacker_windows.jsonl file (bounded, cheap even for a
    multi-hour run) and parse each line's start/end as datetimes."""
    if not ATTACKER_WINDOWS_FILE.exists():
        return []
    records = []
    with open(ATTACKER_WINDOWS_FILE) as f:
        lines = f.readlines()[-max_lines:]
    for line in lines:
        try:
            r = json.loads(line)
            r["_start"] = datetime.fromisoformat(r["window_start"])
            r["_end"] = datetime.fromisoformat(r["window_end"])
            records.append(r)
        except Exception:
            continue
    return records


def match_attacker_window(window_start, window_end, records):
    """Find the attacker_windows.jsonl record with the greatest time
    overlap with [window_start, window_end]. Returns None if nothing
    overlaps at all (simulator not running yet / stale file / startup
    race) so the caller can fall back cleanly.

    Accepts records either pre-parsed by _load_attacker_window_records
    (which adds "_start"/"_end" datetime objects) or raw dicts with
    "window_start"/"window_end" ISO strings -- parses on the fly for the
    latter so callers don't have to remember which form to pass."""
    best, best_overlap = None, timedelta(0)
    for r in records:
        start = r.get("_start")
        end = r.get("_end")
        if start is None or end is None:
            try:
                start = datetime.fromisoformat(r["window_start"])
                end = datetime.fromisoformat(r["window_end"])
            except Exception:
                continue
        overlap_start = max(window_start, start)
        overlap_end = min(window_end, end)
        overlap = overlap_end - overlap_start
        if overlap > best_overlap:
            best, best_overlap = r, overlap
    return best


def compute_time_of_day():
    """Fraction of the day elapsed, local wall-clock time (0.0-1.0).
    Stored for reference/dashboard use -- MUST be excluded from model
    inputs at train time (see TRAIN_TIME_FEATURES_TO_DROP)."""
    now = datetime.now()
    return round((now.hour + now.minute / 60.0) / 24.0, 4)


def _traffic_rule_label(total_failed_logins, new_dynamic_edge, has_dynamic_edge,
                         volume_ratio, outside_business_hours, rolling_avg,
                         max_fail, total, has_exploit):
    """Secondary, non-training label kept for comparison with the old
    Stage-4 traffic-counter rule. NOT used by build_dataset.py.
    (Loosened slightly from v1: the failed-logins HIGH clause now checks
    "a dynamic attack edge is currently active" rather than "appeared
    this exact window", so a sustained attack doesn't drop to LOW on its
    second window the way it did before.)"""
    if rolling_avg is None:
        label = "HIGH" if (max_fail > 20 or total > 200 or has_exploit) else \
                ("MEDIUM" if total > 80 else "LOW")
        return label

    high = (
        (total_failed_logins > 20 and has_dynamic_edge)
        or (total_failed_logins < 3 and volume_ratio is not None
            and volume_ratio > 3.0 and outside_business_hours)
    )
    medium = (
        (5 <= total_failed_logins <= 20)
        or (new_dynamic_edge and not (total_failed_logins > 20))
        or (volume_ratio is not None and volume_ratio >= 2.0)
    )
    if high:
        return "HIGH"
    if medium:
        return "MEDIUM"
    return "LOW"


def build_snapshot(events, window_id, window_start, window_end, asset_lookup,
                    attacker_records, rolling_volumes, prev_dynamic_edges):
    
    pair_counts = Counter(
        (e["source"], e["destination"]) for e in events
        if e.get("source") and e.get("destination")
        and e["source"] != e["destination"]  # guard against self-loops
    )

    failed_401 = Counter(e["destination"] for e in events if int(e.get("status", 200)) == 401)
    hits = Counter(e["destination"] for e in events)
    hits.update(e["source"] for e in events)

    # Fix: count real failed LOGIN ATTEMPTS, not every 401 log line a
    # single attempt happens to fan out into across services.
    total_failed_logins = sum(
        1 for e in events
        if e.get("endpoint") == "/api/login" and int(e.get("status", 200)) == 401
    )

    G = nx.DiGraph()
    for n in NODE_ORDER:
        G.add_node(n)
    base_set = set(BASE_EDGES)
    for s, t in BASE_EDGES:
        G.add_edge(s, t, weight=float(pair_counts.get((s, t), 1)), type="base")
    for (s, t), c in pair_counts.items():
        if (s, t) not in base_set:
            if s not in G:
                G.add_node(s)
            if t not in G:
                G.add_node(t)
            G.add_edge(s, t, weight=float(c), type="dynamic_attack")
        else:
            G[s][t]["weight"] = float(c if c else 1)
            if c and ((s, t) == ("auth-service", "admin-role") or c > 30 or failed_401.get(t, 0) > 5):
                G[s][t]["type"] = "dynamic_attack"

    pr = nx.pagerank(G, weight="weight")
    bw = nx.betweenness_centrality(G, weight="weight")
    dc = nx.degree_centrality(G)

    time_of_day = compute_time_of_day()

    matrix = []
    for n in NODE_ORDER:
        a = asset_lookup.get(n, {})
        sec = [float(a.get("cvss", 0)) / 10.0,
               EXPOSURE_MAP.get(a.get("exposure", "private"), 0.0),
               PRIVILEGE_MAP.get(a.get("privilege", "limited"), 0.0),
               0.0,
               AUTH_STRENGTH_MAP.get(n, 0.5),
               TYPE_MAP.get(a.get("type", ""), 0.0)]
        fl = float(failed_401.get(n, 0))
        hc = float(hits.get(n, 0))
        anomaly = round(min(max(fl / 20.0, hc / 200.0), 1.0), 4)
        graph = [round(pr.get(n, 0.0), 6), round(bw.get(n, 0.0), 6),
                 round(dc.get(n, 0.0), 6), float(G.in_degree(n)), float(G.out_degree(n))]
        matrix.append(sec + graph + [anomaly, time_of_day])

    total = len(events)
    max_fail = max(failed_401.values()) if failed_401 else 0

    current_dynamic_edges = {(s, t) for s, t, d in G.edges(data=True) if d["type"] == "dynamic_attack"}
    new_dynamic_edge = bool(current_dynamic_edges - prev_dynamic_edges)
    has_dynamic_edge = bool(current_dynamic_edges)

    if len(rolling_volumes) >= MIN_HISTORY_FOR_RATIO_RULES:
        rolling_avg = sum(rolling_volumes) / len(rolling_volumes)
    else:
        rolling_avg = None
    volume_ratio = (total / rolling_avg) if (rolling_avg and rolling_avg > 1e-6) else None
    outside_business_hours = time_of_day < 0.375 or time_of_day > 0.75
    has_exploit = ("auth-service", "admin-role") in pair_counts

    label_traffic_rule = _traffic_rule_label(
        total_failed_logins, new_dynamic_edge, has_dynamic_edge, volume_ratio,
        outside_business_hours, rolling_avg, max_fail, total, has_exploit,
    )

    # ---- ground-truth label: match this window to the simulator's own
    # record of what it was actually doing during this exact interval. ----
    attacker_rec = match_attacker_window(window_start, window_end, attacker_records)
    if attacker_rec is not None:
        label = attacker_rec.get("ground_truth_label", label_traffic_rule)
        attacker_state = attacker_rec.get("phase", "idle")
        run_id = attacker_rec.get("run_id", "unknown")
        compromised_node = attacker_rec.get("compromised_node")
        label_source = "ground_truth"
    else:
        # Simulator not running yet / no overlapping record found -- fall
        # back to the traffic-rule label so the pipeline never crashes,
        # but this is clearly flagged so you can filter these out later.
        label = label_traffic_rule
        attacker_state = "idle"
        run_id = "unknown"
        compromised_node = None
        label_source = "traffic_rule_fallback"

    scheduled_phase = PHASE_COMPAT_MAP.get(attacker_state, "idle")

    rolling_volumes.append(total)
    prev_dynamic_edges.clear()
    prev_dynamic_edges.update(current_dynamic_edges)

    edges = [{"source": s, "target": t, "weight": d["weight"], "type": d["type"]} for s, t, d in G.edges(data=True)]
    return {
        "window": window_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "label": label,
        "label_source": label_source,
        "label_traffic_rule": label_traffic_rule,
        "event_count": total,
        "max_failed_login": max_fail,
        "total_failed_logins": total_failed_logins,
        "new_dynamic_edge": new_dynamic_edge,
        "has_dynamic_edge": has_dynamic_edge,
        "rolling_avg_volume": round(rolling_avg, 2) if rolling_avg is not None else None,
        "time_of_day": time_of_day,
        "run_id": run_id,
        "scheduled_phase": scheduled_phase,
        "attacker_state": attacker_state,
        "compromised_node": compromised_node,
        "nodes": NODE_ORDER, "feature_names": FEATURE_NAMES,
        "feature_matrix": matrix, "edges": edges,
    }


def _warm_start_history(existing_files):
    rolling_volumes = deque(maxlen=ROLLING_WINDOW)
    prev_dynamic_edges = set()
    if not existing_files:
        return rolling_volumes, prev_dynamic_edges

    tail = sorted(existing_files, key=lambda p: int(p.stem.split("_")[1]))[-ROLLING_WINDOW:]
    last_snap = None
    for f in tail:
        try:
            with open(f) as fh:
                s = json.load(fh)
            rolling_volumes.append(int(s.get("event_count", 0)))
            last_snap = s
        except Exception:
            continue

    if last_snap is not None:
        for e in last_snap.get("edges", []):
            if e.get("type") == "dynamic_attack":
                prev_dynamic_edges.add((e["source"], e["target"]))

    return rolling_volumes, prev_dynamic_edges


def _check_feature_consistency(existing_files):
    """Loudly warn (don't silently proceed) if snapshots already in
    SNAP_DIR were built with a different feature schema -- e.g. the old
    12-feature static-simulator files."""
    if not existing_files:
        return
    sample = sorted(existing_files, key=lambda p: int(p.stem.split("_")[1]))[-1]
    try:
        with open(sample) as f:
            s = json.load(f)
        existing_names = s.get("feature_names")
    except Exception:
        return
    if existing_names != FEATURE_NAMES:
        print("=" * 70)
        print("[WARNING] Existing snapshots in", SNAP_DIR, "use a DIFFERENT")
        print("  feature schema than this script:")
        print("    existing:", existing_names)
        print("    current :", FEATURE_NAMES)
        print("  Mixing these into one training run will make build_dataset.py")
        print("  reject or misalign them. Move the old folder aside, e.g.:")
        print("    mv graph_model/snapshots graph_model/snapshots_old")
        print("=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-sec", type=int, default=30)
    args = ap.parse_args()
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    lookup = load_assets()

    existing = list(SNAP_DIR.glob("window_*.json"))
    _check_feature_consistency(existing)
    window_id = len(existing)
    rolling_volumes, prev_dynamic_edges = _warm_start_history(existing)

    print(f"Snapshot builder: log={LOG_FILE}, window_sec={args.window_sec}, start={window_id}")
    print(f"  warm-started rolling history: {len(rolling_volumes)} windows, "
          f"{len(prev_dynamic_edges)} prior dynamic edges")
    print(f"  REMINDER: at train time, run build_dataset.py with "
          f"--drop-features {' '.join(TRAIN_TIME_FEATURES_TO_DROP)}")
    print(f"  Labels are ground-truth from {ATTACKER_WINDOWS_FILE.name} when available; "
          f"label_traffic_rule is kept only as a secondary/debug field.")

    while True:
        events, window_start, window_end = read_window(args.window_sec)
        attacker_records = _load_attacker_window_records()
        snap = build_snapshot(events, window_id, window_start, window_end, lookup,
                               attacker_records, rolling_volumes, prev_dynamic_edges)
        with open(SNAP_DIR / f"window_{window_id}.json", "w") as f:
            json.dump(snap, f, indent=2)
        print(f"window_{window_id}.json: run={snap['run_id']} state={snap['attacker_state']} "
              f"label={snap['label']} ({snap['label_source']}) "
              f"traffic_rule={snap['label_traffic_rule']} "
              f"events={snap['event_count']} failed_logins={snap['total_failed_logins']} "
              f"compromised={snap['compromised_node']}")
        window_id += 1
        time.sleep(args.window_sec)


if __name__ == "__main__":
    main()