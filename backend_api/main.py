"""Risk-scoring API (contract v0.1). Mock model: swap predict_window() for the real one later.
Run from the project root:  uvicorn backend.main:app --reload --port 8000
"""
import asyncio, json, random, sys
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

HERE = Path(__file__).parent
CFG = json.loads((HERE / "config.json").read_text())

# ---- real model (graph_model/inference.py) with mock fallback ----
BASE = HERE.parent
sys.path.insert(0, str(BASE))
PREDICTOR, LOAD_ERROR = None, None
try:
    from graph_model.inference import RiskPredictor, BadInput
    _ckpt = CFG.get("checkpoint_path")
    PREDICTOR = RiskPredictor(BASE / _ckpt) if _ckpt else RiskPredictor()
    print(f"[model] loaded {PREDICTOR.model_name} (seq_len={PREDICTOR.seq_len})")
except Exception as _ex:  # torch missing, checkpoint missing, ...
    LOAD_ERROR = repr(_ex)
    print(f"[model] NOT loaded, using MOCK predictions: {LOAD_ERROR}")

    class BadInput(ValueError):
        pass

NODE_ORDER = [
    "api-server", "auth-service", "payment-service", "worker-service",
    "user-docs", "kyc-files", "audit-logs", "backups",
    "db-access-role", "lambda-exec-role", "readonly-role", "payment-role",
    "admin-role", "payment-lambda", "kyc-lambda", "notify-lambda",
    "fintech-api-gateway", "tx-queue", "notify-queue", "mysql-db",
    "redis-cache", "waf-firewall", "load-balancer", "cloudtrail-logs", "kms-keys",
]
LABELS = ["LOW", "MEDIUM", "HIGH"]
SEQ_LEN = PREDICTOR.seq_len if PREDICTOR else CFG["seq_len"]

# ---- in-memory state ----
STATE = {"history": [], "alerts": [], "latest_graph": None, "buffer": [],
         "replay": {"running": False, "sent": 0, "total": 0, "task": None}}
SUBSCRIBERS: set = set()

app = FastAPI(title="Risk Scoring API", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=CFG["cors_origins"],
                   allow_methods=["*"], allow_headers=["*"])


def err(code, message, status):
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


@app.exception_handler(HTTPException)
async def http_exc(_: Request, e: HTTPException):
    return err("BAD_INPUT" if e.status_code < 500 else "INTERNAL", str(e.detail), e.status_code)


@app.exception_handler(RequestValidationError)
async def val_exc(_: Request, e: RequestValidationError):
    return err("BAD_INPUT", str(e.errors()[:1]), 422)


# ---- helpers ----
def node_type(n):
    if n.endswith("-role"): return "role"
    if n.endswith("-lambda"): return "function"
    if n.endswith("-queue"): return "queue"
    if n in ("user-docs", "kyc-files", "audit-logs", "backups", "cloudtrail-logs"): return "storage"
    if n in ("mysql-db", "redis-cache", "kms-keys"): return "data"
    return "service"


def wend(w):
    """Window end time; falls back to 'timestamp' for old static snapshots."""
    return w.get("window_end") or w.get("timestamp") or ""


def wsort_key(w):
    n = w.get("window")
    return (w.get("window_start") or w.get("timestamp") or "", n if isinstance(n, int) else 0)


def mock_predict(windows):
    """MOCK. Replace with real model inference using CFG['checkpoint_path'].
    Seeded by the last window so the same input gives the same output."""
    last = windows[-1]
    rng = random.Random(wend(last))
    raw = [rng.random() ** 2 for _ in LABELS]
    # nudge HIGH up when the window has attack edges
    attack = sum(1 for e in last.get("edges", []) if e.get("type") == "dynamic_attack")
    raw[2] += min(attack * 0.15, 1.0)
    s = sum(raw)
    probs = {l: round(v / s, 4) for l, v in zip(LABELS, raw)}
    risk_class = max(probs, key=probs.get)
    node_risk = {n: round(rng.random() * 0.4, 3) for n in NODE_ORDER}
    for e in last.get("edges", []):  # nodes touched by attack edges look riskier
        if e.get("type") == "dynamic_attack":
            for k in ("source", "target"):
                if e[k] in node_risk:
                    node_risk[e[k]] = round(min(1.0, node_risk[e[k]] + 0.4 + rng.random() * 0.2), 3)
    return {
        "window_end": wend(last),
        "risk_class": risk_class,
        "probabilities": probs,
        "horizon": CFG["horizon"],
        "node_risk": node_risk,
        "top_paths": mock_paths(last, node_risk),
        "placeholder": ["node_risk", "top_paths"],
        "model": {"name": CFG["model_name"], "version": CFG["model_version"]},
    }


def predict_window(windows, pad_short=False):
    if PREDICTOR is None:
        return mock_predict(windows)
    pred = PREDICTOR.predict(windows, pad_short=pad_short)
    if not pred.get("window_end"):
        pred["window_end"] = wend(windows[-1])
    return pred


def mock_paths(window, node_risk):
    out_edges = {}
    for e in window.get("edges", []):
        out_edges.setdefault(e["source"], []).append(e)
    paths = []
    starts = ["api-server", "fintech-api-gateway", "load-balancer"]
    for start in starts:
        path, cur, seen = [start], start, {start}
        while len(path) < 5:
            nxt = [e for e in out_edges.get(cur, []) if e["target"] not in seen]
            if not nxt: break
            best = max(nxt, key=lambda e: node_risk.get(e["target"], 0) + e.get("weight", 0) * 0.01)
            cur = best["target"]; seen.add(cur); path.append(cur)
        if len(path) > 1:
            score = round(sum(node_risk.get(n, 0) for n in path) / len(path), 3)
            paths.append({"nodes": path, "score": score})
    return sorted(paths, key=lambda p: -p["score"])[:5]


def build_graph(window, pred):
    if PREDICTOR is not None:
        g = PREDICTOR.graph_view(window, pred)
        for n in g["nodes"]:
            n["type"] = node_type(n["id"])
        return g
    nodes_in_win = window.get("nodes") or NODE_ORDER
    return {
        "window_end": pred["window_end"],
        "risk_class": pred["risk_class"],
        "nodes": [{"id": n, "risk": pred["node_risk"].get(n, 0.0), "type": node_type(n)} for n in nodes_in_win],
        "edges": [{"source": e["source"], "target": e["target"], "weight": e.get("weight", 1),
                   "type": e.get("type", "base"), "is_new": e.get("type", "base") != "base"}
                  for e in window.get("edges", [])],
    }


def record(window, pred):
    STATE["history"].append({k: pred[k] for k in ("window_end", "risk_class", "probabilities")})
    STATE["history"] = STATE["history"][-1000:]
    STATE["latest_graph"] = build_graph(window, pred)
    top = max(pred["node_risk"], key=pred["node_risk"].get)
    STATE["alerts"].append({"window_end": pred["window_end"], "risk_class": pred["risk_class"],
                            "probabilities": pred["probabilities"], "top_node": top,
                            "top_paths": pred["top_paths"]})
    STATE["alerts"] = STATE["alerts"][-1000:]


async def broadcast(pred):
    for ws in list(SUBSCRIBERS):
        try:
            await ws.send_json(pred)
        except Exception:
            SUBSCRIBERS.discard(ws)


# ---- endpoints ----
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": PREDICTOR is not None, "model_error": LOAD_ERROR}


@app.get("/meta")
def meta():
    if PREDICTOR is not None:
        m = PREDICTOR.meta()
        m["window_seconds"] = CFG["window_seconds"]
        return m
    return {
        "model": {"name": CFG["model_name"], "version": CFG["model_version"]},
        "seq_len": SEQ_LEN, "horizon": CFG["horizon"],
        "feature_names": ["cvss", "exposure", "privilege", "patch_status", "auth_strength",
                          "asset_type", "in_strength", "out_strength", "new_conn"],  # placeholder list
        "node_order": NODE_ORDER, "window_seconds": CFG["window_seconds"], "labels": LABELS,
    }


@app.post("/predict")
def predict(body: dict):
    windows = body.get("windows")
    if not isinstance(windows, list) or len(windows) != SEQ_LEN:
        raise HTTPException(400, f"need {SEQ_LEN} windows, got {len(windows) if isinstance(windows, list) else 0}")
    try:
        pred = predict_window(windows)
    except BadInput as ex:
        raise HTTPException(400, str(ex))
    record(windows[-1], pred)
    return pred


@app.get("/graph/latest")
def graph_latest():
    if STATE["latest_graph"] is None:
        raise HTTPException(400, "no data yet; call /predict or /replay/start")
    return STATE["latest_graph"]


@app.get("/history")
def history(limit: int = 100):
    return {"items": STATE["history"][-limit:]}


@app.get("/alerts")
def alerts(min_level: str = "MEDIUM", limit: int = 50):
    if min_level not in LABELS:
        raise HTTPException(400, f"min_level must be one of {LABELS}")
    lvl = LABELS.index(min_level)
    items = [a for a in STATE["alerts"] if LABELS.index(a["risk_class"]) >= lvl]
    return {"items": items[-limit:]}


# ---- replay ----
async def replay_loop(directory: Path, speed: float):
    files = []
    for f in directory.glob("window_*.json"):
        try:
            files.append(json.loads(f.read_text()))
        except Exception:
            continue
    files.sort(key=wsort_key)
    R = STATE["replay"]
    R.update(running=True, sent=0, total=len(files))
    STATE["buffer"] = []
    try:
        for w in files:
            STATE["buffer"].append(w)
            STATE["buffer"] = STATE["buffer"][-SEQ_LEN:]
            R["sent"] += 1
            try:
                pred = predict_window(STATE["buffer"], pad_short=True)
            except BadInput:
                pred = None  # skip a malformed window
            if pred:
                record(w, pred)
                await broadcast(pred)
            await asyncio.sleep(CFG["window_seconds"] / max(speed, 0.01))
    finally:
        R["running"] = False


@app.get("/replay/status")
def replay_status():
    R = STATE["replay"]
    return {"running": R["running"], "sent": R["sent"], "total": R["total"]}


@app.post("/replay/start")
async def replay_start(body: dict = None):
    body = body or {}
    if STATE["replay"]["running"]:
        raise HTTPException(400, "replay already running")
    directory = Path(body.get("dir") or CFG["snapshot_dir"])
    if not directory.is_dir():
        raise HTTPException(400, f"folder not found: {directory}")
    if not any(directory.glob("window_*.json")):
        raise HTTPException(400, f"no window_*.json files in {directory}")
    STATE["history"], STATE["alerts"], STATE["latest_graph"] = [], [], None
    STATE["replay"]["task"] = asyncio.create_task(replay_loop(directory, float(body.get("speed", 1))))
    return {"started": True, "dir": str(directory), "speed": body.get("speed", 1)}


@app.post("/replay/stop")
async def replay_stop():
    t = STATE["replay"]["task"]
    if t and not t.done():
        t.cancel()
    STATE["replay"]["running"] = False
    return {"stopped": True}


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    SUBSCRIBERS.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        SUBSCRIBERS.discard(ws)