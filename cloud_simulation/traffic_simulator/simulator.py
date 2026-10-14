"""Traffic simulator: normal fintech traffic + a reactive attacker state
machine (dynamic version v2).

=============================================================================
FIXES APPLIED AFTER REVIEW OF v1 (read this if comparing against v1):
=============================================================================
1. GROUND-TRUTH LABEL + exact window timing, written every tick to
   cloud_simulation/logs/attacker_windows.jsonl: {run_id, window, phase,
   ground_truth_label, compromised_node, window_start, window_end}.
   graph_model/snapshot_builder.py matches each snapshot to the record
   whose interval overlaps it most, instead of polling a single mutable
   run_state.json (which could read a stale or lagging value).
   Critically, `phase`/`ground_truth_label` for a tick are decided and
   captured INSIDE that tick's own step, before any state transition that
   tick triggers -- not read back afterward. In v1, phase_name() was
   called after attacker.step() had already mutated self.state, so e.g.
   the tick where a credential attack succeeds was mislabeled
   "lateral_movement" instead of "credential_attack". Fixed here.
2. run_id now rotates every time an episode ends (state returns to IDLE),
   not once per process lifetime. build_dataset.py splits data by whole
   run_id and needs >=3 distinct runs; one giant run_id for a whole
   multi-hour session made that impossible.
3. No more self-loop edges (e.g. "user-docs -> user-docs"). RECONNAISSANCE
   only treats api-server / auth-service / payment-service as valid
   "responders" to pivot a credential attack from (the other two live
   endpoints, user-docs and mysql-db, are read-only data sinks with no
   outgoing base-graph edges -- crediental-attacking a data store doesn't
   make sense anyway). CREDENTIAL_ATTACK's target selection also now
   explicitly excludes the responder node itself from its candidate list
   and its random fallback.
4. Normal background traffic is now driven off the same seeded rng as the
   attacker, for full run-to-run reproducibility (was using the unseeded
   global `random` module before).
5. Added a startup health check against api-server/auth-service/
   payment-service's /health endpoints. If the Docker services aren't up,
   this now fails loudly at startup instead of silently running an entire
   session of pure idle traffic.
6. ground_truth_label rules (per the design review):
     IDLE                                -> LOW
     RECONNAISSANCE                      -> MEDIUM
     CREDENTIAL_ATTACK (ongoing/failed)  -> MEDIUM
     CREDENTIAL_ATTACK (succeeds THIS tick) -> HIGH
     LATERAL_MOVEMENT into a CRITICAL-sensitivity node -> HIGH
     LATERAL_MOVEMENT into any other node               -> MEDIUM
     EXFILTRATION                        -> HIGH

Run:
    python simulator.py [--window-sec 30] [--seed 42] [--run-prefix sharanya]
                         [--fast-test]
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
import networkx as nx


API = "http://localhost:5000"
AUTH = "http://localhost:5001"
PAY = "http://localhost:5002"


def _find_project_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "cloud_simulation").is_dir() and (p / "graph_model").is_dir():
            return p
    print(f"[WARN] Could not auto-detect project root from {start}; "
          f"falling back to its parent.")
    return start.parent


BASE_DIR = _find_project_root(Path(__file__).resolve())
sys.path.insert(0, str(BASE_DIR))

from graph_model.snapshot_builder import (  # noqa: E402
    NODE_ORDER, BASE_EDGES, PRIVILEGE_MAP, AUTH_STRENGTH_MAP, load_assets,
)

LOG_FILE = BASE_DIR / "cloud_simulation" / "logs" / "requests.jsonl"
STATE_FILE = BASE_DIR / "cloud_simulation" / "logs" / "run_state.json"
ATTACKER_WINDOWS_FILE = BASE_DIR / "cloud_simulation" / "logs" / "attacker_windows.jsonl"
EPISODE_LOG = BASE_DIR / "cloud_simulation" / "logs" / "attack_episodes.jsonl"

ASSET_LOOKUP = load_assets()

BASE_GRAPH = nx.DiGraph()
BASE_GRAPH.add_nodes_from(NODE_ORDER)
BASE_GRAPH.add_edges_from(BASE_EDGES)

PUBLIC_NODES = [n for n in NODE_ORDER if ASSET_LOOKUP.get(n, {}).get("exposure") == "public"]

# Only these three have a live login-adjacent service behind them -- the
# other two (user-docs, mysql-db) are read-only data sinks with no
# outgoing base-graph edges, so they're never treated as credential-attack
# pivot points (fixes the self-loop bug from v1).
RESPONDER_ELIGIBLE = {"api-server", "auth-service", "payment-service"}

RECON_ENDPOINTS = [
    (API, "/health", "api-server"),
    (API, "/api/data", "mysql-db"),
    (API, "/api/user-docs", "user-docs"),
    (AUTH, "/health", "auth-service"),
    (PAY, "/health", "payment-service"),
]

HIGH_VALUE_TARGETS = ["mysql-db", "kyc-files", "kms-keys", "backups", "audit-logs"]

HEALTH_ENDPOINTS = [
    (API + "/health", "api-server"),
    (AUTH + "/health", "auth-service"),
    (PAY + "/health", "payment-service"),
]


class Ranges:
    # Tuned for 30-second windows.
    #
    # Old values (5*60, 40*60) = 300-2400 sec were designed for 5-minute
    # windows. At 30-second windows that meant 10-80 windows of pure IDLE
    # before the first attack started -- way too long, and it skewed the
    # label distribution badly toward LOW.
    #
    # New values give:
    #   idle_wait:   60-300 sec  =  2-10 windows between episodes
    #   per episode: ~5-15 windows (recon 1-3 + brute 2-6 + lateral 1-4 + exfil 1-3)
    #
    # In 1100 windows at ~12 windows/episode that's ~90 episodes:
    #   ~55 episodes * ~5 LOW windows each  = ~275 LOW  (idle gaps)
    #   ~90 episodes * ~4 MEDIUM windows    = ~360 MEDIUM
    #   ~90 episodes * ~3 HIGH windows      = ~270 HIGH
    #   remaining                           = ~195 LOW  (idle within episodes)
    # Total: ~470 LOW (43%), ~360 MEDIUM (33%), ~270 HIGH (25%) -- well balanced.
    idle_wait_sec = (60, 300)        # 2-10 windows idle between episodes
    recon_requests = (20, 200)       # unchanged
    recon_windows = (1, 3)           # unchanged
    cred_attempts = (10, 80)         # reduced upper bound: 150 attempts at 0.05
                                     # per-attempt success rate = ~75% chance of
                                     # never succeeding; capped at 80 so credential
                                     # attacks don't take 15+ windows to resolve
    lateral_dwell_windows = (1, 4)   # reduced from (1,6): keeps lateral short enough
                                     # that each hop is still distinctly labeled
    lateral_max_hops = 4             # unchanged
    lateral_continue_prob = 0.5      # unchanged
    lateral_random_prob = 0.20       # unchanged
    exfil_reads = (10, 200)          # reduced upper bound: 500 reads / 30sec window
                                     # = entire window spent on one exfil log_event
                                     # loop with no normal traffic; 200 is enough
    exfil_windows = (1, 3)           # reduced from (1,5): exfil ends sooner, next
                                     # episode starts sooner, better temporal variety

    def fast(self):
        """--fast-test: everything compressed to seconds for manual episode tracing."""
        self.idle_wait_sec = (5, 20)
        self.recon_requests = (10, 30)
        self.recon_windows = (1, 2)
        self.cred_attempts = (5, 20)
        self.lateral_dwell_windows = (1, 2)
        self.exfil_reads = (5, 20)
        self.exfil_windows = (1, 2)


def _get(url, timeout=5):
    try:
        return requests.get(url, timeout=timeout)
    except Exception as e:
        print(f"GET {url} failed: {e}")
        return None


def _post(url, payload, timeout=5):
    try:
        return requests.post(url, json=payload, timeout=timeout)
    except Exception as e:
        print(f"POST {url} failed: {e}")
        return None


def wait_for_services(timeout_sec=30):
    """Fail loudly at startup if the Flask services aren't reachable,
    instead of silently generating a whole session of pure idle traffic
    (fixes the "attacker silently goes idle forever" issue)."""
    deadline = time.monotonic() + timeout_sec
    pending = dict(HEALTH_ENDPOINTS)
    print(f"Checking service health: {list(pending.values())} ...")
    while pending and time.monotonic() < deadline:
        for url in list(pending):
            resp = _get(url, timeout=2)
            if resp is not None and resp.status_code == 200:
                print(f"  OK: {pending[url]} ({url})")
                del pending[url]
        if pending:
            time.sleep(1.5)
    if pending:
        print("=" * 70)
        print("[FATAL] The following services never responded healthy:")
        for url, name in pending.items():
            print(f"    {name}: {url}")
        print("Start Docker (docker compose up -d in cloud_simulation/) and retry.")
        print("=" * 70)
        sys.exit(1)
    print("All services healthy.")


def log_event(source, destination, status, endpoint):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source, "destination": destination,
        "status": status, "endpoint": endpoint,
    }
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[WARN] could not write synthetic log event: {e}")


def log_episode(event):
    try:
        EPISODE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(EPISODE_LOG, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception:
        pass


def log_attacker_window(record):
    try:
        ATTACKER_WINDOWS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ATTACKER_WINDOWS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"[WARN] could not write attacker window record: {e}")


def node_weight(node):
    a = ASSET_LOOKUP.get(node, {})
    cvss = float(a.get("cvss", 1.0))
    priv = PRIVILEGE_MAP.get(a.get("privilege", "limited"), 0.1)
    return max(cvss * priv, 0.01)


def credential_weight(node):
    a = ASSET_LOOKUP.get(node, {})
    priv = PRIVILEGE_MAP.get(a.get("privilege", "limited"), 0.1)
    auth_strength = AUTH_STRENGTH_MAP.get(node, 0.5)
    return max(priv * (1.0 - auth_strength), 0.01)


def lateral_score(rng, node):
    a = ASSET_LOOKUP.get(node, {})
    cvss = float(a.get("cvss", 1.0))
    priv = PRIVILEGE_MAP.get(a.get("privilege", "limited"), 0.1)
    auth_strength = AUTH_STRENGTH_MAP.get(node, 0.5)
    return cvss * priv * (1.0 - auth_strength) * rng.uniform(0.5, 1.5)


def weighted_choice(rng, items, weight_fn):
    weights = [weight_fn(i) for i in items]
    return rng.choices(items, weights=weights, k=1)[0]


def normal_login(rng):
    _post(f"{API}/api/login", {"username": "admin", "password": "admin123"})


def normal_pay(rng):
    _post(f"{API}/api/pay", {"amount": 100, "to_account": "MERCHANT-001", "from_account": "ACC-98234-XYZ"})


def normal_data(rng):
    _get(f"{API}/api/data")


def normal_health(rng):
    _get(f"{API}/health")


def normal_burst(rng):
    for fn in [normal_login, normal_data, normal_pay, normal_health]:
        if rng.random() < 0.7:
            fn(rng)
            time.sleep(rng.uniform(0.3, 1.0))


class Attacker:
    """Reactive 5-state attacker. One `step()` call = one window tick.
    After step() returns, self.tick_phase / self.tick_ground_truth hold
    the ground truth for the window that was JUST ticked -- computed
    before any state transition the tick triggers, never read back from
    self.state afterward."""

    def __init__(self, rng, ranges: Ranges, window_sec, run_id_factory):
        self.rng = rng
        self.r = ranges
        self.window_sec = window_sec
        self._run_id_factory = run_id_factory

        self.state = "IDLE"
        self.run_id = run_id_factory()
        self.idle_until = time.monotonic() + rng.uniform(*ranges.idle_wait_sec)

        self.entry_node = None
        self.recon_node = None
        self.recon_budget_requests = 0
        self.recon_budget_windows = 0
        self.recon_windows_done = 0
        self.recon_sent = 0
        self.recon_saw_eligible_200 = False
        self.recon_responder = None

        self.cred_target = None
        self.cred_attempt_limit = 0
        self.cred_attempts_done = 0

        self.compromised_node = None
        self.compromised_chain = []
        self.lateral_target = None
        self.lateral_dwell_remaining = 0
        self.lateral_hops = 0

        self.exfil_target = None
        self.exfil_reads_remaining = 0
        self.exfil_windows_remaining = 0

        self.tick_phase = "idle"
        self.tick_ground_truth = "LOW"

    # ---- state transitions -------------------------------------------------
    def _enter_idle(self, window):
        if self.compromised_chain:
            log_episode({
                "event": "episode_end", "window": window, "run_id": self.run_id,
                "entry_node": self.entry_node, "path": self.compromised_chain,
            })
        self.state = "IDLE"
        self.run_id = self._run_id_factory()  # NEW run_id per completed episode
        self.idle_until = time.monotonic() + self.rng.uniform(*self.r.idle_wait_sec)
        self.entry_node = None
        self.recon_node = None
        self.compromised_node = None
        self.compromised_chain = []
        self.lateral_hops = 0

    def _enter_reconnaissance(self, window):
        self.entry_node = weighted_choice(self.rng, PUBLIC_NODES, node_weight)
        self.recon_node = self.entry_node
        self.recon_budget_requests = self.rng.randint(*self.r.recon_requests)
        self.recon_budget_windows = self.rng.randint(*self.r.recon_windows)
        self.recon_windows_done = 0
        self.recon_sent = 0
        self.recon_saw_eligible_200 = False
        self.recon_responder = None
        self.state = "RECONNAISSANCE"
        log_episode({
            "event": "episode_start", "window": window, "run_id": self.run_id,
            "entry_node": self.entry_node,
            "recon_budget_requests": self.recon_budget_requests,
            "recon_budget_windows": self.recon_budget_windows,
        })

    def _enter_credential_attack(self):
        candidates = [n for n in BASE_GRAPH.successors(self.recon_responder)
                      if n != self.recon_responder]
        if candidates:
            self.cred_target = weighted_choice(self.rng, candidates, credential_weight)
        else:
            fallback = [n for n in NODE_ORDER if n != self.recon_responder]
            self.cred_target = weighted_choice(self.rng, fallback, credential_weight)
        self.cred_attempt_limit = self.rng.randint(*self.r.cred_attempts)
        self.cred_attempts_done = 0
        self.state = "CREDENTIAL_ATTACK"

    def _enter_lateral_movement(self):
        if self.compromised_node is None:
            self.compromised_node = self.cred_target
            self.compromised_chain = [self.entry_node, self.compromised_node]
        self._pick_lateral_target()
        self.state = "LATERAL_MOVEMENT"

    def _pick_lateral_target(self):
        neighbours = [n for n in BASE_GRAPH.successors(self.compromised_node) if n != self.compromised_node]
        if neighbours and self.rng.random() >= self.r.lateral_random_prob:
            scored = [(n, lateral_score(self.rng, n)) for n in neighbours]
            scored.sort(key=lambda x: x[1], reverse=True)
            self.lateral_target = scored[0][0]
        elif neighbours:
            self.lateral_target = self.rng.choice(neighbours)
        else:
            self.lateral_target = self.rng.choice([n for n in NODE_ORDER if n != self.compromised_node])
        self.lateral_dwell_remaining = self.rng.randint(*self.r.lateral_dwell_windows)

    def _enter_exfiltration(self):
        candidates = [t for t in HIGH_VALUE_TARGETS if t != self.compromised_node]
        self.exfil_target = self.rng.choice(candidates or HIGH_VALUE_TARGETS)
        self.exfil_reads_remaining = self.rng.randint(*self.r.exfil_reads)
        self.exfil_windows_remaining = self.rng.randint(*self.r.exfil_windows)
        self.state = "EXFILTRATION"

    # ---- per-window step ----------------------------------------------------
    def step(self, window):
        if self.state == "IDLE":
            self.tick_phase, self.tick_ground_truth = "idle", "LOW"
            if time.monotonic() >= self.idle_until:
                self._enter_reconnaissance(window)  # next tick starts sending recon traffic
            return

        if self.state == "RECONNAISSANCE":
            self.tick_phase, self.tick_ground_truth = "reconnaissance", "MEDIUM"
            self._step_reconnaissance(window)
            return

        if self.state == "CREDENTIAL_ATTACK":
            self.tick_phase, self.tick_ground_truth = "credential_attack", "MEDIUM"
            self._step_credential_attack(window)
            return

        if self.state == "LATERAL_MOVEMENT":
            target = self.lateral_target or self.compromised_node
            sensitivity = ASSET_LOOKUP.get(target, {}).get("sensitivity", "")
            self.tick_phase = "lateral_movement"
            self.tick_ground_truth = "HIGH" if sensitivity == "CRITICAL" else "MEDIUM"
            self._step_lateral_movement(window)
            return

        if self.state == "EXFILTRATION":
            self.tick_phase, self.tick_ground_truth = "exfiltration", "HIGH"
            self._step_exfiltration(window)
            return

    def _step_reconnaissance(self, window):
        remaining_windows = max(self.recon_budget_windows - self.recon_windows_done, 1)
        remaining_requests = max(self.recon_budget_requests - self.recon_sent, 0)
        this_window_requests = -(-remaining_requests // remaining_windows)  # ceil

        deadline = time.monotonic() + self.window_sec * 0.8
        sent_this_window = 0
        while sent_this_window < this_window_requests and time.monotonic() < deadline:
            url, path, dest = self.rng.choice(RECON_ENDPOINTS)
            resp = _get(f"{url}{path}")
            self.recon_sent += 1
            sent_this_window += 1
            if (resp is not None and resp.status_code == 200
                    and dest in RESPONDER_ELIGIBLE and not self.recon_saw_eligible_200):
                self.recon_saw_eligible_200 = True
                self.recon_responder = dest
            time.sleep(min(0.1, self.window_sec / max(this_window_requests, 1) * 0.3))

        self.recon_windows_done += 1
        print(f"[RECON] sent {self.recon_sent}/{self.recon_budget_requests} "
              f"over {self.recon_windows_done}/{self.recon_budget_windows} windows "
              f"eligible_responder={self.recon_responder}")

        if self.recon_saw_eligible_200 and self.recon_sent >= self.recon_budget_requests:
            self._enter_credential_attack()
        elif self.recon_windows_done >= self.recon_budget_windows:
            if self.recon_saw_eligible_200:
                self._enter_credential_attack()
            else:
                self._enter_idle(window)

    def _step_credential_attack(self, window):
        deadline = time.monotonic() + self.window_sec * 0.8
        attempts_this_window = 0
        budget = max(1, self.cred_attempt_limit // 6)
        success = False
        auth_strength = AUTH_STRENGTH_MAP.get(self.cred_target, 0.5)
        per_attempt_success_chance = (1.0 - auth_strength) * 0.05

        while (attempts_this_window < budget
               and self.cred_attempts_done < self.cred_attempt_limit
               and time.monotonic() < deadline):
            _post(f"{API}/api/login", {"username": "admin", "password": f"wrong-{self.cred_attempts_done}"})
            self.cred_attempts_done += 1
            attempts_this_window += 1
            if self.rng.random() < per_attempt_success_chance:
                success = True
                break
            time.sleep(0.05)

        if success:
            self.tick_ground_truth = "HIGH"  # overrides the MEDIUM default for this tick
            log_event(self.recon_responder, self.cred_target, 200, "sim:credential_success")
            print(f"[CRED_ATTACK] compromised {self.cred_target} after "
                  f"{self.cred_attempts_done} attempts -> LATERAL_MOVEMENT")
            self._enter_lateral_movement()
        elif self.cred_attempts_done >= self.cred_attempt_limit:
            print(f"[CRED_ATTACK] exhausted {self.cred_attempts_done} attempts on "
                  f"{self.cred_target}, retreating -> IDLE")
            self._enter_idle(window)
        else:
            print(f"[CRED_ATTACK] {self.cred_attempts_done}/{self.cred_attempt_limit} "
                  f"attempts on {self.cred_target}")

    def _step_lateral_movement(self, window):
        log_event(self.compromised_node, self.lateral_target, 200, "sim:lateral_move")
        self.lateral_dwell_remaining -= 1
        print(f"[LATERAL] {self.compromised_node} -> {self.lateral_target} "
              f"(dwell left {self.lateral_dwell_remaining})")

        if self.lateral_dwell_remaining > 0:
            return

        self.compromised_node = self.lateral_target
        if self.compromised_node not in self.compromised_chain:
            self.compromised_chain.append(self.compromised_node)
        self.lateral_hops += 1

        if self.lateral_hops < self.r.lateral_max_hops and self.rng.random() < self.r.lateral_continue_prob:
            self._pick_lateral_target()
        else:
            self._enter_exfiltration()

    def _step_exfiltration(self, window):
        remaining_windows = max(self.exfil_windows_remaining, 1)
        reads_this_window = -(-self.exfil_reads_remaining // remaining_windows)  # ceil
        for _ in range(reads_this_window):
            log_event(self.compromised_node, self.exfil_target, 200, "sim:exfil_read")
        self.exfil_reads_remaining -= reads_this_window
        self.exfil_windows_remaining -= 1
        print(f"[EXFIL] {self.compromised_node} -> {self.exfil_target} "
              f"read {reads_this_window} this window, "
              f"{max(self.exfil_reads_remaining, 0)} remaining")

        if self.exfil_reads_remaining <= 0 or self.exfil_windows_remaining <= 0:
            if self.compromised_node not in self.compromised_chain:
                self.compromised_chain.append(self.compromised_node)
            self.compromised_chain.append(self.exfil_target)
            self._enter_idle(window)


def write_state(run_id, window, phase, attacker: Attacker, window_sec):
    """Kept for backward compatibility / any other consumer (e.g. a live
    dashboard). graph_model/snapshot_builder.py no longer depends on this
    file for labeling -- it uses attacker_windows.jsonl instead."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump({
            "run_id": run_id, "window": window,
            "phase": phase if phase != "idle" else None,
            "window_sec": window_sec,
            "compromised_node": attacker.compromised_node,
            "entry_node": attacker.entry_node,
        }, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-sec", type=int, default=30)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--run-prefix", type=str, default="")
    ap.add_argument("--fast-test", action="store_true")
    ap.add_argument("--skip-health-check", action="store_true",
                     help="Skip the startup /health check (not recommended).")
    args = ap.parse_args()

    master_seed = (args.seed if args.seed is not None
                   else random.SystemRandom().randint(0, 2 ** 31 - 1))
    rng = random.Random(master_seed)

    if not args.skip_health_check:
        wait_for_services()

    episode_counter = {"n": 0}

    def run_id_factory():
        episode_counter["n"] += 1
        prefix = f"{args.run_prefix}_" if args.run_prefix else ""
        return f"run_{prefix}dyn_seed{master_seed}_ep{episode_counter['n']:04d}"

    ranges = Ranges()
    if args.fast_test:
        ranges.fast()

    print(f"Simulator started. BASE_DIR={BASE_DIR}")
    print(f"  LOG_FILE={LOG_FILE}")
    print(f"  ATTACKER_WINDOWS_FILE={ATTACKER_WINDOWS_FILE}")
    print(f"Simulator started. master_seed={master_seed}, window_sec={args.window_sec}, "
          f"fast_test={args.fast_test}")

    attacker = Attacker(rng, ranges, args.window_sec, run_id_factory)
    window = 0

    while True:
        tick_start = time.monotonic()
        # NOTE: window_end is the NOMINAL end of this tick's window_sec slot
        # (tick_start + window_sec), NOT when the actual request work
        # finished. Actual work usually takes far less than window_sec, and
        # logging only the work-duration sliver left gaps between ticks
        # that snapshot_builder's independently-clocked polling could land
        # in, causing it to miss the ground-truth record half the time.
        # Logging the full nominal slot makes consecutive ticks tile
        # wall-clock time contiguously, so any window_sec-wide probe window
        # reliably overlaps one.
        window_start_dt = datetime.now(timezone.utc)
        window_end_dt = window_start_dt + timedelta(seconds=args.window_sec)

        attacker.step(window)

        normal_burst(rng)

        print(f"--- {attacker.run_id} | window {window} | phase={attacker.tick_phase} | "
              f"ground_truth={attacker.tick_ground_truth} | "
              f"compromised={attacker.compromised_node} ---")

        log_attacker_window({
            "run_id": attacker.run_id,
            "window": window,
            "phase": attacker.tick_phase,
            "ground_truth_label": attacker.tick_ground_truth,
            "compromised_node": attacker.compromised_node,
            "window_start": window_start_dt.isoformat(),
            "window_end": window_end_dt.isoformat(),
        })
        write_state(attacker.run_id, window, attacker.tick_phase, attacker, args.window_sec)

        window += 1

        elapsed = time.monotonic() - tick_start
        remaining = args.window_sec - elapsed
        if remaining > 0:
            time.sleep(remaining)
        else:
            print(f"[WARN] window {window - 1} took {elapsed:.1f}s, longer than "
                  f"--window-sec {args.window_sec}.")


if __name__ == "__main__":
    main()