"""Traffic simulator: normal fintech traffic + windowed attacks.

Each cycle of `--cycle-windows` windows gets a freshly randomized attack
schedule (order, dwell duration, idle gaps) instead of repeating the same
fixed pattern forever. This prevents the downstream model from learning
"window index -> attack type" instead of real graph/behavioral signal.

Every cycle is assigned a run_id and its schedule is written to
cloud_simulation/logs/run_schedules/<run_id>.json for reproducibility.
The current run_id/window/phase is also written to
cloud_simulation/logs/run_state.json on every tick so snapshot_builder.py
can tag each snapshot with which run/phase it came from.

Run:
    python simulator.py [--window-sec 300] [--cycle-windows 13]
                         [--seed 42] [--run-prefix sharanya]
"""

import argparse
import json
import random
import time
from pathlib import Path

import requests


API = "http://localhost:5000"
AUTH = "http://localhost:5001"
PAY = "http://localhost:5002"


def _find_project_root(start: Path) -> Path:
    """Walk up from this file until we find the folder containing both
    cloud_simulation/ and graph_model/ -- that's the project root, regardless
    of how deeply simulator.py itself is nested (e.g. cloud_simulation/
    traffic_simulator/simulator.py)."""

    for p in [start] + list(start.parents):
        if (p / "cloud_simulation").is_dir() and (p / "graph_model").is_dir():
            return p

    print(
        f"[WARN] Could not auto-detect project root from {start}; "
        f"falling back to its parent. If state files end up in the wrong "
        f"place, check this path."
    )
    return start.parent


BASE_DIR = _find_project_root(Path(__file__).resolve())

STATE_FILE = BASE_DIR / "cloud_simulation" / "logs" / "run_state.json"
SCHEDULE_DIR = BASE_DIR / "cloud_simulation" / "logs" / "run_schedules"

PHASES = ["recon", "brute", "exploit", "exfil"]


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


def normal_login():
    _post(
        f"{API}/api/login",
        {"username": "admin", "password": "admin123"}
    )


def normal_pay():
    _post(
        f"{API}/api/pay",
        {
            "amount": 100,
            "to_account": "MERCHANT-001",
            "from_account": "ACC-98234-XYZ"
        }
    )


def normal_data():
    _get(f"{API}/api/data")


def normal_health():
    _get(f"{API}/health")


def normal_burst():
    for fn in [normal_login, normal_data, normal_pay, normal_health]:
        if random.random() < 0.7:
            fn()
            time.sleep(random.uniform(0.5, 2.0))


def attack_recon():
    print("[ATTACK] recon: 150 rapid GETs")

    for _ in range(150):
        _get(f"{API}/api/data")
        _get(f"{API}/api/user-docs")
        time.sleep(0.1)


def attack_brute():
    print("[ATTACK] brute: 80 wrong passwords -> 401s")

    for i in range(80):
        _post(
            f"{API}/api/login",
            {
                "username": "admin",
                "password": f"wrong-{i}"
            }
        )
        time.sleep(0.05)


def attack_exploit():
    print("[ATTACK] exploit: forged token -> auth-service->admin-role edge")

    for tok in ["admin-token-abc123", "finance-token-abc123"]:
        _post(
            f"{AUTH}/validate-token",
            {"token": tok}
        )
        time.sleep(0.2)

    _post(
        f"{API}/api/pay",
        {
            "amount": 50000,
            "to_account": "ATTACKER",
            "from_account": "ACC-98234-XYZ"
        }
    )


def attack_exfil():
    print("[ATTACK] exfil: 80 doc/txn reads")

    for _ in range(30):
        _get(f"{API}/api/user-docs")
        time.sleep(0.1)

    for _ in range(30):
        _get(f"{PAY}/transactions")
        time.sleep(0.1)

    for _ in range(20):
        _get(f"{API}/api/data")
        time.sleep(0.1)


ATTACKS = {
    "recon": attack_recon,
    "brute": attack_brute,
    "exploit": attack_exploit,
    "exfil": attack_exfil
}


def generate_schedule(
    rng,
    cycle_windows,
    min_gap=1,
    max_gap=3,
    max_dwell=2
):
    """Randomize order, dwell duration, and idle gaps for one cycle.

    Returns {window_index: phase}. Windows not present in the dict are idle
    (no scheduled attack; normal_burst traffic only).
    """

    phases = PHASES[:]
    rng.shuffle(phases)

    schedule = {}

    cursor = rng.randint(0, max_gap)

    for phase in phases:
        if cursor >= cycle_windows:
            break

        dwell = rng.randint(1, max_dwell)

        for w in range(
            cursor,
            min(cursor + dwell, cycle_windows)
        ):
            schedule[w] = phase

        cursor += dwell + rng.randint(min_gap, max_gap)

    return schedule


def write_state(
    run_id,
    window,
    cycle_window,
    phase,
    window_sec
):
    STATE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(STATE_FILE, "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "window": window,
                "cycle_window": cycle_window,
                "phase": phase,
                "window_sec": window_sec,
            },
            f
        )


def save_schedule(
    run_id,
    seed,
    cycle_windows,
    window_sec,
    schedule
):
    SCHEDULE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        SCHEDULE_DIR / f"{run_id}.json",
        "w"
    ) as f:
        json.dump(
            {
                "run_id": run_id,
                "seed": seed,
                "cycle_windows": cycle_windows,
                "window_sec": window_sec,
                "schedule": schedule,
            },
            f,
            indent=2
        )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--window-sec",
        type=int,
        default=30
    )

    ap.add_argument(
        "--cycle-windows",
        type=int,
        default=13,
        help="Windows per cycle before a new random schedule is drawn."
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Master seed. Omit for a fresh random seed (recorded in logs either way)."
    )

    ap.add_argument(
        "--run-prefix",
        type=str,
        default="",
        help="Optional namespace prefix for run IDs, e.g. sharanya."
    )

    ap.add_argument(
        "--min-gap",
        type=int,
        default=1
    )

    ap.add_argument(
        "--max-gap",
        type=int,
        default=3
    )

    ap.add_argument(
        "--max-dwell",
        type=int,
        default=2
    )

    args = ap.parse_args()

    master_seed = (
        args.seed
        if args.seed is not None
        else random.SystemRandom().randint(
            0,
            2**31 - 1
        )
    )

    rng = random.Random(master_seed)

    print(f"Simulator started. BASE_DIR={BASE_DIR}")
    print(f"  STATE_FILE={STATE_FILE}")
    print(f"  SCHEDULE_DIR={SCHEDULE_DIR}")

    print(
        f"Simulator started. "
        f"master_seed={master_seed}, "
        f"window_sec={args.window_sec}, "
        f"cycle_windows={args.cycle_windows}, "
        f"run_prefix={args.run_prefix or '(none)'}"
    )

    window = 0
    cycle_idx = 0
    schedule = {}
    run_id = None

    while True:

        tick_start = time.monotonic()

        cycle_window = window % args.cycle_windows

        if cycle_window == 0:

            cycle_idx += 1

            # ---------------------------------------------------------
            # NEW: namespace the run ID when --run-prefix is provided
            # ---------------------------------------------------------
            if args.run_prefix:
                run_id = (
                    f"run_{args.run_prefix}_"
                    f"{cycle_idx:04d}_"
                    f"seed{master_seed}"
                )
            else:
                # Keeps compatibility with your existing 200 snapshots.
                run_id = (
                    f"run_{cycle_idx:04d}_"
                    f"seed{master_seed}"
                )

            schedule = generate_schedule(
                rng,
                args.cycle_windows,
                args.min_gap,
                args.max_gap,
                args.max_dwell
            )

            save_schedule(
                run_id,
                master_seed,
                args.cycle_windows,
                args.window_sec,
                schedule
            )

            print(
                f"=== New cycle: {run_id} "
                f"schedule={schedule} ==="
            )

        phase = schedule.get(cycle_window)

        print(
            f"--- {run_id} | "
            f"global_window {window} | "
            f"cycle_window {cycle_window} | "
            f"phase={phase or 'idle'} ---"
        )

        write_state(
            run_id,
            window,
            cycle_window,
            phase,
            args.window_sec
        )

        if phase:
            ATTACKS[phase]()

        normal_burst()

        window += 1

        # Lock ticks to wall-clock boundaries.
        elapsed = time.monotonic() - tick_start

        remaining = args.window_sec - elapsed

        if remaining > 0:
            time.sleep(remaining)
        else:
            print(
                f"[WARN] window {window - 1} took "
                f"{elapsed:.1f}s, longer than "
                f"--window-sec {args.window_sec}. "
                f"Increase --window-sec or shorten "
                f"attack functions to avoid drift."
            )


if __name__ == "__main__":
    main()