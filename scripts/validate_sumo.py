#!/usr/bin/env python3
"""
Phase 0: SUMO sanity baseline — validate network and routes.

Usage:
    python scripts/validate_sumo.py [--gui]

Runs sumo (or sumo-gui) with the project config, parses tripinfo output,
and reports pass/fail against Phase 0 criteria:
  1. No route errors, no stuck trains.
  2. All scheduled vehicles inserted.
  3. All inserted vehicles completed their routes.
  4. Basic metrics: #inserted, #completed, avg duration, avg waiting, depart delay.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUMO_CONFIG = PROJECT_ROOT / "sumo" / "two_station.sumocfg"
TRIPINFO_OUT = PROJECT_ROOT / "sumo" / "output" / "tripinfo.xml"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def ensure_output_dir() -> None:
    (PROJECT_ROOT / "sumo" / "output").mkdir(parents=True, exist_ok=True)


def run_sumo(gui: bool) -> int:
    """Run SUMO headless (or gui), return process returncode."""
    exe = "sumo-gui" if gui else "sumo"
    cmd = [
        exe,
        "-c", str(SUMO_CONFIG),
        "--tripinfo-output", str(TRIPINFO_OUT),
        "--no-warnings", "true",
        "--no-step-log", "true",
    ]
    print(f"[run] {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def parse_tripinfo(path: Path) -> List[Dict]:
    """Parse SUMO tripinfo xml -> list of per-vehicle dicts."""
    if not path.exists():
        return []
    tree = ET.parse(path)
    trips = []
    for child in tree.getroot():
        if child.tag == "tripinfo":
            trips.append({
                "id": child.get("id"),
                "depart": float(child.get("depart", 0)),
                "departDelay": float(child.get("departDelay", 0)),
                "arrival": float(child.get("arrival", -1)),
                "duration": float(child.get("duration", -1)),
                "routeLength": float(child.get("routeLength", -1)),
                "waitingTime": float(child.get("waitingTime", 0)),
                "vType": child.get("vType", ""),
                "vaporized": child.get("vaporized", "false"),
            })
    return trips


def report(trips: List[Dict]) -> int:
    """Print metrics table and return 0 on pass, 1 on failure."""
    n = len(trips)
    completed = [t for t in trips if t["arrival"] >= 0]
    n_done = len(completed)

    # ------------------------------------------------------------------
    # per-vehicle table
    # ------------------------------------------------------------------
    header = f"{'id':<8} {'type':<4} {'depart':>7} {'arrival':>8} {'dur':>7} {'wait':>6} {'depDelay':>8}  status"
    print(f"\n{'─' * len(header)}")
    print(header)
    print(f"{'─' * len(header)}")
    for t in trips:
        status = "OK" if t["arrival"] >= 0 else "MISSING"
        flag = "  ⚠" if status != "OK" else ""
        print(
            f"{t['id']:<8} {t['vType']:<4} {t['depart']:>7.0f} "
            f"{t['arrival']:>8.1f} {t['duration']:>7.1f} "
            f"{t['waitingTime']:>6.1f} {t['departDelay']:>8.1f}  {status}{flag}"
        )

    # ------------------------------------------------------------------
    # aggregate metrics (Phase 0, step 4)
    # ------------------------------------------------------------------
    avg_dur = sum(t["duration"] for t in completed) / n_done if n_done else 0
    avg_wait = sum(t["waitingTime"] for t in trips) / n if n else 0
    avg_dep_delay = sum(t["departDelay"] for t in trips) / n if n else 0
    stuck = [t for t in trips if t["arrival"] < 0]

    print(f"\n{'─' * 50}")
    print(f"  Phase 0 指标")
    print(f"{'─' * 50}")
    print(f"  计划车辆数        : {n}")
    print(f"  插入车辆数        : {n}")
    print(f"  完成车辆数        : {n_done}")
    print(f"  平均 duration     : {avg_dur:.1f} s")
    print(f"  平均 waitingTime  : {avg_wait:.1f} s")
    print(f"  平均 departDelay  : {avg_dep_delay:.1f} s")
    print(f"  未到达 (stuck)    : {len(stuck)}")

    # ------------------------------------------------------------------
    # Phase 0 pass criteria
    # ------------------------------------------------------------------
    print(f"\n{'─' * 50}")
    print(f"  Phase 0 检查")
    print(f"{'─' * 50}")

    failures = 0

    # P0-1: every planned vehicle appears in tripinfo
    expected_ids = {"K641", "D133", "G5050", "K667",
                    "G1408", "G5060", "Z386", "K1666"}
    seen_ids = {t["id"] for t in trips}
    if seen_ids == expected_ids:
        print(f"  ✓ 车辆插入        全部 8 辆已插入")
    else:
        missing = expected_ids - seen_ids
        extra = seen_ids - expected_ids
        print(f"  ✗ 车辆插入        缺少: {missing or '无'}, 多余: {extra or '无'}")
        failures += 1

    # P0-2: no train is stuck (every inserted train completed)
    if len(stuck) == 0:
        print(f"  ✓ 路线完成        全部 {n_done}/{n} 到达终点")
    else:
        print(f"  ✗ 路线完成        {len(stuck)} 辆车未到达: {[t['id'] for t in stuck]}")
        failures += 1

    # P0-3: no teleport / vaporized
    vap = [t for t in trips if t.get("vaporized", "") != ""]
    if not vap:
        print(f"  ✓ 未触发传送      无车辆被 vaporized")
    else:
        print(f"  ✗ 触发传送        {len(vap)} 辆车被 vaporized: {[t['id'] for t in vap]}")
        failures += 1

    if failures == 0:
        print(f"\n  ✅ Phase 0 PASS")
    else:
        print(f"\n  ❌ Phase 0 FAIL — {failures} 项未通过")

    return 0 if failures == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 SUMO sanity baseline")
    parser.add_argument("--gui", action="store_true",
                        help="Use sumo-gui instead of headless sumo")
    args = parser.parse_args()

    ensure_output_dir()

    t0 = time.perf_counter()
    rc = run_sumo(gui=args.gui)
    elapsed = time.perf_counter() - t0
    print(f"[run] exit code {rc}, wall time {elapsed:.1f}s")

    if rc != 0:
        print("SUMO exited with error — check network/route files for issues.")
        sys.exit(1)

    trips = parse_tripinfo(TRIPINFO_OUT)
    if not trips:
        print("No tripinfo entries found — no vehicles were simulated.")
        sys.exit(1)

    sys.exit(report(trips))


if __name__ == "__main__":
    main()
