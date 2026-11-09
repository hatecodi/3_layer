#!/usr/bin/env python3
"""Fleet-level joint-SA scaling: time-to-target vs number of trains.

Produces the two curves for the "why coordination must be learned, not searched"
figure:

  B (main)  joint SA's time-to-target vs N, log y.  The joint decision vector is
            ALL trains' per-section speed modes -> 3**(4N) search space, so the
            iterations needed grow; wall-clock is therefore super-linear.
  A (inset) the search-space cardinality 3**(4N) vs N, log y — the reason.

Reference: the learned L3 worker's per-section cost is O(1) in N (one forward
pass per section per transit), i.e. a FLAT line on the same axes.

Target: the decomposed (per-transit independent) solution, scored on the JOINT
evaluator.  The question the curve answers is "how long must the global search
run merely to MATCH solving each leg on its own?"

NOTE: unlike `src/control/l3_sa.py` (per-transit, fixed 3**4 = 81 space), the
cost measured here grows with fleet size BY CONSTRUCTION of the joint vector.

Usage:
    python scripts/bench_joint_sa_scaling.py --scales 10 20 --n-iter 3000
    python scripts/bench_joint_sa_scaling.py --scales 10 15 20 25 30
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment.scenario_loader import load_scenario
from src.env.five_station_env import FiveStationEnv
from src.control.joint_sa import (
    JointTransit, build_joint_vector, joint_sa_optimize, make_evaluator,
)


def build_instance(scale: int, incident: bool = False):
    """All trains' FIRST corridor traversal, as one joint instance."""
    scenario = f"five_station_{scale}t"
    cfg = load_scenario(scenario_name=scenario)
    env = FiveStationEnv(cfg=cfg, seed=0, enable_vt=False,
                         enable_initial_delay=False,
                         enable_corridor_incident=incident)
    sim = env._l3_sim
    p = sim.p

    transits = []
    skipped = []
    for tid in sorted(cfg.trains):
        train = cfg.trains[tid]
        route = env._get_train_route(train)
        if len(route) < 2:
            skipped.append(tid); continue
        origin, nxt = route[0], route[1]
        corridor = f"{origin}_to_{nxt}"
        sections = sim._sections.get(corridor)
        if not sections:
            skipped.append(tid); continue
        planned = train.planned
        if origin not in planned or nxt not in planned:
            skipped.append(tid); continue
        dep = planned[origin].dep
        arr = planned[nxt].arr
        if dep <= 0 or arr <= 0:
            skipped.append(tid); continue
        distance = sum(s.length_m for s in sections)
        if distance <= 0:
            skipped.append(tid); continue
        planned_transit = max(arr - dep, 1.0)
        speeds = p.get_speeds(train.priority)
        schedule_cruise = min(
            distance / max(planned_transit - p.corridor_overhead_sec, 1.0),
            speeds.max_mps)
        inc = None
        if incident:
            key = tid
            got = dict(getattr(env, "_corridor_incidents", {}) or {}).get(corridor)
            if got:
                inc = (got[0], got[1], got[2], got[3]) if len(got) >= 4 else None
        transits.append(JointTransit(
            train_id=tid, corridor_id=corridor, departure=dep,
            priority=train.priority, sections=sections,
            bounds=sim._sec_bounds.get(corridor, [0.0] * (len(sections) + 1)),
            planned_arrival=arr, schedule_cruise_mps=schedule_cruise,
            incident_window=inc))
    build_joint_vector(transits)
    return cfg, sim, transits, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=int, nargs="+",
                    default=[10, 15, 20, 25, 30])
    ap.add_argument("--n-iter", type=int, default=3000,
                    help="joint-SA iteration budget per measured run")
    ap.add_argument("--ref-iters", type=int, default=30000,
                    help="iteration budget for the reference run that defines V*")
    ap.add_argument("--rho", type=float, default=0.9,
                    help="target = V_cruise + rho*(V* - V_cruise)")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--incident", action="store_true")
    ap.add_argument("--out", type=Path,
                    default=Path("output/9.20/joint_sa_scaling.csv"))
    args = ap.parse_args()

    reward_fn = lambda dev, t: -abs(dev)
    rows = []
    print(f"{'scale':>6} {'trains':>7} {'vars':>5} {'cruise':>9} {'V*':>9} "
          f"{'gain':>8} {'target':>9} {'reached':>8} {'iters':>7} {'wall_s':>8}")

    for scale in args.scales:
        cfg, sim, transits, skipped = build_instance(scale, args.incident)
        n_vars = sum(t.n_sections for t in transits)
        if not transits:
            print(f"{scale:>6}  -- no transits (skipped={len(skipped)})")
            continue

        objective = make_evaluator(sim, transits, reward_fn)
        # Anchors on a scale-invariant RELATIVE gap.  An absolute target
        # ("match the decomposed solution, scored jointly") is NOT usable: at
        # large N every leg independently prefers FAST, so the decomposed vector
        # jams when scored jointly and scores WORSE than all-CRUISE — the target
        # would sit below the starting point.
        v_cruise = objective([0] * n_vars)
        ref = joint_sa_optimize(sim, transits, reward_fn, target=float("inf"),
                                n_iter=args.ref_iters, seed=0)
        v_star = ref.best_value
        gain = v_star - v_cruise
        target = v_cruise + args.rho * gain

        for seed in range(args.seeds):
            res = joint_sa_optimize(sim, transits, reward_fn, target=target,
                                    n_iter=args.n_iter, seed=seed)
            rows.append({
                "scale": scale, "n_trains": len(transits), "n_vars": n_vars,
                "n_corridors": len({t.corridor_id for t in transits}),
                "log10_space": f"{n_vars * math.log10(3):.2f}",
                "v_cruise": f"{v_cruise:.4f}", "v_star": f"{v_star:.4f}",
                "gain": f"{gain:.4f}", "rho": args.rho,
                "target": f"{target:.4f}", "seed": seed,
                "reached": int(res.reached),
                "iters_to_target": res.iters_to_target if res.iters_to_target is not None else "",
                "wall_s_to_target": f"{res.wall_s_to_target:.4f}" if res.wall_s_to_target is not None else "",
                "total_wall_s": f"{res.total_wall_s:.4f}",
                "best_value": f"{res.best_value:.4f}",
                "evals": res.evals,
            })
            it = res.iters_to_target if res.reached else f">{args.n_iter}"
            wt = f"{res.wall_s_to_target:.2f}" if res.wall_s_to_target is not None else "n/a"
            print(f"{scale:>6} {len(transits):>7} {n_vars:>5} {v_cruise:>9.2f} "
                  f"{v_star:>9.2f} {gain:>8.2f} {target:>9.2f} "
                  f"{str(res.reached):>8} {str(it):>7} {wt:>8}")

    if rows:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nCSV: {args.out}")


if __name__ == "__main__":
    main()
