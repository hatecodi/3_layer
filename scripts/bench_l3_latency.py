#!/usr/bin/env python3
"""E1 — decision-latency benchmark: SA (batch re-solve) vs learned L3 worker.

Measures per-transit decision wall-clock time on a real C_to_D corridor:
  - SA:      sa_optimize_profile (300 annealing iterations over all sections) —
             the offline re-solve a batch optimizer must run per transit.
  - MAPPO:   the L3 worker's sequential per-section select_action (actor fwd +
             sample + critic fwd), i.e. the actual online decision loop.

Key structural fact the numbers show:
  - SA cost per transit = O(n_sections x n_iter) — a WHOLE new solve every time
    the state changes (a surprise / perturbation).
  - MAPPO cost per transit = O(n_sections) with a FIXED per-section forward —
    adaptation is just "run the next section's forward".
Both scale linearly in n_sections (ratio ~constant), so the decisive number is
the PER-ADAPTATION-EVENT cost: SA pays a full 300-iteration re-solve, MAPPO pays
one section forward.  Both numbers are current-implementation (Python/torch
overhead-bound) upper bounds — a batched MAPPO forward would be far cheaper.

Caveat: the scaling ratio does NOT widen with corridor length (both are linear);
what compounds is the number of perturbations (each costs SA a full re-solve).

Usage:
    python scripts/bench_l3_latency.py [--scenario five_station_15t]
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.experiment.scenario_loader import load_scenario
from src.env.five_station_env import FiveStationEnv
from src.control.l3_sa import sa_optimize_profile
from src.marl.l3_worker import CorridorActor, CorridorCritic, L3WorkerTrainer, L3_OBS_DIM


def _pick_corridor_train(cfg, corridor_key):
    a, b = corridor_key.split("_to_")
    for tid, t in cfg.trains.items():
        pa, pb = t.planned.get(a), t.planned.get(b)
        if pa and pb and pa.dep > 0 and pb.arr > 0:
            return t
    raise RuntimeError(f"no train over {corridor_key}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="five_station_15t")
    ap.add_argument("--corridor", default="C_to_D")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--l3", default=None, help="trained L3 checkpoint (else random init)")
    args = ap.parse_args()

    cfg = load_scenario(scenario_name=args.scenario)
    env = FiveStationEnv(cfg=cfg, seed=0, enable_vt=False,
                         enable_initial_delay=False, enable_corridor_incident=False)
    sim = env._l3_sim
    sections = sim._sections.get(args.corridor)
    bounds = sim._sec_bounds.get(args.corridor, [0.0])
    if not sections:
        sys.exit(f"no sections for {args.corridor}")
    n_sec = len(sections)
    distance = sum(s.length_m for s in sections)
    train = _pick_corridor_train(cfg, args.corridor)
    priority = train.priority
    a, b = args.corridor.split("_to_")
    planned_transit = max(train.planned[b].arr - train.planned[a].dep, 1.0)
    speeds = sim.p.get_speeds(priority)
    schedule_cruise = min(
        distance / max(planned_transit - sim.p.corridor_overhead_sec, 1.0),
        speeds.max_mps)
    planned_arrival = train.planned[b].arr
    departure = train.planned[a].dep

    sim._corridors.record_occupancy_abs(
        args.corridor, "front", [departure - 200.0] * n_sec,
        [departure - 200.0 + r for r in
         [sim.section_run_time(s, 0, priority, 1.0, schedule_cruise) for s in sections]])
    reward_fn = lambda dev, t: -abs(dev)
    print(f"corridor={args.corridor} sections={n_sec} distance={distance:.0f}m "
          f"P{priority} cruise={schedule_cruise:.1f}m/s planned_transit={planned_transit:.0f}s")

    # ---- SA full re-solve (n_iter=300), the cost of ONE adaptation ----
    n_sa = max(args.n // 100, 50)
    sa_samples = []
    for _ in range(n_sa):
        t0 = perf_counter()
        sa_optimize_profile(sim, args.corridor, departure, priority, sections, bounds,
                            planned_arrival, reward_fn, incident_window=None,
                            schedule_cruise_mps=schedule_cruise, n_iter=300)
        sa_samples.append((perf_counter() - t0) * 1000.0)
    sa_ms = statistics.median(sa_samples)

    # ---- MAPPO worker per-section decision (select_action: actor+critic) ----
    dev = "cpu"
    actor = CorridorActor(L3_OBS_DIM, hidden_dim=128)
    critic_local = CorridorCritic(L3_OBS_DIM, hidden_dim=128)
    critic_global = CorridorCritic(L3_OBS_DIM, hidden_dim=128)
    if args.l3:
        ck = torch.load(args.l3, map_location=dev, weights_only=False)
        actor.load_state_dict(ck["l3_actor"])
        critic_local.load_state_dict(ck["l3_critic_local"])
        critic_global.load_state_dict(ck["l3_critic_global"])
    actor.eval(); critic_local.eval(); critic_global.eval()
    worker = L3WorkerTrainer(actor, critic_local, critic_global, lr=1e-3, device=dev)
    obs = torch.randn(1, L3_OBS_DIM)
    for _ in range(2000):
        worker.select_action(obs)
    w_samples = []
    for _ in range(args.n):
        t0 = perf_counter()
        worker.select_action(obs)
        w_samples.append((perf_counter() - t0) * 1000.0)
    per_sec_ms = statistics.median(w_samples)
    a_samples = []
    for _ in range(args.n):
        t0 = perf_counter()
        with torch.no_grad():
            worker.actor.dist(obs).sample()
        a_samples.append((perf_counter() - t0) * 1000.0)
    per_sec_actor_ms = statistics.median(a_samples)

    full_worker_ms = per_sec_ms * n_sec
    full_actor_ms = per_sec_actor_ms * n_sec
    print(f"\nper-transit decision latency (median, N_worker={args.n}, N_sa={n_sa}):")
    print(f"  SA  (300-it full re-solve):  {sa_ms:7.2f} ms   per whole corridor")
    print(f"  MAPPO per section:           {per_sec_ms:7.2f} ms  (actor-only {per_sec_actor_ms:.2f} ms)")
    print(f"  MAPPO full corridor ({n_sec} sections): {full_worker_ms:7.2f} ms  "
          f"(actor-only {full_actor_ms:.2f} ms)")
    print(f"\nper-adaptation-event (a surprise at section k -> MAPPO just runs "
          f"section k's forward, SA re-solves):")
    print(f"  speedup SA / MAPPO-one-section:  {sa_ms/max(per_sec_ms,1e-9):7.1f}x")
    print(f"  speedup SA / MAPPO-one-section(actor): {sa_ms/max(per_sec_actor_ms,1e-9):7.1f}x")
    print(f"\nscaling with corridor length (SA cost ~ sections x 300 iters; MAPPO ~ sections x 1 fwd):")
    for ns in (4, 10, 20, 40, 80):
        print(f"  {ns:3d} sections:  SA ~{sa_ms*(ns/n_sec):8.1f} ms  |  "
              f"MAPPO ~{full_worker_ms*(ns/n_sec):7.1f} ms  "
              f"(x{sa_ms*(ns/n_sec)/max(full_worker_ms*(ns/n_sec),1e-9):.1f})")


if __name__ == "__main__":
    main()
