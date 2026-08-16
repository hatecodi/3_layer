#!/usr/bin/env python3
"""Diagnose whether greedy per-transit SA leaves coordination headroom under a
corridor speed-restriction incident.

The coordination hypothesis: when a platoon is caught in an incident corridor,
each greedy per-transit SA optimises ONLY its own arrival (front occupant fixed)
and ignores that its speed choice affects the moving-block headway of the
followers.  A slow leader blocks fast followers.  A JOINT SA oracle optimises the
whole platoon's per-section speed profiles together (objective = sum of terminal
rewards, headway coupling via the sim), internalising the externality.

If joint-SA system ontime >> greedy-SA, coordination headroom exists (RL could
learn it).  If they are ~equal, the greedy baseline already captures the value
and direction-1 is dead.

Usage:
    python scripts/diagnose_l3_coordination.py [--corridor C_to_D] [--factor 0.3]
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.control.l3_sim import (
    L3CorridorState,
    L3Simulator,
    SpeedMode,
    calibrate_corridor_sections,
    compute_corridor_headway,
    load_l3_physics,
)
from src.control.l3_sa import sa_optimize_profile
from src.env.reward import L3_FAST_PENALTY, compute_l3_terminal_reward
from src.experiment.scenario_loader import load_scenario

ONTIME_THRESHOLD = 15.0


def build_sim(cfg):
    l3_physics = load_l3_physics(cfg.project_root / "configs" / "five_station")
    corridor_hw = compute_corridor_headway(cfg.trains, l3_physics)
    sections = calibrate_corridor_sections(cfg.trains, l3_physics)
    sim = L3Simulator(l3_physics, corridor_hw, {}, sections=sections)
    return sim, l3_physics


def platoon_on_corridor(cfg, corridor, sim, l3_physics, from_st, to_st):
    """Up-direction trains transiting from_st->to_st, at their planned departure."""
    secs = sim._sections.get(corridor)
    distance = sum(s.length_m for s in secs) if secs else 0.0
    bounds = sim._sec_bounds.get(corridor, [0.0])
    overhead = l3_physics.corridor_overhead_sec
    full_up = ["A", "B", "C", "D", "E"]
    full_down = ["E", "D", "C", "B", "A"]
    platoon = []
    for tid, t in cfg.trains.items():
        if t.direction != "up":
            continue
        # train's actual origin..dest must span from_st..to_st (in order)
        o_idx = full_up.index(t.origin_station)
        d_idx = full_up.index(t.destination_station)
        if o_idx > full_up.index(from_st) or d_idx < full_up.index(to_st):
            continue
        pa = t.planned.get(from_st)
        pb = t.planned.get(to_st)
        if not (pa and pb):
            continue
        if pa.dep <= 0 or pb.arr <= 0:
            continue
        speeds = l3_physics.get_speeds(t.priority)
        transit = max(pb.arr - pa.dep, 1.0)
        sc = min(distance / max(transit - overhead, 1.0), speeds.max_mps)
        platoon.append({
            "id": tid, "dep": pa.dep, "plan_arr": pb.arr, "prio": t.priority,
            "sc": sc, "entry": pa.dep + l3_physics.throat_out_sec,
        })
    platoon.sort(key=lambda x: x["entry"])
    return platoon, secs, bounds, distance


def greedy_sa(sim, corridor, secs, bounds, platoon, sf, saf, n_iter=300):
    """Front-to-back greedy per-transit SA.  Returns (arrivals, ontime_count)."""
    sim._corridors = L3CorridorState()
    arrivals = []
    for tr in platoon:
        reward_fn = lambda dev, t: compute_l3_terminal_reward(dev, True, priority=tr["prio"])
        modes, _ = sa_optimize_profile(
            sim, corridor, tr["entry"], tr["prio"], secs, bounds, tr["plan_arr"],
            reward_fn, n_iter=n_iter, seed=0,
            incident_window=(0.0, 1e9, sf, saf),  # whole-transit restriction
            schedule_cruise_mps=tr["sc"])
        t = tr["entry"]
        entry_times, exit_times = [], []
        for k, sec in enumerate(secs):
            wait, run = sim.section_progress(
                corridor, t, bounds[k], sec, SpeedMode(modes[k]), tr["prio"],
                speed_factor=sf, safety_factor=saf, schedule_cruise_mps=tr["sc"])
            entry_times.append(t)
            t += wait + run
            exit_times.append(t)
        sim._corridors.record_occupancy_abs(corridor, tr["id"], entry_times, exit_times)
        arrivals.append(t + sim.p.approach_sec + sim.p.throat_in_sec)
    return arrivals


def joint_sa(sim, corridor, secs, bounds, platoon, sf, saf, n_iter=800, seed=0):
    """Joint SA over the whole platoon's speed profiles (system objective)."""
    n_tr = len(platoon)
    n_sec = len(secs)
    rng = random.Random(seed)

    def eval_joint(modes_flat):
        sim._corridors = L3CorridorState()
        arrivals = []
        for i, tr in enumerate(platoon):
            m = modes_flat[i * n_sec:(i + 1) * n_sec]
            t = tr["entry"]
            entry_times, exit_times = [], []
            for k, sec in enumerate(secs):
                wait, run = sim.section_progress(
                    corridor, t, bounds[k], sec, SpeedMode(m[k]), tr["prio"],
                    speed_factor=sf, safety_factor=saf, schedule_cruise_mps=tr["sc"])
                entry_times.append(t)
                t += wait + run
                exit_times.append(t)
            sim._corridors.record_occupancy_abs(corridor, tr["id"], entry_times, exit_times)
            arrivals.append(t + sim.p.approach_sec + sim.p.throat_in_sec)
        return arrivals

    def score(modes_flat):
        arrs = eval_joint(modes_flat)
        tot = 0.0
        for i, tr in enumerate(platoon):
            tot += compute_l3_terminal_reward(arrs[i] - tr["plan_arr"], True, priority=tr["prio"])
            tot -= L3_FAST_PENALTY * sum(1 for m in modes_flat[i * n_sec:(i + 1) * n_sec] if m == 1)
        return tot

    best = [0] * (n_tr * n_sec)
    best_score = score(best)
    cur, cur_score = list(best), best_score
    t0, t1 = 10.0, 0.1
    for it in range(n_iter):
        cand = list(cur)
        j = rng.randrange(n_tr * n_sec)
        cand[j] = rng.choice([m for m in (0, 1, 2) if m != cand[j]])
        s = score(cand)
        temp = t0 * ((t1 / t0) ** ((it + 1) / n_iter))
        if s > cur_score or rng.random() < math.exp((s - cur_score) / temp):
            cur, cur_score = cand, s
            if s > best_score:
                best_score, best = s, list(cand)
    return eval_joint(best)


def report(label, arrivals, platoon):
    devs = [arrivals[i] - tr["plan_arr"] for i, tr in enumerate(platoon)]
    ontime = sum(1 for d in devs if abs(d) <= ONTIME_THRESHOLD)
    n = len(devs)
    mean_abs = sum(abs(d) for d in devs) / max(n, 1)
    mean_late = sum(max(0, d) for d in devs) / max(n, 1)
    print(f"  {label}: ontime={ontime}/{n} ({ontime/max(n,1)*100:.1f}%) "
          f"mean|dev|={mean_abs:6.1f}s mean_late={mean_late:6.1f}s")
    print(f"     devs: {[f'{d:+.0f}' for d in devs]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corridor", default="C_to_D")
    ap.add_argument("--factor", type=float, default=0.3)
    ap.add_argument("--safety", type=float, default=1.5)
    args = ap.parse_args()

    cfg = load_scenario(scenario_name="five_station_20t")
    sim, l3_physics = build_sim(cfg)
    from_st, to_st = args.corridor.split("_to_")
    platoon, secs, bounds, distance = platoon_on_corridor(cfg, args.corridor, sim, l3_physics, from_st, to_st)
    print(f"corridor={args.corridor} distance={distance:.0f}m "
          f"platoon={[p['id'] for p in platoon]}")
    print("entries: " + str([f"{p['id']}@{p['entry']:.0f}" for p in platoon]))

    greedy_arr = greedy_sa(sim, args.corridor, secs, bounds, platoon, args.factor, args.safety)
    joint_arr = joint_sa(sim, args.corridor, secs, bounds, platoon, args.factor, args.safety)

    print(f"\nincident factor={args.factor} safety={args.safety}:")
    report("greedy per-transit SA", greedy_arr, platoon)
    report("joint SA oracle      ", joint_arr, platoon)
    g_ont = sum(1 for i in range(len(platoon)) if abs(greedy_arr[i]-platoon[i]['plan_arr'])<=ONTIME_THRESHOLD)
    j_ont = sum(1 for i in range(len(platoon)) if abs(joint_arr[i]-platoon[i]['plan_arr'])<=ONTIME_THRESHOLD)
    print(f"  -> headroom: joint - greedy ontime = {j_ont - g_ont:+d} trains")


if __name__ == "__main__":
    main()
