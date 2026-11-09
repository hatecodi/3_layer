#!/usr/bin/env python3
"""Verify the envelope-drift statistics quoted in Ch3 §3.6.3.

The two numbers under test (previously quoted as "measured" with NO producing
script in the repo):

    1. 70-76% of decision points have a dynamic envelope different from the
       static timetable envelope;
    2. only 0% (clean / normal incident) ~ 2.3% (extreme) of them actually
       RELAX the envelope enough to rescue an action that was illegal under the
       static envelope; rising monotonically with disturbance severity.

Definitions are pinned to the implementation (tolerance 1 s):

    static  B^s = cfg.trains[i].bounds[station]              (timetable expansion)
    dynamic B^d = env._get_effective_bounds(i, station, t_now)
                  (five_station_env.py:1292-1333)

    drift    |dB.arr_min| > TOL  or  |dB.dep_max| > TOL
    relax    dep_max widens  or  arr_min moves earlier
    tighten  the converse
    gained   #legal(B^d) > #legal(B^s)
    rescued  #legal(B^s) == 0 and #legal(B^d) > 0     <-- the strict sense of
                                                          "rescues an otherwise
                                                          illegal action"

Usage:
    python scripts/measure_envelope_drift.py --episodes 60 --arm incident
    python scripts/measure_envelope_drift.py --episodes 60 --arm incident --severity 2.0
    python scripts/measure_envelope_drift.py --episodes 60 --arm clean
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.experiment.scenario_loader import load_scenario
from src.env.five_station_env import FiveStationEnv, TRAVEL_TIME_TO_THROAT
from src.env.observation_builder import get_obs_dim, critic_input
from src.marl.actor_critic import Layer2ActorCritic, compact_global_state
from src.marl.mappo import MAPPOTrainer
from src.marl.l3_worker import CorridorActor, CorridorCritic, L3WorkerTrainer, L3_OBS_DIM
from src.control.action_mask import compute_joint_action_mask
from src.control.resource_manager import build_conflict_graph

DEFAULT_CKPT = ("output/8_16/training/five_station_15t_hierarchical/"
                "20260816_140621/checkpoints/best.pt")
TOL = 1.0


def count_legal(mask_set) -> int:
    return sum(1 for m in mask_set.mask if m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="five_station_15t")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--blocks", type=int, default=3,
                    help="independent seed blocks (seeds = seed0 + b*1e6)")
    ap.add_argument("--arm", choices=["clean", "incident"], default="incident")
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--seed0", type=int, default=900000)
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "output/9.20/envelope_drift.csv")
    args = ap.parse_args()

    dev = torch.device("cpu")
    cfg = load_scenario(scenario_name=args.scenario)
    od = get_obs_dim(cfg)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)

    ac = Layer2ActorCritic(obs_dim=od, k_local=1, hidden_dim=128)
    ac.load_state_dict(ck["ac_state"]); ac.eval()
    for p in ac.parameters():
        p.requires_grad = False
    l2 = MAPPOTrainer(ac, lr=3e-4, gamma=.99, gae_lambda=.95, clip_ratio=.2,
                      ppo_epochs=5, minibatch_size=256, entropy_coef=.01,
                      value_coef=.5, max_grad_norm=1.0, total_episodes=1, device=dev)

    la = CorridorActor(obs_dim=L3_OBS_DIM, hidden_dim=128)
    cl = CorridorCritic(obs_dim=L3_OBS_DIM, hidden_dim=128)
    cg = CorridorCritic(obs_dim=L3_OBS_DIM, hidden_dim=128)
    la.load_state_dict(ck["l3_actor"]); cl.load_state_dict(ck["l3_critic_local"])
    cg.load_state_dict(ck["l3_critic_global"])
    l3 = L3WorkerTrainer(la, cl, cg, lr=1e-3, device=dev)

    inc_on = (args.arm == "incident")
    # counters
    n = defaultdict(int)
    shifts = []                       # dep_max shift (dynamic - static)
    per_ep = []

    for b in range(args.blocks):
        base = args.seed0 + b * 1_000_000
        for ep_i in range(args.episodes):
            l2.advance_episode()
            env = FiveStationEnv(
                cfg=cfg, seed=base + ep_i, batch_window_sec=120.0,
                enable_vt=False, enable_initial_delay=False,
                enable_corridor_incident=inc_on, incident_scale=args.severity,
                l3_worker=l3)
            obs, info = env.reset()
            ep_counts = defaultdict(int)
            done = False
            while not done:
                t_now = env._t_now
                for tid in info.get("pending", []):
                    journey = env._journeys[tid]
                    station = journey.current_station
                    train = cfg.trains[tid]
                    Bs = train.bounds.get(station)
                    if Bs is None:
                        continue
                    Bd = env._get_effective_bounds(tid, station, t_now)

                    d_arr = Bd.arr_min - Bs.arr_min
                    d_dep = Bd.dep_max - Bs.dep_max

                    n["decisions"] += 1
                    ep_counts["decisions"] += 1
                    drifted = abs(d_arr) > TOL or abs(d_dep) > TOL
                    if drifted:
                        n["drift"] += 1; ep_counts["drift"] += 1
                    if d_dep > TOL or d_arr < -TOL:
                        n["relax"] += 1; ep_counts["relax"] += 1
                    if d_dep < -TOL or d_arr > TOL:
                        n["tighten"] += 1; ep_counts["tighten"] += 1
                    shifts.append(d_dep)

                    # masks under both envelopes
                    ms = compute_joint_action_mask(
                        train, station, t_now, TRAVEL_TIME_TO_THROAT, cfg,
                        bounds_override=Bs)
                    md = compute_joint_action_mask(
                        train, station, t_now, TRAVEL_TIME_TO_THROAT, cfg,
                        bounds_override=Bd)
                    ns, nd_ = count_legal(ms), count_legal(md)
                    if nd_ > ns:
                        n["gained"] += 1; ep_counts["gained"] += 1
                    if ns == 0 and nd_ > 0:
                        n["rescued"] += 1; ep_counts["rescued"] += 1
                    if ns == 0:
                        n["dead_static"] += 1; ep_counts["dead_static"] += 1

                cgx = build_conflict_graph(info.get("requests", []))
                gobs = {a: critic_input(a, obs, cgx, k_max=1) for a in obs}
                gs = compact_global_state(info.get("metrics", {}))
                with torch.no_grad():
                    aa = l2.collect_step(obs, info.get("action_masks", {}),
                                         gobs, gs, device=dev)
                ad = {k: (v.platform, v.entry) for k, v in aa.items()}
                obs, r, term, trunc, info = env.step(ad, lambda_team=l2.lam_team)
                done = term or trunc
            per_ep.append(ep_counts)

    D = max(n["decisions"], 1)
    mean_shift = sum(shifts) / max(len(shifts), 1)
    row = {
        "arm": args.arm, "severity": args.severity,
        "scenario": args.scenario, "episodes": args.blocks * args.episodes,
        "decisions": n["decisions"],
        "drift_pct": f"{100 * n['drift'] / D:.2f}",
        "relax_pct": f"{100 * n['relax'] / D:.2f}",
        "tighten_pct": f"{100 * n['tighten'] / D:.2f}",
        "gained_pct": f"{100 * n['gained'] / D:.2f}",
        "rescued_pct": f"{100 * n['rescued'] / D:.2f}",
        "dead_static_pct": f"{100 * n['dead_static'] / D:.2f}",
        "mean_dep_max_shift_s": f"{mean_shift:.2f}",
    }

    print(f"\n=== arm={args.arm} severity={args.severity} "
          f"({args.blocks}x{args.episodes} eps, {n['decisions']} decision points) ===")
    print(f"  drift (envelope differs)      : {row['drift_pct']}%")
    print(f"  relax (widened)               : {row['relax_pct']}%")
    print(f"  tighten (narrowed)            : {row['tighten_pct']}%")
    print(f"  gained (#legal up)            : {row['gained_pct']}%")
    print(f"  RESCUED (static infeasible -> : {row['rescued_pct']}%"
          f"   [static-infeasible base: {row['dead_static_pct']}%]")
    print(f"  mean dep_max shift            : {row['mean_dep_max_shift_s']} s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    exists = args.out.exists()
    with open(args.out, "a" if exists else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(f"\nCSV: {args.out}")


if __name__ == "__main__":
    main()
