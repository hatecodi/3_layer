#!/usr/bin/env python3
"""
Heuristic baseline — run N episodes with variable disturbance, collect metrics.

Usage:
    python scripts/run_heuristic_baseline.py [--episodes 100] [--prob 0.03]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.env.disturbance import CorridorDisturbance
from src.env.two_station_env import TwoStationEnv


def run_episode(env: TwoStationEnv, seed: int) -> Dict:
    """Run one episode with heuristic actions. Return summary dict."""
    obs, info = env.reset()

    step_rewards: List[Dict[str, float]] = []
    total_steps = 0
    episode_reward: Dict[str, float] = defaultdict(float)

    while True:
        # No policy actions → env uses heuristic internally
        obs, reward, terminated, truncated, info = env.step({})
        step_rewards.append(dict(reward))
        for tid, r in reward.items():
            episode_reward[tid] += r
        total_steps += 1

        if terminated or truncated:
            break

    m = info["metrics"]
    return {
        "seed": seed,
        "status": info["episode_status"],
        "steps": total_steps,
        "total_reward": sum(episode_reward.values()),
        "mean_reward_per_train": sum(episode_reward.values())
            / max(len(episode_reward), 1),
        "n_granted": m["n_granted"],
        "n_held": m["n_held"],
        "total_dep_delay": m["total_dep_delay"],
        "total_arr_delay": m["total_arr_delay"],
        "mean_dep_delay": m["mean_dep_delay"],
        "mean_arr_delay": m["mean_arr_delay"],
        "max_dep_delay": m["max_dep_delay"],
        "max_arr_delay": m["max_arr_delay"],
        "unresolved_conflicts": m["unresolved_conflicts"],
        "deadlock_flag": m["deadlock_flag"],
        "completion_rate": m["completion_rate"],
        "hard_bound_infeasible": m["hard_bound_infeasible_count"],
        "n_corridor_affected": len(info.get("corridor_delays", {})),
        "corridor_delay_total": sum(info.get("corridor_delays", {}).values()),
        "step_rewards": step_rewards,
    }


def main():
    parser = argparse.ArgumentParser(description="Heuristic baseline runner")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--prob", type=float, default=0.03,
                        help="Disturbance probability per step")
    parser.add_argument("--corridor", action="store_true", default=True,
                        help="Enable corridor speed restriction")
    parser.add_argument("--no-corridor", action="store_false", dest="corridor",
                        help="Disable corridor speed restriction")
    parser.add_argument("--corridor-start", type=float, default=120.0)
    parser.add_argument("--corridor-end", type=float, default=400.0)
    parser.add_argument("--corridor-ratio", type=float, default=0.3)
    args = parser.parse_args()

    corridor_ds = []
    if args.corridor:
        corridor_ds = [
            CorridorDisturbance(
                corridor_id="A_B_up",
                start_time=args.corridor_start,
                end_time=args.corridor_end,
                speed_ratio=args.corridor_ratio,
            )
        ]
        print(f"  Corridor restriction: A_B_up t=[{args.corridor_start},{args.corridor_end}] "
              f"speed={args.corridor_ratio*100:.0f}%")

    env = TwoStationEnv(
        disturbance_prob=args.prob,
        corridor_disturbances=corridor_ds,
        initial_train_delays={"K999": 60.0},  # K999 delayed from upstream
    )

    results = []
    status_counts: Dict[str, int] = defaultdict(int)
    all_dep_delays: List[float] = []
    all_arr_delays: List[float] = []
    all_rewards: List[float] = []

    for seed in range(args.episodes):
        env.rng.seed(seed)
        r = run_episode(env, seed)
        results.append(r)
        status_counts[r["status"]] += 1
        all_dep_delays.append(r["mean_dep_delay"])
        all_arr_delays.append(r["mean_arr_delay"])
        all_rewards.append(r["mean_reward_per_train"])

    # ---- print summary ----
    print(f"\n{'='*65}")
    print(f"  Heuristic Baseline — {args.episodes} episodes, prob={args.prob}")
    print(f"{'='*65}")

    print(f"\n  Episode outcomes:")
    for status, count in sorted(status_counts.items()):
        pct = count / args.episodes * 100
        bar = "#" * int(pct / 2)
        print(f"    {status:<12} {count:>4} ({pct:5.1f}%)  {bar}")

    print(f"\n  Mean per-train reward (across episodes):")
    valid = [r for r in all_rewards if r == r]  # filter NaN
    print(f"    avg={sum(valid)/max(len(valid),1):.2f}  "
          f"min={min(valid):.2f}  max={max(valid):.2f}")

    print(f"\n  Delay statistics (seconds, across episodes):")
    dep_valid = [d for d in all_dep_delays if d == d]
    arr_valid = [d for d in all_arr_delays if d == d]
    if dep_valid:
        print(f"    mean dep delay:  avg={sum(dep_valid)/len(dep_valid):.1f}s  "
              f"max={max(dep_valid):.1f}s")
    if arr_valid:
        print(f"    mean arr delay:  avg={sum(arr_valid)/len(arr_valid):.1f}s  "
              f"max={max(arr_valid):.1f}s")

    print(f"\n  Steps per episode:")
    steps = [r["steps"] for r in results]
    print(f"    avg={sum(steps)/len(steps):.1f}  "
          f"min={min(steps)}  max={max(steps)}")

    print(f"\n  Corridor disturbance impact:")
    corr_affected = [r["n_corridor_affected"] for r in results]
    corr_delay = [r["corridor_delay_total"] for r in results]
    print(f"    trains affected: avg={sum(corr_affected)/len(corr_affected):.1f}")
    print(f"    total extra delay: avg={sum(corr_delay)/len(corr_delay):.0f}s per episode")

    # ---- per-disturbance-level breakdown ----
    print(f"\n  Sample episodes (first 10):")
    header = f"  {'seed':>4} {'status':>12} {'steps':>5} {'rew/train':>8} {'dep_dly':>7} {'arr_dly':>7} {'held':>4} {'corr':>4}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results[:10]:
        print(f"  {r['seed']:>4} {r['status']:>12} {r['steps']:>5} "
              f"{r['mean_reward_per_train']:>8.2f} {r['mean_dep_delay']:>7.1f} "
              f"{r['mean_arr_delay']:>7.1f} {r['n_held']:>4} {r['n_corridor_affected']:>4}")

    print(f"\n  Done.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
