"""Evaluate heuristic baseline on five-station env for N episodes."""
from __future__ import annotations

import sys
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np

from src.control.action_mask import ENTRY_CHOICES, heuristic_action
from src.env.five_station_env import FiveStationEnv
from src.experiment.scenario_loader import load_scenario


def run_episode(env: FiveStationEnv, seed: int) -> Dict:
    obs, info = env.reset()
    ep_reward = 0.0
    ep_granted = 0
    ep_ontime = 0
    ep_dep_delay = 0.0
    ep_fallback = 0
    ep_held = 0

    # Per-step platform usage
    plat_usage: list[float] = []

    while True:
        active = info.get("active_train_ids", [])
        if not active and env._episode_status.value != "running":
            break

        action_dict = {}
        for tid in active:
            train = env.cfg.trains[tid]
            st = env._journeys[tid].current_station
            try:
                bounds = env._get_effective_bounds(tid, st, env._t_now)
                plat, entry, _ = heuristic_action(
                    train, st, env._t_now, 0.0, env.cfg, bounds_override=bounds
                )
                platforms = train.candidate_platforms.get(st, [])
                plat_idx = platforms.index(plat) if plat in platforms else 0
                entry_idx = ENTRY_CHOICES.index(entry) if entry in ENTRY_CHOICES else 0
                action_dict[tid] = (plat_idx, entry_idx)
            except ValueError:
                pass

        obs, reward, terminated, truncated, info = env.step(action_dict)

        metrics = info.get("metrics", {})
        ep_granted += metrics.get("n_granted", 0)
        ep_ontime += metrics.get("n_ontime_dep", 0)
        ep_dep_delay += metrics.get("total_dep_delay", 0.0)
        ep_held += metrics.get("n_held", 0)
        ep_reward += sum(reward.values())
        if metrics.get("platform_utilization", 0) > 0:
            plat_usage.append(metrics["platform_utilization"])

        # Count fallbacks
        for g in info.get("granted_windows", []):
            if not g.winner:
                ep_fallback += 1

        if terminated or truncated:
            break

    return {
        "reward": ep_reward,
        "granted": ep_granted,
        "ontime": ep_ontime,
        "dep_delay": ep_dep_delay,
        "fallback": ep_fallback,
        "held": ep_held,
        "plat_util": np.mean(plat_usage) if plat_usage else 0.0,
        "ontime_rate": ep_ontime / max(ep_granted, 1),
        "status": env._episode_status.value,
    }


def main():
    n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    cfg = load_scenario(scenario_name="five_station")

    all_results = []
    for ep in range(n_episodes):
        env = FiveStationEnv(cfg, seed=ep)
        result = run_episode(env, ep)
        all_results.append(result)

        if (ep + 1) % max(1, n_episodes // 10) == 0:
            recent = all_results[-max(1, n_episodes // 10):]
            avg_rw = np.mean([r["reward"] for r in recent])
            avg_dd = np.mean([r["dep_delay"] for r in recent]) / max(np.mean([r["granted"] for r in recent]), 1)
            avg_ot = np.mean([r["ontime_rate"] for r in recent])
            avg_fb = np.mean([r["fallback"] for r in recent])
            avg_pu = np.mean([r["plat_util"] for r in recent])
            print(f"  ep {ep+1:>4}: reward={avg_rw:>7.1f}  dep_dly={avg_dd:>5.1f}s  "
                  f"ontime={avg_ot*100:>5.1f}%  fallback={avg_fb:>4.1f}  util={avg_pu*100:>4.1f}%")

    # Final summary
    rewards = [r["reward"] for r in all_results]
    dep_dlys = [r["dep_delay"] / max(r["granted"], 1) for r in all_results]
    ontime = [r["ontime_rate"] for r in all_results]
    fallbacks = [r["fallback"] for r in all_results]
    plat_utils = [r["plat_util"] for r in all_results]
    deadlocks = sum(1 for r in all_results if r["status"] == "deadlock")

    print(f"\n=== Heuristic Baseline ({n_episodes} episodes) ===")
    print(f"  Reward:              {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"  Dep delay:           {np.mean(dep_dlys):.1f}s ± {np.std(dep_dlys):.1f}s")
    print(f"  On-time rate:        {np.mean(ontime)*100:.1f}%")
    print(f"  Fallback rate:       {np.mean(fallbacks):.1f} per episode")
    print(f"  Platform utilization: {np.mean(plat_utils)*100:.1f}%")
    print(f"  Deadlocks:           {deadlocks}")


if __name__ == "__main__":
    main()
