"""Run ablation experiments: VT tiers + initial delay vs heuristic.

Experiment A — VT difficulty gradient (no initial delay):
  Easy:   prob=0.1, slots=1,   p1_bias=0.0
  Medium: prob=0.3, slots=1-2, p1_bias=0.3
  Hard:   prob=0.5, slots=2,   p1_bias=0.6

Experiment B — Initial delay only (no VT):
  prob=0.3, range=[30,90], VT disabled
"""
from __future__ import annotations

import sys, os, json
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

import numpy as np
from src.env.five_station_env import FiveStationEnv
from src.experiment.scenario_loader import load_scenario
from src.control.action_mask import ENTRY_CHOICES, DWELL_EXTEND_CHOICES, heuristic_action


VT_TIERS = {
    "VT_easy":   {"enabled": True, "prob_per_slot": 0.1, "slots_per_station": [1, 1], "p1_bias": 0.0},
    "VT_medium": {"enabled": True, "prob_per_slot": 0.3, "slots_per_station": [1, 2], "p1_bias": 0.3},
    "VT_hard":   {"enabled": True, "prob_per_slot": 0.5, "slots_per_station": [2, 2], "p1_bias": 0.6},
}


def eval_heuristic(cfg, vt_override, enable_delay, n_eps=100):
    """Run heuristic baseline for a given config."""
    results = []
    for ep in range(n_eps):
        env = FiveStationEnv(cfg, seed=ep, vt_config_override=vt_override,
                            enable_initial_delay=enable_delay)
        obs, info = env.reset()
        ep_gr, ep_ot, ep_dd, ep_fb = 0, 0, 0.0, 0
        while True:
            active = info.get("active_train_ids", [])
            if not active and env._episode_status.value != "running":
                break
            action_dict = {}
            for tid in active:
                train = cfg.trains[tid]
                st = env._journeys[tid].current_station
                try:
                    bounds = env._get_effective_bounds(tid, st, env._t_now)
                    plat, entry, dwell = heuristic_action(
                        train, st, env._t_now, 0.0, cfg, bounds_override=bounds)
                    platforms = train.candidate_platforms.get(st, [])
                    pi = platforms.index(plat) if plat in platforms else 0
                    ei = ENTRY_CHOICES.index(entry) if entry in ENTRY_CHOICES else 0
                    di = DWELL_EXTEND_CHOICES.index(dwell) if dwell in DWELL_EXTEND_CHOICES else 0
                    action_dict[tid] = (pi, ei, di)
                except ValueError:
                    pass
            obs, reward, terminated, truncated, info = env.step(action_dict)
            m = info.get("metrics", {})
            ep_gr += m.get("n_granted", 0)
            ep_ot += m.get("n_ontime_dep", 0)
            ep_dd += m.get("total_dep_delay", 0.0)
            for g in info.get("granted_windows", []):
                if not g.winner: ep_fb += 1
            if terminated or truncated: break
        results.append({
            "delay": ep_dd / max(ep_gr, 1),
            "ontime": ep_ot / max(ep_gr, 1),
            "fallback": ep_fb,
            "status": env._episode_status.value,
        })
    dd = np.mean([r["delay"] for r in results])
    ot = np.mean([r["ontime"] for r in results])
    fb = np.mean([r["fallback"] for r in results])
    dl = sum(1 for r in results if r["status"] == "deadlock")
    return dd, ot, fb, dl


def run_one_train(cfg, name, vt_override, enable_delay, episodes, extra_hp=None):
    """Run one MAPPO training and return results dict."""
    from src.experiment.run_train import HP, train as train_fn

    hp = dict(HP)
    hp["episodes"] = episodes
    if extra_hp:
        hp.update(extra_hp)

    def env_factory(seed):
        return FiveStationEnv(cfg, seed=seed, vt_config_override=vt_override,
                             enable_initial_delay=enable_delay)

    # Monkey-patch to suppress prints
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        result = train_fn(env_factory, argparse.Namespace(
            env="five_station", episodes=episodes, rollout_steps=512,
            cpu=False, lr=3e-4, seed=42,
        ), obs_dim=31)
    return result


if __name__ == "__main__":
    cfg = load_scenario(scenario_name="five_station")

    print("=" * 70)
    print("  Heuristic Baselines")
    print("=" * 70)

    baselines = {}

    # Experiment A: VT tiers (no delay)
    for name, vt_cfg in VT_TIERS.items():
        dd, ot, fb, dl = eval_heuristic(cfg, vt_cfg, enable_delay=False)
        baselines[name] = {"delay": dd, "ontime": ot}
        print(f"  {name:>12}:  delay={dd:.1f}s  ontime={ot*100:.1f}%  fb={fb:.1f}  deadlock={dl}")

    # Experiment B: Initial delay (no VT)
    dd, ot, fb, dl = eval_heuristic(cfg, {"enabled": False}, enable_delay=True)
    baselines["delay_only"] = {"delay": dd, "ontime": ot}
    print(f"  {'delay_only':>12}:  delay={dd:.1f}s  ontime={ot*100:.1f}%  fb={fb:.1f}  deadlock={dl}")

    # No VT, no delay (clean reference)
    dd, ot, fb, dl = eval_heuristic(cfg, {"enabled": False}, enable_delay=False)
    baselines["clean"] = {"delay": dd, "ontime": ot}
    print(f"  {'clean':>12}:  delay={dd:.1f}s  ontime={ot*100:.1f}%  fb={fb:.1f}  deadlock={dl}")

    print(f"\n  Baselines saved.  Run MAPPO training with:")
    print(f"    python -m src.experiment.run_train --env five_station --episodes 3000")
    print(f"    (set vt_config_override and enable_initial_delay in env factory)")
