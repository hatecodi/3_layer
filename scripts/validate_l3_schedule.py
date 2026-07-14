#!/usr/bin/env python3
"""Validate L3 schedule: check resource overlaps, ontime, per-train delays.

Usage:
    python scripts/validate_l3_schedule.py --l2-ckpt <path> [--l3-ckpt <path>] [--n-episodes 10]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.env.five_station_env import FiveStationEnv
from src.env.observation_builder import get_obs_dim, critic_input
from src.marl.actor_critic import Layer2ActorCritic, compact_global_state
from src.marl.mappo import MAPPOTrainer
from src.marl.l3_worker import CorridorActor, CorridorCritic, L3WorkerTrainer, L3_OBS_DIM
from src.experiment.scenario_loader import load_scenario
from src.control.resource_manager import build_conflict_graph

HP = {
    "obs_dim": 29, "action_dim": 45, "k_local": 1,
    "hidden_dim": 128, "l3_hidden_dim": 64,
    "lr": 3e-4, "l3_lr": 1e-3,
    "gamma": 0.99, "gae_lambda": 0.95, "clip_ratio": 0.2,
    "ppo_epochs": 5, "minibatch_size": 256,
    "entropy_coef_start": 0.01, "entropy_coef_end": 0.005,
    "l3_entropy_coef": 0.01, "value_coef": 0.5,
    "max_grad_norm": 1.0, "rollout_steps": 512,
}


def _build_critic_inputs(obs, requests):
    conflict_graph = build_conflict_graph(requests)
    group_obs: Dict[str, List[float]] = {}
    for agent_id in obs:
        group_obs[agent_id] = critic_input(
            agent_id, obs, conflict_graph, k_max=HP["k_local"])
    return group_obs


@torch.no_grad()
def run_validation_episode(env, l2_trainer):
    obs, info = env.reset()
    events: List[dict] = []
    l3_choices_all: List[dict] = []
    ep_done = False

    while not ep_done:
        active_ids = info.get("active_train_ids", [])
        masks = info.get("action_masks", {})
        requests = info.get("requests", [])
        group_obs = _build_critic_inputs(obs, requests)
        gs_vec = compact_global_state(info.get("metrics", {}))

        agent_actions = l2_trainer.collect_step(
            obs, masks, group_obs, gs_vec, device=torch.device("cpu"))
        action_dict = {aid: (a.platform, a.entry) for aid, a in agent_actions.items()}

        t_now = env._t_now
        grant_windows = info.get("granted_windows", [])
        for g in grant_windows:
            station = env._journeys[g.train_id].current_station
            train_plan = env.cfg.trains[g.train_id].planned
            plan_entry = train_plan.get(station)
            planned_dep = plan_entry.dep if plan_entry else 0.0
            planned_arr = plan_entry.arr if plan_entry else 0.0
            events.append({
                "train": g.train_id,
                "station": station,
                "platform": g.platform,
                "t_arr": g.W_in[0],
                "t_dep": g.W_out[0],
                "planned_arr": planned_arr,
                "planned_dep": planned_dep,
            })

        l3_choices = info.get("l3_speed_choices", {})
        for tid, spd in l3_choices.items():
            l3_choices_all.append({"train": tid, "speed_mode": spd, "t_now": t_now})

        next_obs, reward, terminated, truncated, info = env.step(
            action_dict, lambda_team=l2_trainer.lam_team)
        ep_done = terminated or truncated
        obs = next_obs

    return events, l3_choices_all, info


def check_overlaps(events: List[dict], platform_min_gap: float = 5.0) -> List[str]:
    violations: List[str] = []
    throat_time = 3.0

    # Platform overlap check
    bookings: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for e in events:
        key = (e["station"], e["platform"])
        bookings[key].append(e)

    for (station, platform), evts in sorted(bookings.items()):
        evts = sorted(evts, key=lambda e: e["t_arr"])
        for i in range(len(evts)):
            for j in range(i + 1, len(evts)):
                a, b = evts[i], evts[j]
                if a["train"] == b["train"]:
                    continue
                gap = b["t_arr"] - a["t_dep"]
                if gap < platform_min_gap:
                    violations.append(
                        f"  PLATFORM OVERLAP: {station}/{platform} | "
                        f"{a['train']} dep={a['t_dep']:.1f} -> {b['train']} arr={b['t_arr']:.1f} "
                        f"gap={gap:.1f}s (min={platform_min_gap}s)"
                    )

    # Throat busy check: trains at same station departing within 6s
    for station in sorted(set(e["station"] for e in events)):
        evts = sorted([e for e in events if e["station"] == station], key=lambda e: e["t_dep"])
        for i in range(len(evts)):
            for j in range(i + 1, len(evts)):
                a, b = evts[i], evts[j]
                if a["train"] == b["train"]:
                    continue
                gap = abs(b["t_dep"] - a["t_dep"])
                if gap < throat_time * 2:
                    violations.append(
                        f"  THROAT BUSY: {station} | "
                        f"{a['train']}({a['platform']}) dep={a['t_dep']:.1f} vs "
                        f"{b['train']}({b['platform']}) dep={b['t_dep']:.1f} "
                        f"gap={gap:.1f}s (min={throat_time*2:.0f}s)"
                    )

    return violations


def main():
    parser = argparse.ArgumentParser(description="Validate L3 schedule")
    parser.add_argument("--l2-ckpt", type=str, required=True, help="L2 manager checkpoint")
    parser.add_argument("--l3-ckpt", type=str, default=None, help="L3 worker checkpoint")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--vt", action="store_true")
    parser.add_argument("--delay", action="store_true")
    args = parser.parse_args()

    cfg = load_scenario(scenario_name="five_station")
    obs_dim = get_obs_dim(cfg)
    device = torch.device("cpu")

    # L2 Manager
    l2_ckpt = torch.load(args.l2_ckpt, map_location=device, weights_only=False)
    print(f"L2 checkpoint episode: {l2_ckpt.get('episode', '?')}")

    ac = Layer2ActorCritic(obs_dim=obs_dim, k_local=HP["k_local"], hidden_dim=HP["hidden_dim"])
    ac.load_state_dict(l2_ckpt["ac_state"])
    ac.eval()
    for p in ac.parameters():
        p.requires_grad = False
    l2_trainer = MAPPOTrainer(
        ac, lr=HP["lr"], gamma=HP["gamma"], gae_lambda=HP["gae_lambda"],
        clip_ratio=HP["clip_ratio"], ppo_epochs=HP["ppo_epochs"],
        minibatch_size=HP["minibatch_size"], entropy_coef=HP["entropy_coef_start"],
        value_coef=HP["value_coef"], max_grad_norm=HP["max_grad_norm"],
        total_episodes=1, device=device)

    # L3 Worker
    if args.l3_ckpt:
        l3_ckpt = torch.load(args.l3_ckpt, map_location=device, weights_only=False)
        l3_actor = CorridorActor(obs_dim=L3_OBS_DIM, hidden_dim=HP["l3_hidden_dim"])
        l3_critic = CorridorCritic(obs_dim=L3_OBS_DIM, hidden_dim=HP["l3_hidden_dim"])
        l3_actor.load_state_dict(l3_ckpt["l3_actor"])
        l3_critic.load_state_dict(l3_ckpt["l3_critic"])
        l3_actor.eval()
        l3_critic.eval()
        for p in list(l3_actor.parameters()) + list(l3_critic.parameters()):
            p.requires_grad = False
        l3_worker = L3WorkerTrainer(
            l3_actor, l3_critic, lr=HP["l3_lr"], gamma=HP["gamma"],
            gae_lambda=HP["gae_lambda"], clip_ratio=HP["clip_ratio"],
            ppo_epochs=HP["ppo_epochs"], entropy_coef=HP["l3_entropy_coef"],
            value_coef=HP["value_coef"], max_grad_norm=HP["max_grad_norm"],
            device=device)
        print(f"L3 Worker: loaded")
    else:
        l3_worker = None
        print("L3 Worker: NONE (rule-based cruise)")

    all_violations = []

    for ep in range(args.n_episodes):
        l2_trainer.advance_episode()
        env = FiveStationEnv(
            cfg=cfg, seed=ep, batch_window_sec=120.0,
            enable_vt=args.vt, enable_initial_delay=args.delay,
            l3_worker=l3_worker,
        )
        events, l3_choices, info = run_validation_episode(env, l2_trainer)

        violations = check_overlaps(events)
        metrics = info.get("metrics", {})

        print(f"\n--- Episode {ep} | status={info.get('episode_status', '?')} ---")
        print(f"  Metrics: n_granted={metrics.get('n_granted','?')}  "
              f"ontime_dep={metrics.get('n_ontime_dep','?')}  "
              f"ontime_rate={metrics.get('ontime_rate','?'):.2f}  "
              f"total_dep_delay={metrics.get('total_dep_delay','?'):.1f}s  "
              f"mean_dep_delay={metrics.get('mean_dep_delay','?'):.1f}s  "
              f"mean_arr_delay={metrics.get('mean_arr_delay','?'):.1f}s")

        delays_by_train = defaultdict(list)
        for e in events:
            planned_dep = e["planned_dep"]
            delay = abs(e["t_dep"] - planned_dep) if planned_dep > 0 else 0
            delays_by_train[e["train"]].append((e["station"], e["t_dep"], planned_dep, delay))

        print(f"  Per-train departure delays:")
        for tid in sorted(delays_by_train.keys()):
            entries = delays_by_train[tid]
            parts = [f"{st}(dep={dep:.1f} dly={dly:.1f}s)" for st, dep, _, dly in entries]
            total_dly = sum(d for _, _, _, d in entries)
            p = cfg.trains[tid].priority
            n_stops = len(entries)
            print(f"    {tid} (P{p}, {n_stops}deps): sum_delay={total_dly:.1f}s  " + " | ".join(parts))

        print(f"\n  Schedule timeline:")
        for e in sorted(events, key=lambda e: e["t_dep"]):
            dly = e["t_dep"] - e["planned_dep"] if e["planned_dep"] else 0
            print(f"    {e['train']:>6s} @ {e['station']}/{e['platform']}  "
                  f"arr={e['t_arr']:7.1f}  dep={e['t_dep']:7.1f}  "
                  f"planned_dep={e['planned_dep']:7.1f}  delay={dly:+5.1f}s")

        if l3_choices:
            from collections import Counter
            cnt = Counter(c["speed_mode"] for c in l3_choices)
            modes = {0: "CRUISE", 1: "FAST", 2: "SLOW"}
            parts = [f"{modes.get(k, k)}={v}" for k, v in sorted(cnt.items())]
            print(f"\n  L3 speed choices: {', '.join(parts)}")

        if violations:
            print(f"\n  *** VIOLATIONS ({len(violations)}): ***")
            for v in violations:
                print(v)
            all_violations.extend(violations)
        else:
            print(f"  No violations.")

    print(f"\n{'='*60}")
    if all_violations:
        print(f"TOTAL VIOLATIONS across {args.n_episodes} episodes: {len(all_violations)}")
    else:
        print(f"ALL CLEAN: No resource violations across {args.n_episodes} episodes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
