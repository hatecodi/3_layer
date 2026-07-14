#!/usr/bin/env python3
"""Quick L2-only training — verify persistent calendar prevents resource overlaps.

Usage:
    python scripts/train_l2_quick.py --episodes 500
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.env.five_station_env import FiveStationEnv
from src.env.observation_builder import get_obs_dim, critic_input
from src.marl.actor_critic import Layer2ActorCritic, compact_global_state
from src.marl.mappo import MAPPOTrainer, AgentAction
from src.marl.rollout_buffer import RolloutBuffer, AgentTransition
from src.experiment.scenario_loader import load_scenario
from src.control.resource_manager import build_conflict_graph, ResourceCalendar

HP = {
    "obs_dim": 29, "action_dim": 45, "k_local": 1, "hidden_dim": 128,
    "lr": 3e-4, "gamma": 0.99, "gae_lambda": 0.95, "clip_ratio": 0.2,
    "ppo_epochs": 5, "minibatch_size": 256,
    "entropy_coef_start": 0.01, "entropy_coef_end": 0.005,
    "value_coef": 0.5, "max_grad_norm": 1.0, "rollout_steps": 512,
}


def _build_critic_inputs(obs, requests):
    conflict_graph = build_conflict_graph(requests)
    group_obs: Dict[str, List[float]] = {}
    for agent_id in obs:
        group_obs[agent_id] = critic_input(
            agent_id, obs, conflict_graph, k_max=HP["k_local"])
    return group_obs


@torch.no_grad()
def validate_episode(env, l2_trainer):
    obs, info = env.reset()
    all_events = []
    held_count = 0
    ep_done = False

    while not ep_done:
        masks = info.get("action_masks", {})
        requests = info.get("requests", [])
        group_obs = _build_critic_inputs(obs, requests)
        gs_vec = compact_global_state(info.get("metrics", {}))

        agent_actions = l2_trainer.collect_step(
            obs, masks, group_obs, gs_vec, device=torch.device("cpu"))
        action_dict = {aid: (a.platform, a.entry) for aid, a in agent_actions.items()}

        # Record grants before step
        for g in info.get("granted_windows", []):
            station = env._journeys[g.train_id].current_station
            all_events.append({
                "train": g.train_id, "station": station,
                "platform": g.platform,
                "t_arr": g.W_in[0], "t_dep": g.W_out[0],
            })
        held_count += len(info.get("held_train_ids", []))

        next_obs, reward, terminated, truncated, info = env.step(
            action_dict, lambda_team=l2_trainer.lam_team)
        ep_done = terminated or truncated
        obs = next_obs
    return all_events, held_count, info


def check_overlaps(events):
    violations = []
    bookings = defaultdict(list)
    for e in events:
        bookings[(e["station"], e["platform"])].append(e)
    for (st, plat), evts in bookings.items():
        evts.sort(key=lambda e: e["t_arr"])
        for i in range(len(evts)):
            for j in range(i + 1, len(evts)):
                a, b = evts[i], evts[j]
                if a["train"] == b["train"]:
                    continue
                gap = b["t_arr"] - a["t_dep"]
                if gap < 5.0:
                    violations.append(
                        f"OVERLAP: {st}/{plat} {a['train']}[{a['t_arr']:.0f},{a['t_dep']:.0f}] "
                        f"vs {b['train']}[{b['t_arr']:.0f},{b['t_dep']:.0f}] gap={gap:.1f}s")
    return violations


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cpu")
    cfg = load_scenario(scenario_name="five_station")
    obs_dim = get_obs_dim(cfg)

    ac = Layer2ActorCritic(obs_dim=obs_dim, k_local=HP["k_local"], hidden_dim=HP["hidden_dim"])
    l2_trainer = MAPPOTrainer(
        ac, lr=HP["lr"], gamma=HP["gamma"], gae_lambda=HP["gae_lambda"],
        clip_ratio=HP["clip_ratio"], ppo_epochs=HP["ppo_epochs"],
        minibatch_size=HP["minibatch_size"], entropy_coef=HP["entropy_coef_start"],
        value_coef=HP["value_coef"], max_grad_norm=HP["max_grad_norm"],
        total_episodes=args.episodes, device=device)
    l2_buffer = RolloutBuffer(gamma=HP["gamma"], gae_lambda=HP["gae_lambda"])

    torch.manual_seed(args.seed)
    print(f"Training L2 only, {args.episodes} episodes (no L3 worker, no VT, no delay)")

    total_violations = 0
    total_held = 0
    validation_interval = max(1, args.episodes // 10)

    for ep in range(args.episodes):
        l2_trainer.advance_episode()
        entropy_progress = min(ep / max(args.episodes * 0.3, 1), 1.0)
        l2_trainer.entropy_coef = (
            HP["entropy_coef_start"]
            + (HP["entropy_coef_end"] - HP["entropy_coef_start"]) * entropy_progress
        )

        env = FiveStationEnv(
            cfg=cfg, seed=ep, batch_window_sec=120.0,
            enable_vt=False, enable_initial_delay=False, l3_worker=None)
        obs, info = env.reset()
        ep_done = False
        ep_reward = 0.0
        ep_granted = 0
        ep_held = 0

        while not ep_done:
            active_ids = info.get("active_train_ids", [])
            masks = info.get("action_masks", {})
            requests = info.get("requests", [])
            group_obs = _build_critic_inputs(obs, requests)
            gs_vec = compact_global_state(info.get("metrics", {}))

            agent_actions = l2_trainer.collect_step(
                obs, masks, group_obs, gs_vec, device=device)
            action_dict = {aid: (a.platform, a.entry) for aid, a in agent_actions.items()}

            next_obs, reward, terminated, truncated, info = env.step(
                action_dict, lambda_team=l2_trainer.lam_team)
            ep_done = terminated or truncated
            metrics = info.get("metrics", {})
            ep_granted += metrics.get("n_granted", 0)
            ep_held += metrics.get("n_held", 0)
            reward_breakdown = info.get("reward_breakdown", {})

            # L2 transitions
            step_transitions = []
            for agent_id in active_ids:
                if agent_id not in agent_actions:
                    continue
                a = agent_actions[agent_id]
                r_local = reward_breakdown.get(agent_id, {}).get("local", 0.0)
                r_global = reward_breakdown.get(agent_id, {}).get("global_", 0.0)
                m = masks.get(agent_id)
                if m is not None:
                    pm = list(m[0]) + [False] * (4 - len(m[0]))
                    em = list(m[1])
                    p_mask = torch.tensor(pm, dtype=torch.bool)
                    e_mask = torch.tensor(em, dtype=torch.bool)
                else:
                    p_mask = torch.zeros(4, dtype=torch.bool)
                    e_mask = torch.zeros(6, dtype=torch.bool)
                step_transitions.append(AgentTransition(
                    agent_id=agent_id,
                    obs=torch.tensor(obs[agent_id], dtype=torch.float32),
                    platform=a.platform, entry=a.entry,
                    logprob=a.logprob,
                    platform_mask=p_mask, entry_mask=e_mask,
                    value_local=a.value_local, value_global=a.value_global,
                    reward_local=r_local, reward_global=r_global,
                    done=ep_done,
                    global_state=torch.tensor(gs_vec, dtype=torch.float32),
                    local_group_obs=torch.tensor(
                        group_obs.get(agent_id, obs[agent_id]), dtype=torch.float32),
                ))
            if step_transitions:
                l2_buffer.add_step(step_transitions,
                    torch.tensor(gs_vec, dtype=torch.float32))

            for tid, r in reward.items():
                ep_reward += r
            obs = next_obs

            if l2_buffer.num_steps >= HP["rollout_steps"]:
                final_v = {}
                if not ep_done:
                    for agent_id in active_ids:
                        if agent_id in agent_actions:
                            final_v[agent_id] = agent_actions[agent_id].value_local
                l2_buffer.compute_gae(final_v, 0.0)
                l2_trainer.update(l2_buffer)
                l2_buffer.clear()

        # End of episode PPO update
        if l2_buffer.num_steps > 0:
            l2_buffer.compute_gae({}, 0.0)
            l2_trainer.update(l2_buffer)
            l2_buffer.clear()

        total_held += ep_held
        status = info.get("episode_status", "?")

        if (ep + 1) % validation_interval == 0 or ep < 5:
            # Validate for conflicts
            env_val = FiveStationEnv(
                cfg=cfg, seed=999, batch_window_sec=120.0,
                enable_vt=False, enable_initial_delay=False, l3_worker=None)
            events, held, val_info = validate_episode(env_val, l2_trainer)
            violations = check_overlaps(events)
            total_violations += len(violations)

            mean_r = ep_reward / max(len(reward), 1) if 'reward' in dir() else 0
            print(f"  ep {ep:>5} | "
                  f"reward={ep_reward:>7.1f} | "
                  f"held={ep_held:>2} | "
                  f"violations={len(violations):>1} | "
                  f"mean_dep_dly={metrics.get('mean_dep_delay', 0):.1f}s | "
                  f"ontime={metrics.get('ontime_rate', 0):.2f} | "
                  f"{status}")

    print(f"\nTotal held across training: {total_held}")
    print(f"Total violations in validation: {total_violations}")

    # Final thorough validation
    print(f"\n=== Final validation (10 episodes) ===")
    all_violations = []
    for ep in range(10):
        env_val = FiveStationEnv(
            cfg=cfg, seed=1000 + ep, batch_window_sec=120.0,
            enable_vt=False, enable_initial_delay=False, l3_worker=None)
        events, held, val_info = validate_episode(env_val, l2_trainer)
        violations = check_overlaps(events)
        all_violations.extend(violations)
        if violations:
            for v in violations:
                print(f"  ep {ep}: {v}")
    if not all_violations:
        print(f"  ALL CLEAN — no platform overlaps in 10 validation episodes")
    else:
        print(f"  TOTAL: {len(all_violations)} platform overlaps")


if __name__ == "__main__":
    main()
