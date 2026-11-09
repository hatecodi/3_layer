#!/usr/bin/env python3
"""Continue hierarchical training from checkpoint with VT enabled.

Usage:
    python scripts/continue_with_vt.py --l2-ckpt <path> --l3-ckpt <path> --episodes 3000
"""
from __future__ import annotations

import argparse, csv, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.env.five_station_env import FiveStationEnv
from src.env.observation_builder import get_obs_dim, critic_input
from src.marl.actor_critic import Layer2ActorCritic, compact_global_state
from src.marl.mappo import MAPPOTrainer, AgentAction
from src.marl.rollout_buffer import RolloutBuffer, AgentTransition
from src.marl.l3_worker import CorridorActor, CorridorCritic, L3WorkerTrainer, L3_OBS_DIM
from src.experiment.scenario_loader import load_scenario
from src.control.resource_manager import build_conflict_graph

HP = {
    "obs_dim": 29, "action_dim": 45, "k_local": 1, "hidden_dim": 128, "l3_hidden_dim": 128,
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
        group_obs[agent_id] = critic_input(agent_id, obs, conflict_graph, k_max=HP["k_local"])
    return group_obs


def main():
    parser = argparse.ArgumentParser(description="Continue training with VT from checkpoint")
    parser.add_argument("--l2-ckpt", type=str, required=True, help="L2 manager checkpoint")
    parser.add_argument("--l3-ckpt", type=str, default=None, help="L3 worker checkpoint")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cpu")
    cfg = load_scenario(scenario_name="five_station")
    obs_dim = get_obs_dim(cfg)

    # Load L2
    l2_ckpt = torch.load(args.l2_ckpt, map_location=device, weights_only=False)
    ac = Layer2ActorCritic(obs_dim=obs_dim, k_local=HP["k_local"], hidden_dim=HP["hidden_dim"])
    ac.load_state_dict(l2_ckpt["ac_state"])
    l2_trainer = MAPPOTrainer(
        ac, lr=HP["lr"] * 0.1, gamma=HP["gamma"], gae_lambda=HP["gae_lambda"],
        clip_ratio=HP["clip_ratio"], ppo_epochs=HP["ppo_epochs"],
        minibatch_size=HP["minibatch_size"], entropy_coef=HP["entropy_coef_end"],
        value_coef=HP["value_coef"], max_grad_norm=HP["max_grad_norm"],
        total_episodes=args.episodes, device=device)
    l2_buffer = RolloutBuffer(gamma=HP["gamma"], gae_lambda=HP["gae_lambda"])
    print(f"L2 loaded from: {args.l2_ckpt}")

    # Load L3
    if args.l3_ckpt:
        l3_ckpt = torch.load(args.l3_ckpt, map_location=device, weights_only=False)
        l3_actor = CorridorActor(obs_dim=L3_OBS_DIM, hidden_dim=HP["l3_hidden_dim"])
        l3_critic = CorridorCritic(obs_dim=L3_OBS_DIM, hidden_dim=HP["l3_hidden_dim"])
        l3_critic_g = CorridorCritic(obs_dim=L3_OBS_DIM, hidden_dim=HP["l3_hidden_dim"])
        l3_actor.load_state_dict(l3_ckpt["l3_actor"])
        l3_critic.load_state_dict(l3_ckpt["l3_critic_local"])
        l3_critic_g.load_state_dict(l3_ckpt["l3_critic_global"])
        l3_trainer = L3WorkerTrainer(
            l3_actor, l3_critic, l3_critic_g, lr=HP["l3_lr"] * 0.5, gamma=HP["gamma"],
            gae_lambda=HP["gae_lambda"], clip_ratio=HP["clip_ratio"],
            ppo_epochs=HP["ppo_epochs"], entropy_coef=HP["l3_entropy_coef"] * 0.3,
            value_coef=HP["value_coef"], max_grad_norm=HP["max_grad_norm"], device=device)
        print(f"L3 loaded from: {args.l3_ckpt}")
    else:
        l3_trainer = None

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "output" / "five_station_vt_continue" / run_id
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {run_dir}")

    torch.manual_seed(args.seed)
    episode_metrics = []
    best_reward = -float("inf")

    for ep in range(args.episodes):
        l2_trainer.advance_episode()
        if l3_trainer:
            l3_trainer.entropy_coef = max(HP["l3_entropy_coef"] * 0.3 * (1 - ep / args.episodes), 0.001)

        env = FiveStationEnv(
            cfg=cfg, seed=ep + 10000, batch_window_sec=120.0,
            enable_vt=True, enable_initial_delay=True, l3_worker=l3_trainer)
        obs, info = env.reset()
        ep_done = False
        ep_reward = 0.0
        ep_held = 0
        ep_granted = 0
        ep_ontime = 0
        ep_dep_dly = 0.0

        while not ep_done:
            active_ids = info.get("active_train_ids", [])
            masks = info.get("action_masks", {})
            requests = info.get("requests", [])
            group_obs = _build_critic_inputs(obs, requests)
            gs_vec = compact_global_state(info.get("metrics", {}))

            agent_actions = l2_trainer.collect_step(obs, masks, group_obs, gs_vec, device=device)
            action_dict = {aid: (a.platform, a.entry) for aid, a in agent_actions.items()}

            next_obs, reward, terminated, truncated, info = env.step(
                action_dict, lambda_team=l2_trainer.lam_team)
            ep_done = terminated or truncated
            metrics = info.get("metrics", {})
            ep_granted += metrics.get("n_granted", 0)
            ep_ontime += metrics.get("n_ontime_dep", 0)
            ep_dep_dly += metrics.get("total_dep_delay", 0.0)
            ep_held += metrics.get("n_held", 0)
            reward_breakdown = info.get("reward_breakdown", {})

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
                    agent_id=agent_id, obs=torch.tensor(obs[agent_id], dtype=torch.float32),
                    platform=a.platform, entry=a.entry, logprob=a.logprob,
                    platform_mask=p_mask, entry_mask=e_mask,
                    value_local=a.value_local, value_global=a.value_global,
                    reward_local=r_local, reward_global=r_global, done=ep_done,
                    global_state=torch.tensor(gs_vec, dtype=torch.float32),
                    local_group_obs=torch.tensor(group_obs.get(agent_id, obs[agent_id]), dtype=torch.float32),
                ))
            if step_transitions:
                l2_buffer.add_step(step_transitions, torch.tensor(gs_vec, dtype=torch.float32))

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

            if l3_trainer and len(l3_trainer.buffer) >= 64:
                l3_trainer.update()

        if l2_buffer.num_steps > 0:
            l2_buffer.compute_gae({}, 0.0)
            l2_trainer.update(l2_buffer)
            l2_buffer.clear()
        if l3_trainer and len(l3_trainer.buffer) >= 16:
            l3_trainer.update()

        mean_dep = ep_dep_dly / max(ep_granted, 1)
        ont_rate = ep_ontime / max(ep_granted, 1)
        ep_summary = {
            "episode": ep, "total_reward": ep_reward,
            "mean_dep_delay": mean_dep, "ontime_rate": ont_rate,
            "n_held": ep_held, "n_granted": ep_granted,
        }
        episode_metrics.append(ep_summary)

        if (ep + 1) % 100 == 0 or ep < 5:
            print(f"  ep {ep:>5} | reward={ep_reward:>8.1f} | "
                  f"dep_dly={mean_dep:>5.1f}s | ontime={ont_rate:.2f} | "
                  f"held={ep_held:>3} | granted={ep_granted:>3}")

        if ep_reward > best_reward:
            best_reward = ep_reward
            save = {"ac_state": ac.state_dict(), "episode": ep, "hp": HP}
            if l3_trainer:
                save["l3_actor"] = l3_trainer.actor.state_dict()
                save["l3_critic_local"] = l3_trainer.critic_local.state_dict()
                save["l3_critic_global"] = l3_trainer.critic_global.state_dict()
            torch.save(save, ckpt_dir / "best.pt")

    # Final
    save = {"ac_state": ac.state_dict(), "episode_metrics": episode_metrics, "hp": HP}
    if l3_trainer:
        save["l3_actor"] = l3_trainer.actor.state_dict()
        save["l3_critic_local"] = l3_trainer.critic_local.state_dict()
        save["l3_critic_global"] = l3_trainer.critic_global.state_dict()
    torch.save(save, ckpt_dir / "final.pt")

    # Summary
    last_n = min(300, len(episode_metrics))
    recent = episode_metrics[-last_n:]
    avg_dly = sum(e["mean_dep_delay"] for e in recent) / len(recent)
    avg_ont = sum(e["ontime_rate"] for e in recent) / len(recent)
    avg_rew = sum(e["total_reward"] for e in recent) / len(recent)
    rew_std = (sum((e["total_reward"]-avg_rew)**2 for e in recent)/len(recent))**0.5
    print(f"\n  Last {last_n} ep: reward={avg_rew:.1f} +/- {rew_std:.1f} "
          f"dly={avg_dly:.1f}s ontime={avg_ont:.2f}")
    print(f"  Checkpoint: {ckpt_dir / 'final.pt'}")

    csv_path = run_dir / "training_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=episode_metrics[0].keys())
        w.writeheader()
        w.writerows(episode_metrics)
    print(f"  CSV: {csv_path}")


if __name__ == "__main__":
    main()
