#!/usr/bin/env python3
"""Train L3 Worker only — loads a frozen L2 Manager, trains corridor speed controller.

Usage:
    python -m src.experiment.run_train_l3_only --ckpt <path> [--ep 2000]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.base_env import BaseRailwayEnv
from src.env.observation_builder import get_obs_dim, critic_input
from src.marl.actor_critic import Layer2ActorCritic, compact_global_state
from src.marl.mappo import MAPPOTrainer, AgentAction
from src.marl.rollout_buffer import RolloutBuffer, AgentTransition
from src.marl.l3_worker import (
    CorridorActor, CorridorCritic, L3WorkerTrainer, L3_OBS_DIM,
)
from src.control.resource_manager import build_conflict_graph

HP = {
    "obs_dim": 29,
    "action_dim": 45,
    "k_local": 1,
    "hidden_dim": 128,
    "l3_hidden_dim": 64,
    "lr": 3e-4,
    "l3_lr": 1e-3,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.2,
    "ppo_epochs": 5,
    "minibatch_size": 256,
    "entropy_coef_start": 0.01,
    "entropy_coef_end": 0.005,
    "l3_entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 1.0,
    "rollout_steps": 512,
}


def _build_critic_inputs(obs, requests):
    conflict_graph = build_conflict_graph(requests)
    group_obs: Dict[str, List[float]] = {}
    for agent_id in obs:
        group_obs[agent_id] = critic_input(
            agent_id, obs, conflict_graph, k_max=HP["k_local"])
    return group_obs


def train_l3_only(
    env_factory,
    args,
    obs_dim: int = 29,
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    # L2 Manager — load from checkpoint, freeze
    ac = Layer2ActorCritic(obs_dim=obs_dim, k_local=HP["k_local"],
                           hidden_dim=HP["hidden_dim"])
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ckpt_hp = ckpt.get("hp", {})
    # Load L2 state dict from checkpoint (ac_state for both flat and hierarchical)
    ac.load_state_dict(ckpt["ac_state"])
    for param in ac.parameters():
        param.requires_grad = False
    ac.eval()
    print(f"Loaded L2 Manager from: {args.ckpt}")
    print(f"  checkpoint episode: {ckpt.get('episode', '?')}")

    # L2 trainer (inference-only, no updates)
    l2_trainer = MAPPOTrainer(
        ac, lr=HP["lr"], gamma=HP["gamma"], gae_lambda=HP["gae_lambda"],
        clip_ratio=HP["clip_ratio"], ppo_epochs=HP["ppo_epochs"],
        minibatch_size=HP["minibatch_size"], entropy_coef=HP["entropy_coef_start"],
        value_coef=HP["value_coef"], max_grad_norm=HP["max_grad_norm"],
        total_episodes=args.episodes, device=device)

    # L3 Worker — train from scratch
    l3_actor = CorridorActor(obs_dim=L3_OBS_DIM, hidden_dim=HP["l3_hidden_dim"])
    l3_critic = CorridorCritic(obs_dim=L3_OBS_DIM, hidden_dim=HP["l3_hidden_dim"])
    l3_trainer = L3WorkerTrainer(
        l3_actor, l3_critic, lr=HP["l3_lr"], gamma=HP["gamma"],
        gae_lambda=HP["gae_lambda"], clip_ratio=HP["clip_ratio"],
        ppo_epochs=HP["ppo_epochs"], entropy_coef=HP["l3_entropy_coef"],
        value_coef=HP["value_coef"], max_grad_norm=HP["max_grad_norm"],
        device=device)

    # Checkpoint dir
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "output" / f"{args.env}_l3_only" / run_id
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    episode_metrics: List[Dict] = []
    best_reward = -float("inf")
    n_episodes = args.episodes

    for ep in range(n_episodes):
        l2_trainer.advance_episode()

        env = env_factory(seed=ep, l3_worker=l3_trainer)
        obs, info = env.reset()

        ep_reward: Dict[str, float] = defaultdict(float)
        ep_done = False
        ep_total_granted = 0
        ep_total_ontime_dep = 0
        ep_total_dep_delay = 0.0
        l3_update_count = 0

        # L3 speed stats
        l3_cruise_count = 0
        l3_fast_count = 0
        l3_slow_count = 0
        l3_total = 0

        while not ep_done:
            active_ids = info.get("active_train_ids", [])
            masks = info.get("action_masks", {})
            requests = info.get("requests", [])
            group_obs = _build_critic_inputs(obs, requests)
            gs_vec = compact_global_state(info.get("metrics", {}))

            # L2 Manager: sample actions (no gradient)
            with torch.no_grad():
                agent_actions = l2_trainer.collect_step(
                    obs, masks, group_obs, gs_vec, device=device)
            action_dict = {aid: (a.platform, a.entry) for aid, a in agent_actions.items()}

            next_obs, reward, terminated, truncated, info = env.step(
                action_dict, lambda_team=l2_trainer.lam_team)
            ep_done = terminated or truncated
            metrics = info.get("metrics", {})
            ep_total_granted += metrics.get("n_granted", 0)
            ep_total_ontime_dep += metrics.get("n_ontime_dep", 0)
            ep_total_dep_delay += metrics.get("total_dep_delay", 0.0)

            # Count L3 speed choices from env
            l3_choices = info.get("l3_speed_choices", {})
            for _, spd in l3_choices.items():
                l3_total += 1
                if spd == 0:
                    l3_cruise_count += 1
                elif spd == 1:
                    l3_fast_count += 1
                elif spd == 2:
                    l3_slow_count += 1

            for tid, r in reward.items():
                ep_reward[tid] += r

            obs = next_obs

            # L3 PPO update
            if len(l3_trainer.buffer) >= 64:
                l3_trainer.update()
                l3_update_count += 1

        # ---- end of episode ----
        if len(l3_trainer.buffer) >= 16:
            l3_trainer.update()
            l3_update_count += 1

        total_r = sum(ep_reward.values())
        mean_r = total_r / max(len(ep_reward), 1)
        ep_mean_dep = ep_total_dep_delay / max(ep_total_granted, 1)
        ep_ontime = ep_total_ontime_dep / max(ep_total_granted, 1)
        l3_cruise_pct = 100 * l3_cruise_count / max(l3_total, 1)
        l3_fast_pct = 100 * l3_fast_count / max(l3_total, 1)

        ep_summary = {
            "episode": ep,
            "total_reward": total_r,
            "mean_reward": mean_r,
            "mean_dep_delay": ep_mean_dep,
            "ontime_rate": ep_ontime,
            "l3_updates": l3_update_count,
            "l3_cruise_pct": l3_cruise_pct,
            "l3_fast_pct": l3_fast_pct,
        }
        episode_metrics.append(ep_summary)

        if (ep + 1) % max(1, n_episodes // 20) == 0 or ep < 5:
            print(
                f"  ep {ep:>5} | reward={mean_r:>7.2f} | "
                f"dep_dly={ep_mean_dep:>6.1f}s | ontime={ep_ontime:.2f} | "
                f"l3_upd={l3_update_count:>2} | "
                f"l3 cruise={l3_cruise_pct:.0f}% fast={l3_fast_pct:.0f}% | "
                f"{info.get('episode_status', '?'):>12}"
            )

        if mean_r > best_reward:
            best_reward = mean_r
            torch.save(
                {"l3_actor": l3_actor.state_dict(),
                 "l3_critic": l3_critic.state_dict(),
                 "episode": ep, "hp": HP},
                ckpt_dir / "best.pt",
            )

        # Log L3 update stats every 100 episodes
        if (ep + 1) % 100 == 0:
            l3_stats = l3_trainer.get_last_stats()
            if l3_stats:
                stats_str = "  ".join(f"{k}={v:.4f}" for k, v in l3_stats.items())
                print(f"           L3 stats: {stats_str}")

    # ---- final save ----
    torch.save(
        {"l3_actor": l3_actor.state_dict(),
         "l3_critic": l3_critic.state_dict(),
         "episode_metrics": episode_metrics, "hp": HP},
        ckpt_dir / "final.pt",
    )

    # ---- summary ----
    last_n = min(300, len(episode_metrics))
    recent = episode_metrics[-last_n:]
    recent_dep = [e["mean_dep_delay"] for e in recent]
    recent_ontime = [e["ontime_rate"] for e in recent]
    recent_fast = [e["l3_fast_pct"] for e in recent]
    print(f"\n{'='*60}")
    print(f"  L3-Only Training Complete")
    print(f"{'='*60}")
    print(f"  Last {last_n} ep: dep_delay={sum(recent_dep)/len(recent_dep):.1f}s  "
          f"ontime={sum(recent_ontime)/len(recent_ontime):.2f}  "
          f"l3_fast={sum(recent_fast)/len(recent_fast):.1f}%")
    print(f"  Checkpoint: {ckpt_dir / 'final.pt'}")

    # CSV
    csv_path = run_dir / "training_metrics.csv"
    if episode_metrics:
        fieldnames = list(episode_metrics[0].keys())
        for m in episode_metrics:
            for k in list(m.keys()):
                if k not in fieldnames:
                    fieldnames.append(k)
        for m in episode_metrics:
            for k in fieldnames:
                m.setdefault(k, "")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(episode_metrics)
        print(f"  CSV: {csv_path}")

    return {"best_reward": best_reward, "final_metrics": recent_dep}


def main():
    parser = argparse.ArgumentParser(description="Train L3 Worker with frozen L2 Manager")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to L2 checkpoint (.pt)")
    parser.add_argument("--env", type=str, default="five_station")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vt", action="store_true", help="Enable VT disturbance")
    parser.add_argument("--delay", action="store_true", help="Enable initial delay")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    from src.env.five_station_env import FiveStationEnv
    from src.experiment.scenario_loader import load_scenario

    cfg = load_scenario(scenario_name="five_station")
    obs_dim = get_obs_dim(cfg)

    def env_factory(seed: int, l3_worker=None) -> FiveStationEnv:
        return FiveStationEnv(
            cfg=cfg, seed=seed, batch_window_sec=120.0,
            enable_vt=args.vt, enable_initial_delay=args.delay,
            l3_worker=l3_worker,
        )

    train_l3_only(env_factory, args, obs_dim=obs_dim)


if __name__ == "__main__":
    main()
