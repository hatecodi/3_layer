#!/usr/bin/env python3
"""Hierarchical MAPPO training — alternating L2 Manager / L3 Worker phases.

Phase A: Train L2 Manager only (L3 rule-based cruise)     — 3000 ep
Phase B: Freeze L2, train L3 Worker                       — 2000 ep
Phase C: Joint fine-tune (alternating L2/L3 per episode)  — 2000 ep

Usage:
    python -m src.experiment.run_train_hierarchical --env five_station
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.base_env import BaseRailwayEnv
from src.env.observation_builder import get_obs_dim, critic_input
from src.marl.actor_critic import (
    Layer2ActorCritic,
    compact_global_state,
)
from src.marl.mappo import MAPPOTrainer, AgentAction
from src.marl.rollout_buffer import RolloutBuffer, AgentTransition
from src.marl.l3_worker import (
    CorridorActor, CorridorCritic, L3WorkerTrainer,
    L3_OBS_DIM,
)
from src.control.resource_manager import build_conflict_graph

# ---------------------------------------------------------------------------
# hyperparameters
# ---------------------------------------------------------------------------

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
    "entropy_coef_end": 0.003,
    "l3_entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 1.0,
    "rollout_steps": 512,
}

PHASES = {
    "A": {"episodes": 3000, "train_l2": True, "train_l3": False},
    "B": {"episodes": 2000, "train_l2": False, "train_l3": True},
    "C": {"episodes": 2000, "train_l2": True, "train_l3": True},
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _build_critic_inputs(obs, requests):
    conflict_graph = build_conflict_graph(requests)
    group_obs: Dict[str, List[float]] = {}
    for agent_id in obs:
        group_obs[agent_id] = critic_input(
            agent_id, obs, conflict_graph, k_max=HP["k_local"])
    return group_obs


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train_hierarchical(
    env_factory: Callable[[int, Optional[object]], BaseRailwayEnv],
    args,
    obs_dim: int = 29,
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    # L2 Manager (existing architecture)
    ac = Layer2ActorCritic(obs_dim=obs_dim, k_local=HP["k_local"],
                           hidden_dim=HP["hidden_dim"])
    l2_trainer = MAPPOTrainer(
        ac, lr=HP["lr"], gamma=HP["gamma"], gae_lambda=HP["gae_lambda"],
        clip_ratio=HP["clip_ratio"], ppo_epochs=HP["ppo_epochs"],
        minibatch_size=HP["minibatch_size"], entropy_coef=HP["entropy_coef_start"],
        value_coef=HP["value_coef"], max_grad_norm=HP["max_grad_norm"],
        total_episodes=sum(p["episodes"] for p in PHASES.values()),
        device=device)
    l2_buffer = RolloutBuffer(gamma=HP["gamma"], gae_lambda=HP["gae_lambda"])

    # L3 Worker (shared corridor speed controller)
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
    run_dir = PROJECT_ROOT / "output" / f"{args.env}_hierarchical" / run_id
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    episode_metrics: List[Dict] = []
    total_ep = 0
    best_reward = -float("inf")

    for phase_name, phase_cfg in PHASES.items():
        print(f"\n{'='*60}")
        print(f"  Phase {phase_name}: {'L2' if phase_cfg['train_l2'] else ''}"
              f"{' + ' if phase_cfg['train_l2'] and phase_cfg['train_l3'] else ''}"
              f"{'L3' if phase_cfg['train_l3'] else ''}"
              f"  ({phase_cfg['episodes']} ep)")
        print(f"{'='*60}\n")

        for ep in range(phase_cfg["episodes"]):
            l2_trainer.advance_episode()

            # Entropy decay for L2
            entropy_progress = min(total_ep / max(sum(p["episodes"] for p in PHASES.values()) * 0.3, 1), 1.0)
            l2_trainer.entropy_coef = (
                HP["entropy_coef_start"]
                + (HP["entropy_coef_end"] - HP["entropy_coef_start"]) * entropy_progress
            )

            # Attach L3 worker only during L3 training phases
            worker = l3_trainer if phase_cfg["train_l3"] else None
            env = env_factory(seed=total_ep, l3_worker=worker)
            obs, info = env.reset()

            ep_reward: Dict[str, float] = defaultdict(float)
            ep_done = False
            ep_total_granted = 0
            ep_total_ontime_dep = 0
            ep_total_dep_delay = 0.0
            l3_update_count = 0

            while not ep_done:
                active_ids = info.get("active_train_ids", [])
                masks = info.get("action_masks", {})
                requests = info.get("requests", [])
                group_obs = _build_critic_inputs(obs, requests)
                gs_vec = compact_global_state(info.get("metrics", {}))

                # L2 Manager: sample actions
                agent_actions = l2_trainer.collect_step(
                    obs, masks, group_obs, gs_vec, device=device)
                action_dict = {aid: (a.platform, a.entry) for aid, a in agent_actions.items()}

                next_obs, reward, terminated, truncated, info = env.step(action_dict)
                ep_done = terminated or truncated
                metrics = info.get("metrics", {})
                ep_total_granted += metrics.get("n_granted", 0)
                ep_total_ontime_dep += metrics.get("n_ontime_dep", 0)
                ep_total_dep_delay += metrics.get("total_dep_delay", 0.0)
                reward_breakdown = info.get("reward_breakdown", {})

                # L2 transitions
                step_transitions: List[AgentTransition] = []
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
                        platform=a.platform,
                        entry=a.entry,
                        logprob=a.logprob,
                        platform_mask=p_mask,
                        entry_mask=e_mask,
                        value_local=a.value_local,
                        value_global=a.value_global,
                        reward_local=r_local,
                        reward_global=r_global,
                        done=ep_done,
                        global_state=torch.tensor(gs_vec, dtype=torch.float32),
                        local_group_obs=torch.tensor(
                            group_obs.get(agent_id, obs[agent_id]),
                            dtype=torch.float32),
                    ))

                if step_transitions:
                    l2_buffer.add_step(
                        step_transitions,
                        torch.tensor(gs_vec, dtype=torch.float32),
                    )

                for tid, r in reward.items():
                    ep_reward[tid] += r

                obs = next_obs

                # L2 PPO update
                if l2_buffer.num_steps >= HP["rollout_steps"] and phase_cfg["train_l2"]:
                    final_v = {}
                    if not ep_done:
                        for agent_id in active_ids:
                            if agent_id in agent_actions:
                                final_v[agent_id] = agent_actions[agent_id].value_local
                    l2_buffer.compute_gae(final_v, 0.0)
                    l2_trainer.update(l2_buffer)
                    l2_buffer.clear()

                # L3 PPO update (buffer filled inside env._advance_trains)
                if phase_cfg["train_l3"] and len(l3_trainer.buffer) >= 32:
                    l3_trainer.update()
                    l3_update_count += 1

            # ---- end of episode ----
            if phase_cfg["train_l3"] and len(l3_trainer.buffer) > 0:
                l3_trainer.update()
                l3_update_count += 1

            total_r = sum(ep_reward.values())
            mean_r = total_r / max(len(ep_reward), 1)
            ep_mean_dep = ep_total_dep_delay / max(ep_total_granted, 1)
            ep_ontime = ep_total_ontime_dep / max(ep_total_granted, 1)

            ep_summary = {
                "episode": total_ep,
                "phase": phase_name,
                "status": info.get("episode_status", "?"),
                "total_reward": total_r,
                "mean_reward": mean_r,
                "mean_dep_delay": ep_mean_dep,
                "ontime_rate": ep_ontime,
                "l3_updates": l3_update_count,
            }
            episode_metrics.append(ep_summary)

            if (total_ep + 1) % max(1, sum(p["episodes"] for p in PHASES.values()) // 20) == 0 or total_ep < 5:
                print(
                    f"  ep {total_ep:>5} [{phase_name}] | reward={mean_r:>7.2f} | "
                    f"dep_dly={ep_mean_dep:>6.1f}s | ontime={ep_ontime:.2f} | "
                    f"l3_upd={l3_update_count:>2} | "
                    f"{info.get('episode_status', '?'):>12}"
                )

            if mean_r > best_reward:
                best_reward = mean_r
                torch.save(
                    {"ac_state": ac.state_dict(), "l3_actor": l3_actor.state_dict(),
                     "l3_critic": l3_critic.state_dict(), "episode": total_ep, "hp": HP},
                    ckpt_dir / "best.pt",
                )

            total_ep += 1

    # ---- final save ----
    torch.save(
        {"ac_state": ac.state_dict(), "l3_actor": l3_actor.state_dict(),
         "l3_critic": l3_critic.state_dict(), "hp": HP,
         "episode_metrics": episode_metrics},
        ckpt_dir / "final.pt",
    )

    # ---- summary ----
    last_n = min(300, len(episode_metrics))
    recent = episode_metrics[-last_n:]
    recent_dep = [e["mean_dep_delay"] for e in recent]
    print(f"\n{'='*60}")
    print(f"  Training Complete")
    print(f"{'='*60}")
    print(f"  Last {last_n} episodes: dep_delay={sum(recent_dep)/len(recent_dep):.1f}s")
    print(f"  Checkpoint: {ckpt_dir / 'final.pt'}")

    # CSV
    csv_path = run_dir / "training_metrics.csv"
    if episode_metrics:
        fieldnames = list(episode_metrics[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(episode_metrics)
        print(f"  CSV: {csv_path}")

    return {"best_reward": best_reward, "final_metrics": recent_dep}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Hierarchical MAPPO training")
    parser.add_argument("--env", type=str, default="five_station")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--lr", type=float, default=HP["lr"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vt", action="store_true", help="Enable VT disturbance")
    parser.add_argument("--delay", action="store_true", help="Enable initial delay")
    args = parser.parse_args()

    HP["lr"] = args.lr
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

    train_hierarchical(env_factory, args, obs_dim=obs_dim)


if __name__ == "__main__":
    main()
