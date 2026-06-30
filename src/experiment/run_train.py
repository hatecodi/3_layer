#!/usr/bin/env python3
"""MAPPO training script — env-agnostic.

Usage:
    python -m src.experiment.run_train --env two_station --episodes 2000
    python -m src.experiment.run_train --env five_station --episodes 2000
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

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


# ---------------------------------------------------------------------------
# hyperparameters
# ---------------------------------------------------------------------------

HP = {
    "obs_dim": 34,                       # overridden per env
    "action_dim": 8,                     # K schedule options
    "k_local": 1,                       # LocalCritic: agent + 1 neighbour
    "hidden_dim": 128,
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.2,
    "ppo_epochs": 5,
    "minibatch_size": 256,
    "entropy_coef_start": 0.01,          # early exploration
    "entropy_coef_end": 0.003,            # multi-agent stability + convergence balance
    "value_coef": 0.5,
    "max_grad_norm": 1.0,
    "rollout_steps": 512,
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _build_critic_inputs(
    obs: Dict[str, List[float]],
    requests: List,
) -> Dict[str, List[float]]:
    """Build local group observations for each active agent."""
    from src.control.resource_manager import build_conflict_graph

    conflict_graph = build_conflict_graph(requests)
    group_obs: Dict[str, List[float]] = {}
    for agent_id in obs:
        group_obs[agent_id] = critic_input(
            agent_id, obs, conflict_graph, k_max=HP["k_local"])
    return group_obs


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train(env_factory: Callable[[int], BaseRailwayEnv], args, obs_dim: int = 25) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")
    print(f"Hyperparameters: {json.dumps(HP, indent=2)}")

    # Model
    ac = Layer2ActorCritic(
        obs_dim=obs_dim,
        
        k_local=HP["k_local"],
        hidden_dim=HP["hidden_dim"],
    )
    trainer = MAPPOTrainer(
        ac,
        lr=HP["lr"],
        gamma=HP["gamma"],
        gae_lambda=HP["gae_lambda"],
        clip_ratio=HP["clip_ratio"],
        ppo_epochs=HP["ppo_epochs"],
        minibatch_size=HP["minibatch_size"],
        entropy_coef=HP["entropy_coef_start"],
        value_coef=HP["value_coef"],
        max_grad_norm=HP["max_grad_norm"],
        total_episodes=args.episodes,
        device=device,
    )

    buffer = RolloutBuffer(gamma=HP["gamma"], gae_lambda=HP["gae_lambda"])

    # Checkpoint dir
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "output" / args.env / run_id
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir

    # Metrics tracking
    episode_metrics: List[Dict] = []
    total_steps = 0
    best_reward = -float("inf")

    print(f"\n{'='*60}")
    print(f"  Training — {args.episodes} episodes, rollout={HP['rollout_steps']} steps")
    print(f"{'='*60}\n")

    for ep in range(args.episodes):
        trainer.advance_episode()
        # Entropy decay: 0.01 → 0.001 over first 30% of episodes
        entropy_progress = min(ep / max(args.episodes * 0.3, 1), 1.0)
        trainer.entropy_coef = (
            HP["entropy_coef_start"]
            + (HP["entropy_coef_end"] - HP["entropy_coef_start"]) * entropy_progress
        )
        env = env_factory(seed=ep)
        obs, info = env.reset()

        ep_reward: Dict[str, float] = defaultdict(float)
        ep_steps = 0
        ep_done = False
        # Accumulate metrics across all steps (not just final step snapshot)
        ep_total_granted = 0
        ep_total_ontime_dep = 0
        ep_total_ontime_arr = 0
        ep_total_dep_delay = 0.0
        ep_total_arr_delay = 0.0
        ep_total_held = 0

        while not ep_done:
            # ---- collect step ----
            active_ids = info.get("active_train_ids", [])
            masks = info.get("action_masks", {})

            # Build critic inputs
            requests = info.get("requests", [])
            group_obs = _build_critic_inputs(obs, requests)
            gs_vec = compact_global_state(info.get("metrics", {}))

            # Sample actions
            agent_actions = trainer.collect_step(
                obs, masks, group_obs, gs_vec, device=device)

            # Build action_dict
            action_dict = {aid: a.option for aid, a in agent_actions.items()}

            # Step
            next_obs, reward, terminated, truncated, info = env.step(action_dict)
            ep_done = terminated or truncated
            metrics = info.get("metrics", {})
            # Accumulate episode-level metrics
            ep_total_granted += metrics.get("n_granted", 0)
            ep_total_ontime_dep += metrics.get("n_ontime_dep", 0)
            ep_total_ontime_arr += metrics.get("n_ontime_arr", 0)
            ep_total_dep_delay += metrics.get("total_dep_delay", 0.0)
            ep_total_arr_delay += metrics.get("total_arr_delay", 0.0)
            ep_total_held += metrics.get("n_held", 0)
            reward_breakdown = info.get("reward_breakdown", {})

            # Record transitions
            step_transitions: List[AgentTransition] = []
            for agent_id in active_ids:
                if agent_id not in agent_actions:
                    continue
                a = agent_actions[agent_id]
                r_local = reward_breakdown.get(agent_id, {}).get("local", 0.0)
                r_global = reward_breakdown.get(agent_id, {}).get("global_", 0.0)

                # Flat option mask (length K)
                m = masks.get(agent_id, [])
                opt_mask = torch.tensor(list(m), dtype=torch.bool)

                step_transitions.append(AgentTransition(
                    agent_id=agent_id,
                    obs=torch.tensor(obs[agent_id], dtype=torch.float32),
                    option=a.option,
                    logprob=a.logprob,
                    option_mask=opt_mask,
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
                buffer.add_step(
                    step_transitions,
                    torch.tensor(gs_vec, dtype=torch.float32),
                )

            for tid, r in reward.items():
                ep_reward[tid] += r

            obs = next_obs
            ep_steps += 1
            total_steps += 1

            # PPO update when buffer reaches rollout_steps
            if buffer.num_steps >= HP["rollout_steps"]:
                # Compute final values for GAE (non-terminal steps)
                final_v_local = {}
                final_v_global = 0.0
                if not ep_done:
                    for agent_id in active_ids:
                        if agent_id in agent_actions:
                            final_v_local[agent_id] = agent_actions[agent_id].value_local
                            final_v_global = agent_actions[agent_id].value_global

                buffer.compute_gae(final_v_local, final_v_global)
                stats = trainer.update(buffer)
                buffer.clear()
            else:
                stats = {}

        # ---- end of episode ----
        # V4: deadlock penalty applied to ALL agents at episode level
        if info.get("episode_status") == "deadlock":
            DL_PENALTY = 400.0
            for tid in env.cfg.trains:
                ep_reward[tid] += (-DL_PENALTY)

        total_r = sum(ep_reward.values())
        mean_r = total_r / max(len(ep_reward), 1)

        # Episode-level aggregates (not final-step snapshot)
        ep_mean_dep = ep_total_dep_delay / max(ep_total_granted, 1)
        ep_mean_arr = ep_total_arr_delay / max(ep_total_granted, 1)
        ep_ontime = ep_total_ontime_dep / max(ep_total_granted, 1)

        ep_summary = {
            "episode": ep,
            "steps": ep_steps,
            "status": info.get("episode_status", "?"),
            "total_reward": total_r,
            "mean_reward": mean_r,
            "mean_dep_delay": ep_mean_dep,
            "mean_arr_delay": ep_mean_arr,
            "ontime_rate": ep_ontime,
            "n_ontime_dep": ep_total_ontime_dep,
            "n_granted": ep_total_granted,
            "n_held": ep_total_held,
            "lam_team": trainer.lam_team,
            "illegal_actions": 0,
        }
        episode_metrics.append(ep_summary)

        # Log
        if (ep + 1) % max(1, args.episodes // 20) == 0 or ep < 5:
            print(
                f"  ep {ep:>5} | reward={mean_r:>7.2f} | "
                f"dep_dly={ep_mean_dep:>6.1f}s | "
                f"arr_dly={ep_mean_arr:>6.1f}s | "
                f"ontime={ep_ontime:.2f} | "
                f"steps={ep_steps:>2} | λ={trainer.lam_team:.3f} | "
                f"{info.get('episode_status', '?'):>12}"
            )

        # Save best
        if mean_r > best_reward:
            best_reward = mean_r
            torch.save(
                {"ac_state": ac.state_dict(), "episode": ep, "hp": HP},
                ckpt_dir / "best.pt",
            )

    # ---- final save ----
    torch.save(
        {"ac_state": ac.state_dict(), "episode": args.episodes, "hp": HP,
         "episode_metrics": episode_metrics},
        ckpt_dir / "final.pt",
    )

    # ---- summary ----
    last_n = min(100, len(episode_metrics))
    recent = episode_metrics[-last_n:]
    recent_r = [e["mean_reward"] for e in recent]
    recent_dep = [e["mean_dep_delay"] for e in recent]
    recent_arr = [e["mean_arr_delay"] for e in recent]

    print(f"\n{'='*60}")
    print(f"  Training Complete")
    print(f"{'='*60}")
    print(f"  Last {last_n} episodes:")
    print(f"    reward:      avg={sum(recent_r)/len(recent_r):.2f}")
    print(f"    dep_delay:   avg={sum(recent_dep)/len(recent_dep):.1f}s")
    print(f"    arr_delay:   avg={sum(recent_arr)/len(recent_arr):.1f}s")
    print(f"  Best reward: {best_reward:.2f}")
    print(f"  Checkpoint: {ckpt_dir / 'final.pt'}")
    print()

    # Save metrics CSV (primary output for plotting)
    csv_path = log_dir / "training_metrics.csv"
    if episode_metrics:
        fieldnames = list(episode_metrics[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(episode_metrics)
        print(f"  CSV: {csv_path}")

    # Also save JSON (superset)
    with open(log_dir / "training_metrics.json", "w") as f:
        json.dump(episode_metrics, f, indent=2)

    return {"best_reward": best_reward, "final_metrics": recent}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MAPPO training")
    parser.add_argument("--env", type=str, default="two_station",
                        choices=["two_station", "five_station"])
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--rollout-steps", type=int, default=HP["rollout_steps"])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--lr", type=float, default=HP["lr"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    HP["rollout_steps"] = args.rollout_steps
    HP["lr"] = args.lr
    torch.manual_seed(args.seed)

    if args.env == "two_station":
        from src.env.two_station_env import TwoStationEnv
        from src.experiment.scenario_loader import load_scenario

        cfg = load_scenario(scenario_name="two_station")
        from src.env.observation_builder import get_obs_dim
        obs_dim = get_obs_dim(cfg)

        def env_factory(seed: int) -> TwoStationEnv:
            return TwoStationEnv(seed=seed)

    elif args.env == "five_station":
        from src.env.five_station_env import FiveStationEnv
        from src.experiment.scenario_loader import load_scenario

        cfg = load_scenario(scenario_name="five_station")
        from src.env.observation_builder import get_obs_dim
        obs_dim = get_obs_dim(cfg)

        def env_factory(seed: int) -> FiveStationEnv:
            return FiveStationEnv(
                cfg=cfg,
                seed=seed,
                batch_window_sec=120.0,
            )

    train(env_factory, args,  obs_dim=obs_dim)


if __name__ == "__main__":
    main()



# ?? Post-training analysis ?????????????????????????????
