#!/usr/bin/env python3
"""Resume MAPPO training from a checkpoint."""
import sys, os, csv, json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment.run_train import (
    HP, _build_critic_inputs, compact_global_state,
)
from src.marl.rollout_buffer import RolloutBuffer, AgentTransition
from src.marl.actor_critic import Layer2ActorCritic
from src.marl.mappo import MAPPOTrainer
from src.experiment.scenario_loader import load_scenario
from src.env.five_station_env import FiveStationEnv


def resume(checkpoint_path: str, extra_episodes: int, device: str = "cuda"):
    ckpt_path = Path(checkpoint_path)
    ckpt = torch.load(ckpt_path, map_location=device)

    ac_state = ckpt["ac_state"]
    hp = ckpt.get("hp", HP)
    start_ep = ckpt.get("episode", 0)

    print(f"Resuming from episode {start_ep}")
    print(f"Adding {extra_episodes} episodes (total: {start_ep + extra_episodes})")

    # Rebuild model
    obs_dim = hp.get("obs_dim", 25)
    k_local = hp.get("k_local", 1)
    hidden_dim = hp.get("hidden_dim", 128)

    ac = Layer2ActorCritic(
        obs_dim=obs_dim, k_local=k_local, hidden_dim=hidden_dim)
    ac.load_state_dict(ac_state)
    ac.to(device)

    trainer = MAPPOTrainer(
        ac,
        lr=hp["lr"], gamma=hp["gamma"], gae_lambda=hp["gae_lambda"],
        clip_ratio=hp["clip_ratio"], ppo_epochs=hp["ppo_epochs"],
        minibatch_size=hp["minibatch_size"],
        entropy_coef=hp["entropy_coef_start"],
        value_coef=hp["value_coef"], max_grad_norm=hp["max_grad_norm"],
        total_episodes=start_ep + extra_episodes, device=device,
    )

    buffer = RolloutBuffer(gamma=hp["gamma"], gae_lambda=hp["gae_lambda"])

    # Output dir
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "output" / "five_station" / run_id
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_scenario(scenario_name="five_station")

    episode_metrics: List[Dict] = []
    total_steps = 0
    best_reward = -float("inf")

    total_eps = start_ep + extra_episodes
    rollout_steps = hp.get("rollout_steps", 512)

    print(f"\n{'='*60}")
    print(f"  Resume Training — {extra_episodes} additional episodes")
    print(f"  obs_dim={obs_dim}, k_local={k_local}, rollout={rollout_steps}")
    print(f"{'='*60}\n")

    for local_ep in range(extra_episodes):
        ep = start_ep + local_ep
        trainer.advance_episode()
        # Entropy decay: 0.01 -> hp["entropy_coef_end"] over first 30%
        decay_eps = total_eps * 0.3
        entropy_progress = min(ep / max(decay_eps, 1), 1.0)
        trainer.entropy_coef = (
            hp["entropy_coef_start"]
            + (hp["entropy_coef_end"] - hp["entropy_coef_start"]) * entropy_progress
        )

        env = FiveStationEnv(cfg=cfg, seed=ep, batch_window_sec=120.0)
        obs, info = env.reset()

        ep_reward: Dict[str, float] = defaultdict(float)
        ep_steps = 0
        ep_done = False

        while not ep_done:
            active_ids = info.get("active_train_ids", [])
            masks = info.get("action_masks", {})
            requests = info.get("requests", [])
            group_obs = _build_critic_inputs(obs, requests)
            gs_vec = compact_global_state(info.get("metrics", {}))

            agent_actions = trainer.collect_step(
                obs, masks, group_obs, gs_vec, device=device)

            action_dict = {aid: (a.platform, a.entry) for aid, a in agent_actions.items()}
            next_obs, reward, terminated, truncated, info = env.step(action_dict)
            ep_done = terminated or truncated
            metrics = info.get("metrics", {})
            reward_breakdown = info.get("reward_breakdown", {})

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
                buffer.add_step(step_transitions,
                    torch.tensor(gs_vec, dtype=torch.float32))

            for tid, r in reward.items():
                ep_reward[tid] += r

            obs = next_obs
            ep_steps += 1
            total_steps += 1

            if buffer.num_steps >= rollout_steps:
                final_v_local = {}
                final_v_global = 0.0
                if not ep_done:
                    for agent_id in active_ids:
                        if agent_id in agent_actions:
                            final_v_local[agent_id] = agent_actions[agent_id].value_local
                            final_v_global = agent_actions[agent_id].value_global
                buffer.compute_gae(final_v_local, final_v_global)
                trainer.update(buffer)
                buffer.clear()

        if info.get("episode_status") == "deadlock":
            DL_PENALTY = 400.0
            for tid in env.cfg.trains:
                ep_reward[tid] += (-DL_PENALTY)

        m = info.get("metrics", {})
        total_r = sum(ep_reward.values())
        mean_r = total_r / max(len(ep_reward), 1)

        ep_summary = {
            "episode": ep,
            "steps": ep_steps,
            "status": info.get("episode_status", "?"),
            "total_reward": total_r,
            "mean_reward": mean_r,
            "mean_dep_delay": m.get("mean_dep_delay", 0),
            "mean_arr_delay": m.get("mean_arr_delay", 0),
            "ontime_rate": m.get("ontime_rate", 0),
            "n_ontime_dep": m.get("n_ontime_dep", 0),
            "n_granted": m.get("n_granted", 0),
            "n_held": m.get("n_held", 0),
            "lam_team": trainer.lam_team,
            "illegal_actions": 0,
        }
        episode_metrics.append(ep_summary)

        if (local_ep + 1) % max(1, extra_episodes // 20) == 0 or local_ep < 5:
            print(
                f"  ep {ep:>5} | reward={mean_r:>7.2f} | "
                f"dep_dly={m.get('mean_dep_delay', 0):>6.1f}s | "
                f"ontime={m.get('ontime_rate', 0):.2f} | "
                f"steps={ep_steps:>2} | ng={m.get('n_granted',0)} | "
                f"{info.get('episode_status', '?'):>12}"
            )

        if mean_r > best_reward:
            best_reward = mean_r
            torch.save(
                {"ac_state": ac.state_dict(), "episode": ep + 1, "hp": hp},
                ckpt_dir / "best.pt",
            )

    # Final save
    torch.save(
        {"ac_state": ac.state_dict(), "episode": total_eps, "hp": hp,
         "episode_metrics": episode_metrics},
        ckpt_dir / "final.pt",
    )

    # Save CSV
    csv_path = run_dir / "training_metrics.csv"
    if episode_metrics:
        fieldnames = list(episode_metrics[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(episode_metrics)

    with open(run_dir / "training_metrics.json", "w") as f:
        json.dump(episode_metrics, f, indent=2)

    # Summary
    last_n = min(100, len(episode_metrics))
    recent = episode_metrics[-last_n:]
    recent_r = [e["mean_reward"] for e in recent]
    recent_dep = [e["mean_dep_delay"] for e in recent]
    recent_ot = [e["ontime_rate"] for e in recent]

    print(f"\n{'='*60}")
    print(f"  Training Complete — Added {extra_episodes} episodes")
    print(f"{'='*60}")
    print(f"  Last {last_n} episodes:")
    print(f"    reward:      avg={np.mean(recent_r):.2f}")
    print(f"    dep_delay:   avg={np.mean(recent_dep):.1f}s")
    print(f"    ontime:      avg={np.mean(recent_ot)*100:.1f}%")
    print(f"  CSV: {csv_path}")

    return run_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to final.pt checkpoint")
    parser.add_argument("--episodes", type=int, default=3000, help="Extra episodes")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    resume(args.checkpoint, args.episodes, device=device)
