#!/usr/bin/env python3
"""Two-phase VT training: L2 (→0 conflicts) → L3 (frozen L2).

Phase 1: L2 warm-start + VT+delay, train until 0 platform conflicts
Phase 2: Freeze L2, train L3 Worker

Usage:
    python scripts/train_vt_staged.py --l2-ckpt <no-vt best.pt> [--l2-ep 3000] [--l3-ep 2000]
"""
from __future__ import annotations

import argparse, csv, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from src.control.resource_manager import build_conflict_graph, ResourceCalendar

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


@torch.no_grad()
def validate_conflicts(env_factory, l2_trainer, n_ep=10) -> Tuple[int, float, float]:
    """Return (violation_count, mean_delay, mean_ontime)."""
    total_violations = 0
    total_dly = 0.0
    total_ont = 0.0
    for seed in range(n_ep):
        l2_trainer.advance_episode()
        env = env_factory(seed + 9000)
        obs, info = env.reset()
        all_events = []
        ep_done = False
        while not ep_done:
            masks = info.get("action_masks", {})
            requests = info.get("requests", [])
            group_obs = _build_critic_inputs(obs, requests)
            gs_vec = compact_global_state(info.get("metrics", {}))
            agent_actions = l2_trainer.collect_step(
                obs, masks, group_obs, gs_vec, device=torch.device("cpu"))
            action_dict = {aid: (a.platform, a.entry) for aid, a in agent_actions.items()}
            for g in info.get("granted_windows", []):
                all_events.append({"station": env._journeys[g.train_id].current_station,
                                   "platform": g.platform, "train": g.train_id,
                                   "t_arr": g.W_plat[0], "t_dep": g.W_plat[1]})
            next_obs, _, terminated, truncated, info = env.step(
                action_dict, lambda_team=l2_trainer.lam_team)
            ep_done = terminated or truncated
            obs = next_obs
        # Check platform overlaps
        bookings = defaultdict(list)
        for e in all_events:
            bookings[(e["station"], e["platform"])].append(e)
        for evts in bookings.values():
            evts.sort(key=lambda e: e["t_arr"])
            for i in range(len(evts)):
                for j in range(i+1, len(evts)):
                    a, b = evts[i], evts[j]
                    if a["train"] == b["train"]:
                        continue
                    if b["t_arr"] - a["t_dep"] < 5.0:
                        total_violations += 1
        metrics = info.get("metrics", {})
        total_dly += metrics.get("mean_dep_delay", 0)
        total_ont += metrics.get("ontime_rate", 0)
    return total_violations, total_dly / n_ep, total_ont / n_ep


def train_l2_phase(env_factory, l2_trainer, l2_buffer, n_episodes, device, do_validate=True):
    """Train L2 only. Returns episode metrics and best reward."""
    episode_metrics = []
    best_reward = -float("inf")
    validation_interval = max(1, n_episodes // 10)

    for ep in range(n_episodes):
        l2_trainer.advance_episode()
        entropy_progress = min(ep / max(n_episodes * 0.3, 1), 1.0)
        l2_trainer.entropy_coef = (
            HP["entropy_coef_start"] + (HP["entropy_coef_end"] - HP["entropy_coef_start"]) * entropy_progress)

        env = env_factory(seed=ep)
        obs, info = env.reset()
        ep_done = False
        ep_reward = 0.0
        ep_granted = 0
        ep_ontime = 0
        ep_dep_dly = 0.0
        ep_held = 0

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
                p_mask = torch.zeros(4, dtype=torch.bool)
                e_mask = torch.zeros(6, dtype=torch.bool)
                if m is not None:
                    for j in range(min(len(m[0]), 4)):
                        p_mask[j] = bool(m[0][j])
                    for j in range(min(len(m[1]), 6)):
                        e_mask[j] = bool(m[1][j])
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

        if l2_buffer.num_steps > 0:
            l2_buffer.compute_gae({}, 0.0)
            l2_trainer.update(l2_buffer)
            l2_buffer.clear()

        mean_dep = ep_dep_dly / max(ep_granted, 1)
        ont_rate = ep_ontime / max(ep_granted, 1)
        episode_metrics.append({
            "episode": ep, "total_reward": ep_reward,
            "mean_dep_delay": mean_dep, "ontime_rate": ont_rate, "n_held": ep_held,
        })

        if (ep + 1) % validation_interval == 0 or ep < 5:
            if do_validate:
                vio, vdly, vont = validate_conflicts(env_factory, l2_trainer, n_ep=5)
                print(f"  ep {ep:>5} | reward={ep_reward:>8.1f} | "
                      f"dly={mean_dep:>5.1f}s | ont={ont_rate:.2f} | "
                      f"held={ep_held:>3} | violations={vio:>1}")
            else:
                print(f"  ep {ep:>5} | reward={ep_reward:>8.1f} | "
                      f"dly={mean_dep:>5.1f}s | ont={ont_rate:.2f} | held={ep_held:>3}")

        if ep_reward > best_reward:
            best_reward = ep_reward

    return episode_metrics, best_reward


def train_l3_phase(env_factory, l2_trainer, l3_trainer, n_episodes):
    """Train L3 only with frozen L2."""
    episode_metrics = []
    for ep in range(n_episodes):
        l2_trainer.advance_episode()
        env = env_factory(seed=ep + 50000, l3_worker=l3_trainer)
        obs, info = env.reset()
        ep_done = False
        ep_reward = 0.0
        ep_granted, ep_ontime, ep_dep_dly = 0, 0, 0.0
        l3_cruise, l3_fast, l3_slow, l3_total = 0, 0, 0, 0

        while not ep_done:
            masks = info.get("action_masks", {})
            requests = info.get("requests", [])
            group_obs = _build_critic_inputs(obs, requests)
            gs_vec = compact_global_state(info.get("metrics", {}))

            with torch.no_grad():
                agent_actions = l2_trainer.collect_step(obs, masks, group_obs, gs_vec, device=torch.device("cpu"))
            action_dict = {aid: (a.platform, a.entry) for aid, a in agent_actions.items()}

            next_obs, reward, terminated, truncated, info = env.step(
                action_dict, lambda_team=l2_trainer.lam_team)
            ep_done = terminated or truncated
            metrics = info.get("metrics", {})
            ep_granted += metrics.get("n_granted", 0)
            ep_ontime += metrics.get("n_ontime_dep", 0)
            ep_dep_dly += metrics.get("total_dep_delay", 0.0)

            l3_choices = info.get("l3_speed_choices", {})
            for _, spd in l3_choices.items():
                l3_total += 1
                if spd == 0: l3_cruise += 1
                elif spd == 1: l3_fast += 1
                elif spd == 2: l3_slow += 1

            for tid, r in reward.items():
                ep_reward += r
            obs = next_obs

            if len(l3_trainer.buffer) >= 64:
                l3_trainer.update()

        if len(l3_trainer.buffer) >= 16:
            l3_trainer.update()

        mean_dep = ep_dep_dly / max(ep_granted, 1)
        ont_rate = ep_ontime / max(ep_granted, 1)
        cruise_pct = 100 * l3_cruise / max(l3_total, 1)
        fast_pct = 100 * l3_fast / max(l3_total, 1)
        episode_metrics.append({
            "episode": ep, "total_reward": ep_reward,
            "mean_dep_delay": mean_dep, "ontime_rate": ont_rate,
            "l3_cruise_pct": cruise_pct, "l3_fast_pct": fast_pct,
        })

        if (ep + 1) % 100 == 0 or ep < 5:
            print(f"  ep {ep:>5} | reward={ep_reward:>8.1f} | "
                  f"dly={mean_dep:>5.1f}s | ont={ont_rate:.2f} | "
                  f"L3 cruise={cruise_pct:.0f}% fast={fast_pct:.0f}%")

    return episode_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l2-ckpt", type=str, required=True)
    parser.add_argument("--l2-ep", type=int, default=3000)
    parser.add_argument("--l3-ep", type=int, default=2000)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cpu")
    cfg = load_scenario(scenario_name="five_station")
    obs_dim = get_obs_dim(cfg)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "output" / "five_station_vt_staged" / run_id
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {run_dir}")

    def env_factory(seed: int, l3_worker=None):
        return FiveStationEnv(
            cfg=cfg, seed=seed, batch_window_sec=120.0,
            enable_vt=True, enable_initial_delay=True, l3_worker=l3_worker)

    # ---- Phase 1: L2 with VT ----
    print(f"\n{'='*60}")
    print(f"  Phase 1: L2 with VT+delay — warm-start from no-VT checkpoint")
    print(f"{'='*60}")

    l2_ckpt = torch.load(args.l2_ckpt, map_location=device, weights_only=False)
    ac = Layer2ActorCritic(obs_dim=obs_dim, k_local=HP["k_local"], hidden_dim=HP["hidden_dim"])
    ac.load_state_dict(l2_ckpt["ac_state"])
    l2_trainer = MAPPOTrainer(
        ac, lr=HP["lr"], gamma=HP["gamma"], gae_lambda=HP["gae_lambda"],
        clip_ratio=HP["clip_ratio"], ppo_epochs=HP["ppo_epochs"],
        minibatch_size=HP["minibatch_size"], entropy_coef=HP["entropy_coef_start"],
        value_coef=HP["value_coef"], max_grad_norm=HP["max_grad_norm"],
        total_episodes=args.l2_ep, device=device)
    l2_buffer = RolloutBuffer(gamma=HP["gamma"], gae_lambda=HP["gae_lambda"])
    torch.manual_seed(args.seed)

    print(f"Training L2: {args.l2_ep} episodes")
    l2_metrics, _ = train_l2_phase(env_factory, l2_trainer, l2_buffer, args.l2_ep, device, do_validate=True)

    # Final L2 validation
    print(f"\n  Final L2 validation (10ep)...")
    vio, dly, ont = validate_conflicts(env_factory, l2_trainer, n_ep=10)
    print(f"  violations={vio}  mean_dly={dly:.1f}s  ontime={ont:.2f}")
    if vio > 0:
        print(f"  WARNING: {vio} platform conflicts remaining!")
    else:
        print(f"  SAFE: zero platform conflicts")

    # Save L2 checkpoint
    torch.save({"ac_state": ac.state_dict(), "episode": args.l2_ep, "hp": HP},
               ckpt_dir / "l2_best.pt")

    # ---- Phase 2: L3 with frozen L2 ----
    print(f"\n{'='*60}")
    print(f"  Phase 2: L3 — frozen L2, train speed controller")
    print(f"{'='*60}")

    for param in ac.parameters():
        param.requires_grad = False
    ac.eval()

    l3_actor = CorridorActor(obs_dim=L3_OBS_DIM, hidden_dim=HP["l3_hidden_dim"])
    l3_critic = CorridorCritic(obs_dim=L3_OBS_DIM, hidden_dim=HP["l3_hidden_dim"])
    l3_trainer = L3WorkerTrainer(
        l3_actor, l3_critic, lr=HP["l3_lr"], gamma=HP["gamma"],
        gae_lambda=HP["gae_lambda"], clip_ratio=HP["clip_ratio"],
        ppo_epochs=HP["ppo_epochs"], entropy_coef=HP["l3_entropy_coef"],
        value_coef=HP["value_coef"], max_grad_norm=HP["max_grad_norm"], device=device)

    print(f"Training L3: {args.l3_ep} episodes")
    l3_metrics = train_l3_phase(env_factory, l2_trainer, l3_trainer, args.l3_ep)

    # Save final
    torch.save({
        "ac_state": ac.state_dict(),
        "l3_actor": l3_actor.state_dict(),
        "l3_critic": l3_critic.state_dict(),
        "l2_metrics": l2_metrics, "l3_metrics": l3_metrics, "hp": HP,
    }, ckpt_dir / "final.pt")

    # Summary
    l2_tail = l2_metrics[-min(300, len(l2_metrics)):]
    l3_tail = l3_metrics[-min(300, len(l3_metrics)):]
    print(f"\n{'='*60}")
    print(f"  Complete")
    print(f"  L2 last 300: dly={sum(e['mean_dep_delay'] for e in l2_tail)/len(l2_tail):.1f}s  "
          f"ont={sum(e['ontime_rate'] for e in l2_tail)/len(l2_tail):.2f}")
    print(f"  L3 last 300: dly={sum(e['mean_dep_delay'] for e in l3_tail)/len(l3_tail):.1f}s  "
          f"ont={sum(e['ontime_rate'] for e in l3_tail)/len(l3_tail):.2f}")
    print(f"  Checkpoint: {ckpt_dir / 'final.pt'}")

    # CSV
    all_metrics = []
    for m in l2_metrics:
        m["phase"] = "l2"
        m.setdefault("l3_cruise_pct", 0.0)
        m.setdefault("l3_fast_pct", 0.0)
        all_metrics.append(m)
    for m in l3_metrics:
        m["phase"] = "l3"
        all_metrics.append(m)
    csv_path = run_dir / "training_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
        w.writeheader()
        w.writerows(all_metrics)
    print(f"  CSV: {csv_path}")


if __name__ == "__main__":
    main()
