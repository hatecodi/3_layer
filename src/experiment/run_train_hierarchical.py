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
import random
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.base_env import BaseRailwayEnv
from src.env.five_station_env import FiveStationEnv
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
    "obs_dim": 37,                       # five_station: 22 self + 5 next + 2 res + 2*4
    "action_dim": 45,
    "k_local": 1,
    "hidden_dim": 128,
    "l3_hidden_dim": 128,
    "lr": 3e-4,
    "l3_lr": 1e-3,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.2,
    "ppo_epochs": 5,
    "minibatch_size": 256,
    "entropy_coef_start": 0.01,
    "entropy_coef_end": 0.005,            # floor at 0.005 — lower causes collapse
    "l3_entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 1.0,
    "rollout_steps": 512,
}

# Staged protocol (2026-08-14, user): train L2 first with the SA baseline as the
# L3 "experience" (near-optimal corridor execution -> clean scheduling signal),
# then freeze L2 and train L3 to distil/approach SA's corridor behaviour.
PHASES = {
    "A": {"episodes": 3000, "train_l2": True, "train_l3": False, "l3_sa": True},
    "B": {"episodes": 3000, "train_l2": False, "train_l3": True, "l3_sa": False},
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


def eval_hier_heldout(l2_trainer, l3_worker, cfg, seeds, device, incident=True,
                      use_sa=False, closure=False, relevant=False,
                      coverage="base"):
    """Run held-out incident episodes with the CURRENT L2 policy + L3 executor
    (SA during Phase A, learned L3 during Phase B). No learning. Returns
    {arr_ontime, succ, deadlock, trunc}."""
    n_ontime_arr = 0; n_arr = 0; succ = 0; deadlock = 0; trunc = 0
    for s in seeds:
        env = FiveStationEnv(cfg, seed=s, batch_window_sec=120.0, enable_vt=False,
                             enable_initial_delay=False,
                             enable_corridor_incident=incident,
                             enable_corridor_closure=closure,
                             control_coverage=coverage,
                             l3_worker=l3_worker, l3_sa=use_sa)
        obs, info = env.reset()
        done = False
        while not done:
            masks = info.get("action_masks", {})
            requests = info.get("requests", [])
            group_obs = _build_critic_inputs(obs, requests)
            gs_vec = compact_global_state(info.get("metrics", {}))
            agent_actions = l2_trainer.collect_step(
                obs, masks, group_obs, gs_vec, device=device)
            action_dict = {aid: (a.platform, a.entry)
                           for aid, a in agent_actions.items()}
            if relevant:
                allowed = set(info.get("control_relevant_ids", []))
                action_dict = {aid: act for aid, act in action_dict.items()
                               if aid in allowed}
            obs, _, term, trunc_, info = env.step(
                action_dict, lambda_team=l2_trainer.lam_team)
            done = term or trunc_
            mm = info.get("metrics", {})
            n_ontime_arr += mm.get("n_ontime_arr", 0)
            n_arr += mm.get("n_arrivals", 0)
        status = info.get("episode_status", "?")
        succ += status == "success"; deadlock += status == "deadlock"
        trunc += status == "truncated"
    return {"arr_ontime": n_ontime_arr / max(n_arr, 1),
            "succ": succ, "deadlock": deadlock, "trunc": trunc}


def policy_control_prob(ep: int, total: int, p_start: float = 0.1,
                        ramp_frac: float = 0.6) -> float:
    """Curriculum: probability that the policy (not the heuristic) acts.

    Ramps linearly from ``p_start`` at episode 0 to 1.0 at
    ``ramp_frac * total``, then stays at 1.0.  At p<1 the training loop drops
    actions for a random subset, so the env falls back to heuristic_action for
    them — the policy learns on a stable heuristic baseline instead of having
    to rediscover all scheduling from scratch.
    """
    if total <= 0 or ramp_frac <= 0:
        return 1.0
    end = max(ramp_frac * total, 1.0)
    return min(1.0, p_start + (1.0 - p_start) * (ep / end))


def train_hierarchical(
    env_factory: Callable[[int, Optional[object]], BaseRailwayEnv],
    args,
    obs_dim: int = 29,
    phases: Optional[Dict] = None,
    l2_ckpt: Optional[Path] = None,
    l3_ckpt: Optional[Path] = None,
    cfg=None,
    eval_every=0, eval_n=12, eval_seed_base=900000,
    policy_ramp_start: float = 0.1,
    policy_ramp_frac: float = 0.6,
    control_mode: str = "full",
) -> Dict:
    phases = phases or PHASES
    ctrl_rng = random.Random(getattr(args, "seed", 0) * 7919 + 17)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    # L2 Manager (existing architecture)
    ac = Layer2ActorCritic(obs_dim=obs_dim, k_local=HP["k_local"],
                           hidden_dim=HP["hidden_dim"])
    if l2_ckpt:
        state = torch.load(l2_ckpt, map_location=device, weights_only=False)
        try:
            ac.load_state_dict(state["ac_state"])
        except RuntimeError as exc:
            raise RuntimeError(
                f"L2 checkpoint incompatible with obs_dim={obs_dim} "
                "(obs changes 31->34->37: overtaking + corridor closure); "
                "retrain L2 on this branch"
            ) from exc
        l2_resumes = any(p["train_l2"] for p in phases.values())
        print(f"Loaded L2 from {l2_ckpt} "
              f"({'resuming L2 training' if l2_resumes else 'frozen for L3 training'})")
    l2_trainer = MAPPOTrainer(
        ac, lr=HP["lr"], gamma=HP["gamma"], gae_lambda=HP["gae_lambda"],
        clip_ratio=HP["clip_ratio"], ppo_epochs=HP["ppo_epochs"],
        minibatch_size=HP["minibatch_size"], entropy_coef=HP["entropy_coef_start"],
        value_coef=HP["value_coef"], max_grad_norm=HP["max_grad_norm"],
        total_episodes=sum(p["episodes"] for p in phases.values()),
        device=device)
    l2_buffer = RolloutBuffer(gamma=HP["gamma"], gae_lambda=HP["gae_lambda"])

    # L3 Worker (shared corridor speed controller) — TWO critics: local (own
    # punctuality) + global (team term), mirroring L2's local/global design.
    l3_actor = CorridorActor(obs_dim=L3_OBS_DIM, hidden_dim=HP["l3_hidden_dim"])
    l3_critic = CorridorCritic(obs_dim=L3_OBS_DIM, hidden_dim=HP["l3_hidden_dim"])
    l3_critic_g = CorridorCritic(obs_dim=L3_OBS_DIM, hidden_dim=HP["l3_hidden_dim"])
    l3_trainer = L3WorkerTrainer(
        l3_actor, l3_critic, l3_critic_g, lr=HP["l3_lr"], gamma=HP["gamma"],
        gae_lambda=HP["gae_lambda"], clip_ratio=HP["clip_ratio"],
        ppo_epochs=HP["ppo_epochs"], entropy_coef=HP["l3_entropy_coef"],
        value_coef=HP["value_coef"], max_grad_norm=HP["max_grad_norm"],
        device=device)
    if l3_ckpt:
        state = torch.load(l3_ckpt, map_location=device, weights_only=False)
        try:
            l3_actor.load_state_dict(state["l3_actor"])
            l3_critic.load_state_dict(state["l3_critic_local"])
            l3_critic_g.load_state_dict(state["l3_critic_global"])
        except KeyError as exc:
            raise RuntimeError(
                f"L3 checkpoint {l3_ckpt} missing expected key {exc}"
            ) from exc
        print(f"Loaded L3 from {l3_ckpt} "
              f"(episode {state.get('episode', '?')}); continuing L3 training")

    # Checkpoint dir (out_root lets runs be grouped under output/<date>/, e.g. --out-root output/8_16/training)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / args.out_root / f"{args.env}_hierarchical" / run_id
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    episode_metrics: List[Dict] = []
    eval_rows: List[Dict] = []   # held-out eval at checkpoints (eval_every > 0)
    total_ep = 0
    best_reward = -float("inf")
    # Rolling window so reported metrics average over bunched + non-bunched
    # episodes (single-episode sampling hit only non-bunched seeds and looked
    # unrealistically on-time).
    _WIN = max(50, sum(p["episodes"] for p in phases.values()) // 20)
    win_reward = deque(maxlen=_WIN)
    win_dep = deque(maxlen=_WIN)
    win_ontime = deque(maxlen=_WIN)
    win_arr_ontime = deque(maxlen=_WIN)   # realised (L3) arrival punctuality

    for phase_name, phase_cfg in phases.items():
        print(f"\n{'='*60}")
        print(f"  Phase {phase_name}: {'L2' if phase_cfg['train_l2'] else ''}"
              f"{' + ' if phase_cfg['train_l2'] and phase_cfg['train_l3'] else ''}"
              f"{'L3' if phase_cfg['train_l3'] else ''}"
              f"  ({phase_cfg['episodes']} ep)")
        print(f"{'='*60}\n")

        # Phase C: stabilise joint training
        if phase_name == "C":
            for pg in l2_trainer.optimizer.param_groups:
                pg["lr"] = HP["lr"] * 0.3          # L2 lr: 3e-4 → 1e-4
            for pg in l3_trainer.optimizer.param_groups:
                pg["lr"] = HP["l3_lr"] * 0.5        # L3 lr: 1e-3 → 5e-4

        for ep in range(phase_cfg["episodes"]):
            l2_trainer.advance_episode()
            # Curriculum: fraction of decisions the learned policy controls.
            # Non-L2 phases always use the (frozen) policy fully.
            p_ctrl_ep = (policy_control_prob(
                ep, phase_cfg["episodes"], policy_ramp_start, policy_ramp_frac)
                if (phase_cfg["train_l2"] and control_mode in
                    ("ramp", "relevant_plus_ramp")) else 1.0)

            # Entropy decay for L2
            entropy_progress = min(total_ep / max(sum(p["episodes"] for p in PHASES.values()) * 0.3, 1), 1.0)
            l2_trainer.entropy_coef = (
                HP["entropy_coef_start"]
                + (HP["entropy_coef_end"] - HP["entropy_coef_start"]) * entropy_progress
            )

            # Attach L3 worker only during L3 training phases; use the SA baseline
            # as the L3 "experience" during the L2-only phase.
            worker = l3_trainer if phase_cfg["train_l3"] else None
            env = env_factory(seed=total_ep, l3_worker=worker,
                              l3_sa=phase_cfg.get("l3_sa", False))
            obs, info = env.reset()
            # Whether THIS episode actually got a restriction (prob 0.6).  Read
            # once, right after reset: `_corridor_incidents` is only written in
            # reset(), and downstream analysis must stratify on it — an episode
            # with no incident has no "affected" subset at all, and mixing the
            # two would make the per-episode curve look like a broken sensor.
            ep_has_incident = bool(getattr(env, "_corridor_incidents", {}))

            ep_reward: Dict[str, float] = defaultdict(float)
            ep_done = False
            ep_total_granted = 0
            ep_total_ontime_dep = 0
            ep_total_dep_delay = 0.0
            ep_total_ontime_arr = 0
            ep_total_arrivals = 0
            ep_aff_delays: List[float] = []   # arrival delays of AFFECTED transits
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
                # Control gating.  `ramp`: drop a random fraction so the env
                # uses heuristic_action; `relevant`: only coordination-relevant
                # trains (higher-priority follower / pass opportunity) are
                # learned, all others use the heuristic.  Dropped agents get no
                # transition because the policy did not act for them.
                # Relevance gating must also apply when L2 is FROZEN (Phase B):
                # the frozen policy was trained on the relevant subset, so it
                # must not suddenly act on every routine decision.
                if control_mode in ("relevant", "relevant_plus",
                                    "relevant_plus_ramp"):
                    allowed = set(info.get("control_relevant_ids", []))
                    agent_actions = {
                        aid: a for aid, a in agent_actions.items()
                        if aid in allowed}
                    if control_mode == "relevant_plus_ramp" and p_ctrl_ep < 1.0:
                        agent_actions = {
                            aid: a for aid, a in agent_actions.items()
                            if ctrl_rng.random() < p_ctrl_ep}
                elif p_ctrl_ep < 1.0 and agent_actions:
                    agent_actions = {
                        aid: a for aid, a in agent_actions.items()
                        if ctrl_rng.random() < p_ctrl_ep}
                action_dict = {aid: (a.platform, a.entry) for aid, a in agent_actions.items()}

                next_obs, reward, terminated, truncated, info = env.step(
                    action_dict, lambda_team=l2_trainer.lam_team)
                ep_done = terminated or truncated
                metrics = info.get("metrics", {})
                ep_total_granted += metrics.get("n_granted", 0)
                ep_total_ontime_dep += metrics.get("n_ontime_dep", 0)
                ep_total_dep_delay += metrics.get("total_dep_delay", 0.0)
                ep_total_ontime_arr += metrics.get("n_ontime_arr", 0)
                ep_total_arrivals += metrics.get("n_arrivals", 0)
                # AFFECTED-SUBSET delay (incident arm instrumentation).
                # The all-train means above dilute the disturbance: affected
                # transits are only ~3-8% of the total (see
                # collect_train_dispersion.py's affected_vs_overall), so the
                # "does the policy pull disturbed trains back faster" question
                # is only answerable on this subset.  `affected` is the env's
                # existing flag (transit window overlapped the restriction
                # window); the event list is per-step and exists for every L3
                # layer (CRUISE / SA / learned).
                for ev in info.get("l3_transit_events", []):
                    if ev.get("affected"):
                        ep_aff_delays.append(ev["arr_delay"])
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
                if phase_cfg["train_l3"] and len(l3_trainer.buffer) >= 64:
                    l3_trainer.update()
                    l3_update_count += 1

            # ---- end of episode ----
            if phase_cfg["train_l3"] and len(l3_trainer.buffer) >= 16:
                l3_trainer.update()
                l3_update_count += 1

            total_r = sum(ep_reward.values())
            mean_r = total_r / max(len(ep_reward), 1)
            ep_mean_dep = ep_total_dep_delay / max(ep_total_granted, 1)
            ep_ontime = ep_total_ontime_dep / max(ep_total_granted, 1)
            ep_arr_ontime = ep_total_ontime_arr / max(ep_total_arrivals, 1)

            ep_summary = {
                "episode": total_ep,
                "phase": phase_name,
                "status": info.get("episode_status", "?"),
                "total_reward": total_r,
                "mean_reward": mean_r,
                "mean_dep_delay": ep_mean_dep,
                "ontime_rate": ep_ontime,
                "arr_ontime": ep_arr_ontime,
                # affected-subset delay: nan (not 0) when the episode had no
                # affected transit, so downstream means don't get dragged to 0.
                "has_incident": int(ep_has_incident),
                "n_aff_transits": len(ep_aff_delays),
                "aff_arr_delay_mean": (sum(ep_aff_delays) / len(ep_aff_delays)
                                       if ep_aff_delays else float("nan")),
                "aff_arr_delay_max": (max(ep_aff_delays) if ep_aff_delays
                                      else float("nan")),
                "l3_updates": l3_update_count,
            }
            episode_metrics.append(ep_summary)
            win_reward.append(mean_r)
            win_dep.append(ep_mean_dep)
            win_ontime.append(ep_ontime)
            win_arr_ontime.append(ep_arr_ontime)

            # Report rolling-window MEAN (not a single sampled episode), so the
            # logged delay/ontime reflect the mixed bunched + non-bunched mix.
            if (total_ep + 1) % _WIN == 0 or total_ep < 5:
                print(
                    f"  ep {total_ep:>5} [{phase_name}] | reward={sum(win_reward)/len(win_reward):>7.2f} | "
                    f"dep_dly={sum(win_dep)/len(win_dep):>6.1f}s | "
                    f"dep_ontime={sum(win_ontime)/len(win_ontime):.2f} | "
                    f"arr_ontime={sum(win_arr_ontime)/len(win_arr_ontime):.2f} | "
                    f"ctrl={control_mode} p={p_ctrl_ep:.2f} | "
                    f"l3_upd={l3_update_count:>2} | "
                    f"{info.get('episode_status', '?'):>12}"
                )

            if mean_r > best_reward:
                best_reward = mean_r
                torch.save(
                    {"ac_state": ac.state_dict(), "l3_actor": l3_actor.state_dict(),
                     "l3_critic_local": l3_critic.state_dict(),
                     "l3_critic_global": l3_critic_g.state_dict(),
                     "episode": total_ep, "hp": HP},
                    ckpt_dir / "best.pt",
                )

            # ---- held-out eval hook (fixed seeds, current pipeline, no learning) ----
            if eval_every > 0 and (total_ep + 1) % eval_every == 0:
                ev_seeds = [eval_seed_base + i * 7919 for i in range(eval_n)]
                ev = eval_hier_heldout(
                    l2_trainer,
                    l3_trainer if phase_cfg["train_l3"] else None,
                    cfg, ev_seeds, device, incident=args.incident,
                    use_sa=phase_cfg["l3_sa"], closure=args.closure,
                    relevant=(args.control_mode in (
                        "relevant", "relevant_plus", "relevant_plus_ramp")),
                    coverage=("plus" if args.control_mode in (
                        "relevant_plus", "relevant_plus_ramp") else "base"))
                eval_rows.append({
                    "episode": total_ep + 1, "phase": phase_name,
                    "arr_ontime": round(ev["arr_ontime"], 4),
                    "succ": ev["succ"], "deadlock": ev["deadlock"],
                    "trunc": ev["trunc"]})
                print(f"    [heldout @ ep {total_ep+1} {phase_name}] "
                      f"arr_ontime={ev['arr_ontime']:.3f} "
                      f"(succ {ev['succ']}/{len(ev_seeds)})")

            total_ep += 1

    # ---- final save ----
    torch.save(
        {"ac_state": ac.state_dict(), "l3_actor": l3_actor.state_dict(),
         "l3_critic_local": l3_critic.state_dict(),
         "l3_critic_global": l3_critic_g.state_dict(), "hp": HP,
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
        # Collect all keys across all phases to handle mismatched dicts
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

    if eval_rows:
        ev_path = run_dir / "eval_ontime.csv"
        with open(ev_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys()))
            writer.writeheader()
            writer.writerows(eval_rows)
        print(f"  held-out eval CSV: {ev_path}")

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
    parser.add_argument("--incident", action="store_true",
                        help="corridor speed-restriction incident as the training "
                             "disturbance (disables origin-bunching)")
    parser.add_argument("--closure", action="store_true",
                        help="corridor closure (区间中断): no new entries into a "
                             "corridor during a window, designed to hold a "
                             "lower-priority leader behind a higher-priority "
                             "follower (disables origin-bunching)")
    parser.add_argument("--l2-ckpt", type=Path, default=None,
                        help="load a pretrained L2 checkpoint (Phase B: freeze it, "
                             "train L3 against it)")
    parser.add_argument("--l3-ckpt", type=Path, default=None,
                        help="resume L3 actor + critics from a checkpoint "
                             "(e.g. continue Phase B from final.pt)")
    parser.add_argument("--phases", type=str, default="A,B",
                        help="comma-separated phases to run, e.g. 'A' (L2+SA-L3) "
                             "or 'A,B' (then freeze L2, train L3)")
    parser.add_argument("--total-episodes", type=int, default=None,
                        help="rescale the selected phases' episode budgets to sum "
                             "to N, proportional to their PHASES defaults "
                             "(e.g. 7000 -> A=3500, B=3500). Default: PHASES as-is.")
    parser.add_argument("--out-root", type=str, default="output",
                        help="root dir under PROJECT_ROOT for this run, e.g. "
                             "output/8_16/training (default 'output')")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="run a held-out incident eval every N training eps (0=off)")
    parser.add_argument("--eval-n", type=int, default=12,
                        help="number of held-out episodes per eval checkpoint")
    parser.add_argument("--eval-seed-base", type=int, default=900000,
                        help="base seed for the deterministic held-out eval set")
    parser.add_argument("--policy-ramp-start", type=float, default=0.1,
                        help="curriculum: fraction of decisions learned at ep 0 "
                             "(rest use heuristic_action)")
    parser.add_argument("--policy-ramp-frac", type=float, default=0.6,
                        help="curriculum: fraction of the phase over which policy "
                             "control ramps from start to 1.0")
    parser.add_argument("--control-mode",
                        choices=["full", "ramp", "relevant", "relevant_plus",
                                 "relevant_plus_ramp"],
                        default="full",
                        help="L2 control: full=all batch-window trains; "
                             "ramp=heuristic->policy curriculum; "
                             "relevant=only higher-priority-follower / pass cases; "
                             "relevant_plus=relevant + closure + any co-pending "
                             "same-direction partner (~85% coverage); "
                             "relevant_plus_ramp=plus coverage with a "
                             "heuristic->policy control-probability ramp")
    parser.add_argument("--extern-alpha", type=float, default=0.15,
                        help="moving-block externality price (reward.py EXTERN_ALPHA)")
    parser.add_argument("--yield-cap", action="store_true",
                        help="cap yield entry choices to plan/+10/+20 (exclude "
                             "+30/+50) for trains ahead of a higher-priority follower")
    parser.add_argument("--yield-tail-coef", type=float, default=0.15,
                        help="soft penalty per avoidable yield second beyond "
                             "YIELD_TAIL_THRESHOLD=20s")
    args = parser.parse_args()

    HP["lr"] = args.lr
    HP["extern_alpha"] = args.extern_alpha
    HP["yield_tail_coef"] = args.yield_tail_coef
    # Reward functions read the module global at call time.
    import src.env.reward as _reward_mod
    _reward_mod.EXTERN_ALPHA = args.extern_alpha
    _reward_mod.YIELD_TAIL_COEF = args.yield_tail_coef
    print(f"EXTERN_ALPHA = {args.extern_alpha}, "
          f"YIELD_TAIL_COEF = {args.yield_tail_coef}")
    torch.manual_seed(args.seed)

    from src.env.five_station_env import FiveStationEnv
    from src.experiment.scenario_loader import load_scenario

    cfg = load_scenario(scenario_name=args.env)
    obs_dim = get_obs_dim(cfg)

    def env_factory(seed: int, l3_worker=None, l3_sa: bool = False) -> FiveStationEnv:
        cov = ("plus" if args.control_mode in ("relevant_plus",
                                               "relevant_plus_ramp")
               else "full" if args.control_mode == "full" else "base")
        return FiveStationEnv(
            cfg=cfg, seed=seed, batch_window_sec=120.0,
            enable_vt=args.vt,
            enable_initial_delay=(args.delay and not args.incident
                                  and not args.closure),
            enable_corridor_incident=args.incident,
            enable_corridor_closure=args.closure,
            control_coverage=cov,
            l3_worker=l3_worker, l3_sa=l3_sa,
            yield_entry_cap=args.yield_cap,
        )

    phases = {p: dict(PHASES[p]) for p in args.phases.split(",") if p in PHASES}
    if args.total_episodes is not None:
        if args.total_episodes < len(phases):
            parser.error("--total-episodes must be >= number of phases")
        default_total = sum(p["episodes"] for p in phases.values())
        names = list(phases)
        allocated = 0
        for i, name in enumerate(names):
            if i == len(names) - 1:
                phases[name]["episodes"] = args.total_episodes - allocated
            else:
                n_ep = round(args.total_episodes * PHASES[name]["episodes"] / default_total)
                phases[name]["episodes"] = n_ep
                allocated += n_ep
        print("Episode budget: " + ", ".join(
            f"{n}={phases[n]['episodes']}" for n in names))
    train_hierarchical(env_factory, args, obs_dim=obs_dim, phases=phases,
                       l2_ckpt=args.l2_ckpt, l3_ckpt=args.l3_ckpt, cfg=cfg,
                       eval_every=args.eval_every, eval_n=args.eval_n,
                       eval_seed_base=args.eval_seed_base,
                       policy_ramp_start=args.policy_ramp_start,
                       policy_ramp_frac=args.policy_ramp_frac,
                       control_mode=args.control_mode)


if __name__ == "__main__":
    main()
