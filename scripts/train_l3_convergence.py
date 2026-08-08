#!/usr/bin/env python3
"""Standard L3 convergence verification: multi-seed training with per-seed
logging (entropy/L3 loss/reward + eval on a FIXED held-out seed set) and a
slope-criterion plateau check. Writes a combined CSV for the plot script.

Usage:
    python scripts/train_l3_convergence.py \
        --scenario five_station_15t \
        --l2 output/five_station_15t_hierarchical/<run>/checkpoints/final.pt \
        --seeds 3 --episodes 2500 --eval-every 125 \
        --eval-seeds 900000 --eval-n 8 --out output/figures/l3v3_conv_15t_ms.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env.five_station_env import FiveStationEnv
from src.env.observation_builder import get_obs_dim, critic_input
from src.marl.actor_critic import Layer2ActorCritic, compact_global_state
from src.marl.mappo import MAPPOTrainer
from src.marl.l3_worker import CorridorActor, CorridorCritic, L3WorkerTrainer, L3_OBS_DIM
from src.experiment.scenario_loader import load_scenario
from src.control.resource_manager import build_conflict_graph


def slope_of(rows, key):
    """Linear-regression slope of `key` over the last len(rows) points vs episode."""
    n = len(rows)
    if n < 3:
        return float("inf")
    eps = [r["episode"] for r in rows]
    vals = [r[key] for r in rows]
    x0 = eps[0]
    x = [e - x0 for e in eps]
    xm = sum(x) / n
    ym = sum(vals) / n
    den = sum((x[i] - xm) ** 2 for i in range(n))
    if den == 0:
        return float("inf")
    return sum((x[i] - xm) * (vals[i] - ym) for i in range(n)) / den


def plateau_check(rows, key, K, thresh):
    last = rows[-K:]
    return abs(slope_of(last, key)) < thresh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="five_station_15t")
    ap.add_argument("--l2", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=2500)
    ap.add_argument("--eval-every", type=int, default=125)
    ap.add_argument("--eval-seeds", type=int, default=900000)
    ap.add_argument("--eval-n", type=int, default=8)
    ap.add_argument("--plateau-k", type=int, default=8)
    ap.add_argument("--ontime-thresh", type=float, default=0.5)
    ap.add_argument("--dep-thresh", type=float, default=0.15)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--entropy-start", type=float, default=0.05)
    ap.add_argument("--entropy-end", type=float, default=0.01)
    args = ap.parse_args()
    args.out = args.out or Path(f"output/figures/l3v3_conv_{args.scenario}_ms.csv")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_scenario(scenario_name=args.scenario)
    obs_dim = get_obs_dim(cfg)
    l2 = torch.load(args.l2, map_location=dev, weights_only=False)
    ac = Layer2ActorCritic(obs_dim=obs_dim, k_local=1, hidden_dim=128)
    ac.load_state_dict(l2["ac_state"]); ac.eval()
    for p in ac.parameters():
        p.requires_grad = False
    tr = MAPPOTrainer(ac, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2, ppo_epochs=5,
                      minibatch_size=256, entropy_coef=0.01, value_coef=0.5, max_grad_norm=1.0,
                      total_episodes=1, device=dev)

    def bc(obs, requests):
        g = build_conflict_graph(requests)
        return {aid: critic_input(aid, obs, g, k_max=1) for aid in obs}

    all_rows = []
    for seed in range(args.seeds):
        torch.manual_seed(1000 + seed)
        l3 = L3WorkerTrainer(CorridorActor(L3_OBS_DIM, hidden_dim=128),
                             CorridorCritic(L3_OBS_DIM, hidden_dim=128),
                             lr=1e-3, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2, ppo_epochs=5,
                             entropy_coef=args.entropy_start, value_coef=0.5, max_grad_norm=1.0,
                             device=dev)
        ep_rew = []

        def eval_worker():
            ev = Counter(); g = 0; o = 0; d = 0.0
            for s in range(args.eval_n):
                e2 = FiveStationEnv(cfg=cfg, seed=args.eval_seeds + s, enable_vt=False,
                                    enable_initial_delay=True, l3_worker=l3)
                o2, i2 = e2.reset()
                for _ in range(600):
                    m2 = i2.get("action_masks", {})
                    g2 = bc(o2, i2.get("requests", [])); gs2 = compact_global_state(i2.get("metrics", {}))
                    aa2 = tr.collect_step(o2, m2, g2, gs2, device=dev)
                    ad2 = {aid: (a.platform, a.entry) for aid, a in aa2.items()}
                    o2, r2, t2, tr2, i2 = e2.step(ad2, lambda_team=0.3)
                    mm = i2.get("metrics", {})
                    g += mm.get("n_granted", 0); o += mm.get("n_ontime_dep", 0)
                    d += mm.get("total_dep_delay", 0.0)
                    if t2 or tr2:
                        break
            return o / max(g, 1) * 100, d / max(g, 1)

        seed_rows = []
        for ep in range(args.episodes):
            l3.entropy_coef = args.entropy_start - (args.entropy_start - args.entropy_end) * (ep / args.episodes)
            env = FiveStationEnv(cfg=cfg, seed=seed * 1000000 + ep, enable_vt=False,
                                 enable_initial_delay=True, l3_worker=l3)
            obs, info = env.reset()
            for _ in range(600):
                masks = info.get("action_masks", {})
                g0 = bc(obs, info.get("requests", [])); gs = compact_global_state(info.get("metrics", {}))
                aa = tr.collect_step(obs, masks, g0, gs, device=dev)
                ad = {aid: (a.platform, a.entry) for aid, a in aa.items()}
                obs, rew, term, trunc, info = env.step(ad, lambda_team=0.3)
                if term or trunc:
                    break
            ep_rew.append(env._l3_ep_reward)
            if len(l3.buffer) >= 64:
                l3.update()
            if (ep + 1) % args.eval_every == 0:
                st = l3.get_last_stats()
                ont, dep = eval_worker()
                row = {"seed": seed, "episode": ep + 1,
                       "l3_reward": sum(ep_rew[-args.eval_every:]) / max(len(ep_rew[-args.eval_every:]), 1),
                       "entropy": st.get("l3_entropy", 0),
                       "policy_loss": st.get("l3_loss_policy", 0),
                       "value_loss": st.get("l3_loss_value", 0),
                       "ontime_pct": ont, "mean_dep": dep}
                seed_rows.append(row)
                all_rows.append(row)
        plat_ont = plateau_check(seed_rows, "ontime_pct", args.plateau_k, args.ontime_thresh)
        plat_dep = plateau_check(seed_rows, "mean_dep", args.plateau_k, args.dep_thresh)
        print(f"seed {seed}: plateau_ontime={plat_ont} plateau_mean_dep={plat_dep} "
              f"final_ontime={ont:.1f}% dep={dep:.1f}s", flush=True)
        torch.save({"l3_actor": l3.actor.state_dict(), "l3_critic": l3.critic.state_dict()},
                   f"output/l3v3_{args.scenario}_seed{seed}.pt")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "episode", "l3_reward", "entropy",
                                          "policy_loss", "value_loss", "ontime_pct", "mean_dep"])
        w.writeheader(); w.writerows(all_rows)
    print(f"saved {args.out} ({len(all_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
