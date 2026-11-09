#!/usr/bin/env python3
"""Behavior-cloning warm start for the L3 corridor worker from SA-online.

Collects (10-dim per-section obs, speed mode) labels by running the frozen
Layer-2 policy with SA-online as the L3 executor (partial observation, no
global forecast), then trains CorridorActor with cross-entropy.  The saved
checkpoint contains the actor + two random critics and is loadable by
eval_protocol / run_train_hierarchical (--l3-ckpt).

This is the baseline for the SA-BC branch: pure BC approaches SA-online, so it
quantifies the on-time vs tail Pareto trade-off before any tail-aware PPO.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval_protocol import build_conflict_graph, critic_input
from src.env.five_station_env import FiveStationEnv
from src.env.observation_builder import get_obs_dim
from src.experiment.scenario_loader import load_scenario
from src.marl.actor_critic import Layer2ActorCritic
from src.marl.l3_worker import L3_OBS_DIM, CorridorActor, CorridorCritic
from src.marl.mappo import MAPPOTrainer, compact_global_state


def load_l2(cfg, ckpt, device):
    ac = Layer2ActorCritic(obs_dim=get_obs_dim(cfg), k_local=1, hidden_dim=128)
    ac.load_state_dict(torch.load(ckpt, map_location=device,
                                  weights_only=False)["ac_state"])
    ac.eval()
    for p in ac.parameters():
        p.requires_grad = False
    tr = MAPPOTrainer(
        ac, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2,
        ppo_epochs=5, minibatch_size=256, entropy_coef=0.01, value_coef=0.5,
        max_grad_norm=1.0, total_episodes=1, device=device)
    return tr


def collect(cfg, tr, l2_ckpt, device, episodes, seed0, max_samples):
    """L2-policy rollouts with SA-online, recording (L3 obs, SA mode)."""
    data = []
    for ep in range(episodes):
        env = FiveStationEnv(
            cfg=cfg, seed=seed0 + ep, enable_vt=False,
            enable_initial_delay=False, enable_corridor_incident=False,
            enable_corridor_closure=True, l3_sa_online=True,
            record_l3_bc=True, yield_entry_cap=True,
            control_coverage="base")
        obs, info = env.reset()
        for _ in range(600):
            rel = set(info.get("control_relevant_ids", []))
            g0 = build_conflict_graph(info.get("requests", []))
            go = {aid: critic_input(aid, obs, g0, k_max=1) for aid in obs}
            gs = compact_global_state(info.get("metrics", {}))
            aa = tr.collect_step(obs, info.get("action_masks", {}), go, gs,
                                 device=device)
            ad = {aid: (a.platform, a.entry) for aid, a in aa.items()
                  if aid in rel}
            obs, _, term, trunc, info = env.step(ad, lambda_team=0.3)
            for o, a in env._l3_bc_samples:
                data.append((list(o), int(a), ep))
            env._l3_bc_samples.clear()
            if term or trunc:
                break
        if len(data) >= max_samples:
            break
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="five_station_15t")
    ap.add_argument("--l2", type=Path, required=True,
                    help="frozen L2 checkpoint used to generate states")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=600000)
    ap.add_argument("--max-samples", type=int, default=40000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", type=Path,
                    default=Path("output/11_09/bc_l3_saonline.pt"))
    ap.add_argument("--metrics-out", type=Path,
                    default=Path("output/11_09/bc_l3_saonline_metrics.csv"))
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cfg = load_scenario(scenario_name=args.scenario)
    print(f"device={device} l2={args.l2}")

    tr = load_l2(cfg, args.l2, device)
    print(f"collecting SA-online labels: {args.episodes} episodes")
    data = collect(cfg, tr, args.l2, device, args.episodes, args.seed,
                   args.max_samples)
    print(f"collected {len(data)} section samples")
    if not data:
        raise SystemExit("no L3 BC samples collected")

    obs = torch.tensor([d[0] for d in data], dtype=torch.float32)
    act = torch.tensor([d[1] for d in data], dtype=torch.long)
    ep = torch.tensor([d[2] for d in data], dtype=torch.long)
    max_ep = int(ep.max().item())
    val_ep = int(max_ep * (1.0 - args.val_frac))
    tr_idx = (ep <= val_ep).nonzero(as_tuple=True)[0]
    va_idx = (ep > val_ep).nonzero(as_tuple=True)[0]
    if len(va_idx) < 50:
        n = max(50, len(obs) // 10)
        va_idx = torch.arange(len(obs) - n, len(obs))
        tr_idx = torch.arange(0, len(obs) - n)
    print(f"train={len(tr_idx)} val={len(va_idx)} "
          f"action counts={torch.bincount(act).tolist()}")

    actor = CorridorActor(L3_OBS_DIM, hidden_dim=128).to(device)
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    ce = torch.nn.CrossEntropyLoss()

    rows = []
    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        actor.train()
        perm = tr_idx[torch.randperm(len(tr_idx))]
        total = 0.0
        nb = 0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            logits = actor.logits(obs[idx].to(device))
            loss = ce(logits, act[idx].to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            nb += 1
        actor.eval()
        with torch.no_grad():
            logits = actor.logits(obs[va_idx].to(device))
            acc = (logits.argmax(-1).cpu() == act[va_idx]).float().mean().item()
        rows.append({"epoch": epoch, "train_loss": total / max(nb, 1),
                     "val_acc": acc})
        print(f"epoch {epoch:2d} loss={total/max(nb,1):.4f} val_acc={acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "l3_actor": actor.state_dict(),
                "l3_critic_local": CorridorCritic(L3_OBS_DIM, hidden_dim=128).state_dict(),
                "l3_critic_global": CorridorCritic(L3_OBS_DIM, hidden_dim=128).state_dict(),
                "bc_meta": {"teacher": "sa-online", "samples": len(data),
                            "val_acc": acc, "epoch": epoch},
            }, args.out)

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"best val acc={best_acc:.3f}")
    print(f"saved {args.out}")
    print(f"saved {args.metrics_out}")


if __name__ == "__main__":
    main()
