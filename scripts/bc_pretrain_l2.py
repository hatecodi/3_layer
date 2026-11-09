#!/usr/bin/env python3
"""Behavior-cloning warm start for Layer-2 scheduling from the heuristic.

Collects (obs, platform, entry) labels by rolling out the deterministic
heuristic over the *plus* coordination set (closure included), then trains the
Layer-2 actor with masked cross-entropy.  The saved checkpoint is directly
loadable by run_train_hierarchical.py --l2-ckpt ({"ac_state": ...}).

Why: with ~85% policy coverage the untrained PPO policy is far below the
heuristic (cold start).  BC starts it at the heuristic policy; PPO then
fine-tunes.  The critic is not trained here (PPO learns it).
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.control.action_mask import ENTRY_CHOICES, heuristic_action
from src.env.five_station_env import TRAVEL_TIME_TO_THROAT, FiveStationEnv
from src.env.observation_builder import get_obs_dim
from src.experiment.scenario_loader import load_scenario
from src.marl.actor_critic import Layer2ActorCritic


def collect(cfg, episodes: int, seed0: int, coverage: str, closure: bool,
            sa_iters: int, max_samples: int):
    """Heuristic-rollout labels over the coordination-relevant decisions."""
    data = []
    for ep in range(episodes):
        env = FiveStationEnv(
            cfg=cfg, seed=seed0 + ep, enable_vt=False,
            enable_initial_delay=False, enable_corridor_incident=False,
            enable_corridor_closure=closure, l3_sa=True,
            l3_sa_iters=sa_iters, yield_entry_cap=True,
            control_coverage=coverage)
        obs, info = env.reset()
        for _ in range(600):
            rel = set(info.get("control_relevant_ids", []))
            for tid in info.get("active_train_ids", []):
                if tid not in rel:
                    continue
                masks = info.get("action_masks", {}).get(tid)
                if not masks:
                    continue
                pm, em = masks
                st = env._journeys[tid].current_station
                train = cfg.trains[tid]
                bounds = env._get_effective_bounds(tid, st, env._t_now)
                try:
                    plat, entry, _ = heuristic_action(
                        train, st, env._t_now, TRAVEL_TIME_TO_THROAT, cfg,
                        bounds_override=bounds)
                except ValueError:
                    continue
                plats = train.candidate_platforms.get(st, [])
                if plat not in plats or entry not in ENTRY_CHOICES:
                    continue
                pi = plats.index(plat)
                ei = ENTRY_CHOICES.index(entry)
                if pi >= len(pm) or not pm[pi]:
                    continue
                if ei >= len(em) or not em[ei]:
                    continue
                data.append((
                    list(obs[tid]),
                    list(pm) + [False] * (4 - len(pm)),
                    list(em) + [False] * (6 - len(em)),
                    pi, ei, ep))
            obs, _, term, trunc, info = env.step({}, lambda_team=0.3)
            if term or trunc:
                break
        if len(data) >= max_samples:
            break
    return data


def accuracy(model, obs, pm, em, pi, ei, device):
    model.eval()
    with torch.no_grad():
        p_logits, e_logits = model.actor.logits(
            obs.to(device), pm.to(device), em.to(device))
        p_acc = (p_logits.argmax(-1).cpu() == pi).float().mean().item()
        e_acc = (e_logits.argmax(-1).cpu() == ei).float().mean().item()
        joint = ((p_logits.argmax(-1).cpu() == pi)
                 & (e_logits.argmax(-1).cpu() == ei)).float().mean().item()
    return p_acc, e_acc, joint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="five_station_15t")
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--seed", type=int, default=500000)
    ap.add_argument("--coverage", choices=["base", "plus", "full"], default="plus")
    ap.add_argument("--closure", action="store_true")
    ap.add_argument("--sa-iters", type=int, default=300)
    ap.add_argument("--max-samples", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=Path("output/11_09/bc_l2_closure.pt"))
    ap.add_argument("--metrics-out", type=Path,
                    default=Path("output/11_09/bc_l2_closure_metrics.csv"))
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cfg = load_scenario(scenario_name=args.scenario)
    obs_dim = get_obs_dim(cfg)
    print(f"scenario={args.scenario} obs_dim={obs_dim} device={device}")

    print(f"collecting heuristic labels: {args.episodes} episodes, "
          f"coverage={args.coverage}, closure={args.closure}, sa_iters={args.sa_iters}")
    data = collect(cfg, args.episodes, args.seed, args.coverage, args.closure,
                   args.sa_iters, args.max_samples)
    print(f"collected {len(data)} samples from {min(args.episodes, max([d[5] for d in data]) + 1 if data else 0)} episodes")
    if not data:
        raise SystemExit("no BC samples collected")

    obs = torch.tensor([d[0] for d in data], dtype=torch.float32)
    pm = torch.tensor([d[1] for d in data], dtype=torch.bool)
    em = torch.tensor([d[2] for d in data], dtype=torch.bool)
    pi = torch.tensor([d[3] for d in data], dtype=torch.long)
    ei = torch.tensor([d[4] for d in data], dtype=torch.long)
    eps = torch.tensor([d[5] for d in data], dtype=torch.long)

    max_ep = int(eps.max().item())
    val_ep = int(max_ep * (1.0 - args.val_frac))
    tr = (eps <= val_ep).nonzero(as_tuple=True)[0]
    va = (eps > val_ep).nonzero(as_tuple=True)[0]
    if len(va) < 50:
        n = max(50, len(obs) // 10)
        va = torch.arange(len(obs) - n, len(obs))
        tr = torch.arange(0, len(obs) - n)
    print(f"train={len(tr)} val={len(va)}")

    model = Layer2ActorCritic(obs_dim=obs_dim, k_local=1, hidden_dim=128).to(device)
    opt = torch.optim.Adam(model.actor.parameters(), lr=args.lr)
    ce = torch.nn.CrossEntropyLoss()

    rows = []
    best_joint = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = tr[torch.randperm(len(tr))]
        total = 0.0
        nb = 0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            p_logits, e_logits = model.actor.logits(
                obs[idx].to(device), pm[idx].to(device), em[idx].to(device))
            loss = (ce(p_logits, pi[idx].to(device))
                    + ce(e_logits, ei[idx].to(device)))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            nb += 1
        pa, ea, ja = accuracy(model, obs[va], pm[va], em[va], pi[va], ei[va],
                              device)
        rows.append({"epoch": epoch, "train_loss": total / max(nb, 1),
                     "val_platform_acc": pa, "val_entry_acc": ea,
                     "val_joint_acc": ja})
        print(f"epoch {epoch:2d} loss={total/max(nb,1):.4f} "
              f"val platform={pa:.3f} entry={ea:.3f} joint={ja:.3f}")
        if ja > best_joint:
            best_joint = ja
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"ac_state": model.state_dict(),
                        "bc_meta": {"coverage": args.coverage,
                                    "closure": args.closure,
                                    "samples": len(data),
                                    "val_joint_acc": ja,
                                    "epoch": epoch}},
                       args.out)

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"best val joint acc={best_joint:.3f}")
    print(f"saved BC checkpoint: {args.out}")
    print(f"saved metrics: {args.metrics_out}")


if __name__ == "__main__":
    main()
