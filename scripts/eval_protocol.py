#!/usr/bin/env python3
"""Standard verification protocol: fixed held-out eval set, stratified by
bunched/non-bunched disturbance, full delay distribution, learned/SA/CRUISE.

Usage:
    python scripts/eval_protocol.py --scenario five_station_15t \
        --l2 output/five_station_15t_hierarchical/.../final.pt \
        --l3 output/l3v3_15t_smooth_l3.pt \
        --seeds 800000 --n 60
    python scripts/eval_protocol.py --scenario five_station_15t --sa-only ...   (SA)
    python scripts/eval_protocol.py --scenario five_station_15t --cruise-only   (CRUISE)
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env.five_station_env import FiveStationEnv
from src.env.observation_builder import get_obs_dim, critic_input
from src.marl.actor_critic import Layer2ActorCritic, compact_global_state
from src.marl.mappo import MAPPOTrainer
from src.marl.l3_worker import CorridorActor, CorridorCritic, L3WorkerTrainer, L3_OBS_DIM
from src.experiment.scenario_loader import load_scenario
from src.control.resource_manager import build_conflict_graph


def _load_l2(cfg, l2_ckpt, device):
    obs_dim = get_obs_dim(cfg)
    ckpt = torch.load(l2_ckpt, map_location=device, weights_only=False)
    ac = Layer2ActorCritic(obs_dim=obs_dim, k_local=1, hidden_dim=128)
    ac.load_state_dict(ckpt["ac_state"]); ac.eval()
    for p in ac.parameters():
        p.requires_grad = False
    tr = MAPPOTrainer(ac, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2, ppo_epochs=5,
                      minibatch_size=256, entropy_coef=0.01, value_coef=0.5, max_grad_norm=1.0,
                      total_episodes=1, device=device)
    return tr


def _load_l3(l3_ckpt, device):
    ckpt = torch.load(l3_ckpt, map_location=device, weights_only=False)
    a = CorridorActor(L3_OBS_DIM, hidden_dim=128); a.load_state_dict(ckpt["l3_actor"]); a.eval()
    c = CorridorCritic(L3_OBS_DIM, hidden_dim=128); c.load_state_dict(ckpt["l3_critic"]); c.eval()
    for p in list(a.parameters()) + list(c.parameters()):
        p.requires_grad = False
    return L3WorkerTrainer(a, c, lr=1e-3, device=device)


def run_episode(cfg, tr, l3, seed, mode, dev):
    env = FiveStationEnv(cfg=cfg, seed=seed, enable_vt=False, enable_initial_delay=True,
                         l3_worker=(l3 if mode == "learned" else None), l3_sa=(mode == "sa"))
    obs, info = env.reset()
    # classify bunched: any train with origin arrival shifted past its planned dep
    bunched = 0
    for tid, j in env._journeys.items():
        t = cfg.trains[tid]
        if j.current_station == t.origin_station:
            p = t.planned.get(t.origin_station)
            if p and j.arrival_at_current > p.dep + 1:
                bunched += 1
    g = 0; o = 0; delays = []; status = "?"
    for _ in range(600):
        masks = info.get("action_masks", {})
        g0 = build_conflict_graph(info.get("requests", []))
        go = {aid: critic_input(aid, obs, g0, k_max=1) for aid in obs}
        gs = compact_global_state(info.get("metrics", {}))
        aa = tr.collect_step(obs, masks, go, gs, device=dev)
        ad = {aid: (a.platform, a.entry) for aid, a in aa.items()}
        for gr in info.get("granted_windows", []):
            stn = gr.platform.split("_")[0]
            p = cfg.trains[gr.train_id].planned.get(stn)
            if p:
                delays.append(abs(gr.W_out[0] - p.dep))
        obs, rew, term, trunc, info = env.step(ad, lambda_team=0.3)
        m = info.get("metrics", {})
        g += m.get("n_granted", 0); o += m.get("n_ontime_dep", 0)
        if term or trunc:
            status = info["episode_status"]; break
    return {"bunched": bunched >= 1, "g": g, "o": o, "delays": delays, "status": status}


def report(label, ep_results):
    ds = sorted([d for r in ep_results for d in r["delays"]])
    g = sum(r["g"] for r in ep_results); o = sum(r["o"] for r in ep_results)
    n = len(ds)
    if n == 0:
        print(f"{label}: no grants"); return
    p90 = ds[int(n * 0.9)]; p95 = ds[int(n * 0.95)]
    gt60 = sum(1 for d in ds if d > 60) / n * 100
    succ = sum(1 for r in ep_results if r["status"] == "success")
    print(f"{label}: ontime={o/max(g,1)*100:.1f}% mean_dep={sum(ds)/n:.1f}s "
          f"p50={ds[n//2]:.1f} p90={p90:.1f} p95={p95:.1f} max={ds[-1]:.1f} "
          f"delay>60s={gt60:.1f}% succ={succ}/{len(ep_results)} n_grants={g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="five_station_15t")
    ap.add_argument("--l2", required=True)
    ap.add_argument("--l3", default=None)
    ap.add_argument("--sa-only", action="store_true")
    ap.add_argument("--cruise-only", action="store_true")
    ap.add_argument("--seeds", type=int, default=800000)
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_scenario(scenario_name=args.scenario)
    tr = _load_l2(cfg, args.l2, dev)
    l3 = _load_l3(args.l3, dev) if (args.l3 and not args.sa_only) else None

    modes = []
    if args.cruise_only:
        modes = ["cruise"]
    elif args.sa_only:
        modes = ["sa"]
    else:
        modes = ["learned", "sa", "cruise"]

    for mode in modes:
        by_bunch = {"bunched": [], "non": []}
        for s in range(args.seeds, args.seeds + args.n):
            r = run_episode(cfg, tr, l3, s, mode, dev)
            by_bunch["bunched" if r["bunched"] else "non"].append(r)
        print(f"\n== {mode} (held-out {args.n} seeds, start {args.seeds}) ==")
        report("  combined       ", by_bunch["bunched"] + by_bunch["non"])
        report(f"  bunched ({len(by_bunch['bunched'])})", by_bunch["bunched"])
        report(f"  non-bunched ({len(by_bunch['non'])})", by_bunch["non"])


if __name__ == "__main__":
    main()
