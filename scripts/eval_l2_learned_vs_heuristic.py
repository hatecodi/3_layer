#!/usr/bin/env python3
"""Learned L2 vs heuristic L2 on a five-station scenario.

Both policies run the SAME L3 (rule-based CRUISE unless --l3-sa), so any
gap isolates L2 coordination value: does a trained L2 (platform, entry)
beat the deterministic first-legal heuristic on the given config?

Usage:
    python scripts/eval_l2_learned_vs_heuristic.py --scenario five_station_20t \
        --ckpt output/five_station_20t_hierarchical/<run>/checkpoints/final.pt \
        --seeds 10 [--vt] [--delay] [--l3-sa]
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment.scenario_loader import load_scenario
from src.env.five_station_env import FiveStationEnv
from src.env.observation_builder import get_obs_dim, critic_input
from src.marl.actor_critic import Layer2ActorCritic, compact_global_state
from src.marl.mappo import MAPPOTrainer
from src.control.resource_manager import build_conflict_graph

ONTIME_THRESHOLD_SEC = 15.0


def _load_l2(cfg, l2_ckpt, device):
    ckpt = torch.load(l2_ckpt, map_location=device, weights_only=False)
    obs_dim = get_obs_dim(cfg)
    ac = Layer2ActorCritic(obs_dim=obs_dim, k_local=1, hidden_dim=128)
    ac.load_state_dict(ckpt["ac_state"])
    ac.eval()
    for p in ac.parameters():
        p.requires_grad = False
    tr = MAPPOTrainer(ac, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2,
                      ppo_epochs=5, minibatch_size=256, entropy_coef=0.01,
                      value_coef=0.5, max_grad_norm=1.0, total_episodes=1,
                      device=device)
    return tr


def run_episode(cfg, tr, seed, mode, dev, vt, delay, incident, l3_sa, severity=1.0):
    env = FiveStationEnv(cfg=cfg, seed=seed, enable_vt=vt,
                         enable_initial_delay=delay and not incident,
                         enable_corridor_incident=incident, l3_worker=None, l3_sa=l3_sa,
                         incident_scale=severity)
    obs, info = env.reset()
    g = o = 0
    g_dep = o_dep = 0
    retries = guards = 0
    arr_events = []
    for _ in range(600):
        if mode == "learned":
            masks = info.get("action_masks", {})
            g0 = build_conflict_graph(info.get("requests", []))
            go = {aid: critic_input(aid, obs, g0, k_max=1) for aid in obs}
            gs = compact_global_state(info.get("metrics", {}))
            aa = tr.collect_step(obs, masks, go, gs, device=dev)
            ad = {aid: (a.platform, a.entry) for aid, a in aa.items()}
        else:
            ad = {}  # env falls back to heuristic_action for every train
        obs, rew, term, trunc, info = env.step(ad, lambda_team=0.3)
        m = info.get("metrics", {})
        g += m.get("n_arrivals", 0)
        o += m.get("n_ontime_arr", 0)
        g_dep += m.get("n_granted", 0)
        o_dep += m.get("n_ontime_dep", 0)
        retries += len(info.get("retried_train_ids", []))
        guards += info.get("hard_guard_violations", 0)
        for ev in info.get("l3_transit_events", []):
            ev["priority"] = cfg.trains[ev["train_id"]].priority
            arr_events.append(ev)
        if term or trunc:
            break
    return {"g": g, "o": o, "g_dep": g_dep, "o_dep": o_dep,
            "retries": retries, "guards": guards,
            "status": info["episode_status"], "arr_events": arr_events}


def report(label, eps, cfg):
    st = Counter(r["status"] for r in eps)
    g = sum(r["g"] for r in eps); o = sum(r["o"] for r in eps)
    g_dep = sum(r["g_dep"] for r in eps); o_dep = sum(r["o_dep"] for r in eps)
    ret = sum(r["retries"] for r in eps); gd = sum(r["guards"] for r in eps)
    ds = sorted(ev["arr_delay"] for r in eps for ev in r["arr_events"])
    n = len(ds)
    byp = defaultdict(lambda: [0, 0])
    for r in eps:
        for ev in r["arr_events"]:
            byp[ev["priority"]][1] += 1
            byp[ev["priority"]][0] += int(abs(ev["arr_delay"]) <= ONTIME_THRESHOLD_SEC)
    print(f"== {label} ==")
    print(f"  episodes={dict(st)} arrivals={g} ARR_ontime%={o/max(g,1)*100:.1f} "
          f"| DEP_ontime%={o_dep/max(g_dep,1)*100:.1f} (n_granted={g_dep})")
    print(f"  retries={ret} guards={gd}")
    if n:
        print(f"  arr_delay mean={sum(ds)/n:.1f} p50={ds[n//2]:.1f} "
              f"p90={ds[int(n*0.9)]:.1f} max={ds[-1]:.1f}")
    print("  priority ARR ontime (P1/P2/P3):", {
        p: f"{a[0]}/{a[1]}" for p, a in sorted(byp.items())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="five_station_20t")
    ap.add_argument("--ckpt", required=True, help="trained L2 checkpoint (final.pt / best.pt)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--vt", action="store_true")
    ap.add_argument("--delay", action="store_true")
    ap.add_argument("--incident", action="store_true",
                    help="corridor speed-restriction incident (mutually exclusive with --delay)")
    ap.add_argument("--l3-sa", action="store_true")
    ap.add_argument("--severity", type=float, default=1.0,
                    help="incident strength multiplier (>1 opens the L2 re-dispatch window)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cpu" if args.cpu else "cuda")
    cfg = load_scenario(scenario_name=args.scenario)
    tr = _load_l2(cfg, args.ckpt, dev)

    for mode in ("heuristic", "learned"):
        eps = [run_episode(cfg, tr, args.seeds + s, mode, dev,
                           args.vt, args.delay, args.incident, args.l3_sa,
                           args.severity)
               for s in range(args.seeds)]
        report(f"{mode} ({args.seeds} seeds)", eps, cfg)


if __name__ == "__main__":
    main()
