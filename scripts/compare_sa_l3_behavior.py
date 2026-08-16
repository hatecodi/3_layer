#!/usr/bin/env python3
"""Compare SA vs learned-L3 per-section speed-mode distributions.

Runs BOTH under the same seeds / incident and reads the env's _l3_behavior
(mode, section_speed_factor, priority) recorded per section, then reports:
  - overall CRUISE/FAST/SLOW fractions
  - by incident-restricted (factor<1) vs clear section
  - by priority class

Usage:
    python scripts/compare_sa_l3_behavior.py --scenario five_station_15t \
        --l2 <l2.pt> --l3 <l3.pt> --seeds 800000 --n 15 --incident
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
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

NAMES = {0: "CRUISE", 1: "FAST", 2: "SLOW"}


def _report(label, behavior):
    """behavior: list of (mode, section_speed_factor, priority).
    Returns (by_overall, by_rest, by_prio) counts dicts."""
    by_overall = defaultdict(int)
    by_rest = defaultdict(lambda: defaultdict(int))
    by_prio = defaultdict(lambda: defaultdict(int))
    for mode, sf, prio in behavior:
        by_overall[mode] += 1
        by_rest["restricted" if sf < 1.0 else "clear"][mode] += 1
        by_prio[f"P{prio}"][mode] += 1

    def frac(cnt):
        t = sum(cnt.values()) or 1
        return {NAMES[k]: round(v / t * 100) for k, v in sorted(cnt.items())}

    print(f"\n== {label}  (n_sections={len(behavior)}) ==")
    print(f"  overall  : {frac(by_overall)}")
    for k in ("clear", "restricted"):
        if by_rest[k]:
            print(f"  {k:>10}: {frac(by_rest[k])}")
    for p in sorted(by_prio):
        print(f"  {p:>10}: {frac(by_prio[p])}")
    return by_overall, by_rest, by_prio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="five_station_15t")
    ap.add_argument("--l2", required=True)
    ap.add_argument("--l3", required=True)
    ap.add_argument("--seeds", type=int, default=800000)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--incident", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write per-section speed-mode distributions to CSVs here")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_scenario(scenario_name=args.scenario)
    obs_dim = get_obs_dim(cfg)
    ckpt = torch.load(args.l2, map_location=dev, weights_only=False)
    ac = Layer2ActorCritic(obs_dim=obs_dim, k_local=1, hidden_dim=128)
    ac.load_state_dict(ckpt["ac_state"]); ac.eval()
    for p in ac.parameters(): p.requires_grad = False
    tr = MAPPOTrainer(ac, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2, ppo_epochs=5,
                      minibatch_size=256, entropy_coef=0.01, value_coef=0.5, max_grad_norm=1.0,
                      total_episodes=1, device=dev)
    l3c = torch.load(args.l3, map_location=dev, weights_only=False)
    la = CorridorActor(L3_OBS_DIM, hidden_dim=128); la.load_state_dict(l3c["l3_actor"]); la.eval()
    lc_local = CorridorCritic(L3_OBS_DIM, hidden_dim=128); lc_local.load_state_dict(l3c["l3_critic_local"]); lc_local.eval()
    lc_global = CorridorCritic(L3_OBS_DIM, hidden_dim=128); lc_global.load_state_dict(l3c["l3_critic_global"]); lc_global.eval()
    for p in list(la.parameters()) + list(lc_local.parameters()) + list(lc_global.parameters()): p.requires_grad = False
    l3 = L3WorkerTrainer(la, lc_local, lc_global, lr=1e-3, device=dev)

    def run(mode):
        beh = []
        for s in range(args.seeds, args.seeds + args.n):
            env = FiveStationEnv(cfg=cfg, seed=s, enable_vt=False,
                                 enable_initial_delay=not args.incident,
                                 enable_corridor_incident=args.incident,
                                 l3_worker=(l3 if mode == "learned" else None),
                                 l3_sa=(mode == "sa"))
            obs, info = env.reset()
            for _ in range(600):
                masks = info.get("action_masks", {})
                g0 = build_conflict_graph(info.get("requests", []))
                go = {aid: critic_input(aid, obs, g0, k_max=1) for aid in obs}
                gs = compact_global_state(info.get("metrics", {}))
                aa = tr.collect_step(obs, masks, go, gs, device=dev)
                ad = {aid: (a.platform, a.entry) for aid, a in aa.items()}
                obs, rew, term, trunc, info = env.step(ad, lambda_team=0.3)
                if term or trunc:
                    break
            beh.extend(env._l3_behavior)
        return beh

    results = {"SA": _report("SA", run("sa")),
               "learned-L3": _report("learned-L3", run("learned"))}

    if args.out_dir:
        import csv as _csv
        args.out_dir.mkdir(parents=True, exist_ok=True)
        # overall mode counts/pct per method
        with open(args.out_dir / "behavior_overall.csv", "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["method", "n_sections", "CRUISE", "FAST", "SLOW"])
            for m, (ov, _r, _p) in results.items():
                t = sum(ov.values()) or 1
                w.writerow([m, t] + [round(ov[k] / t * 100, 1) for k in (0, 1, 2)])
        # restricted vs clear per method
        with open(args.out_dir / "behavior_restricted.csv", "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["method", "sections", "CRUISE", "FAST", "SLOW"])
            for m, (_o, rest, _p) in results.items():
                for k in ("clear", "restricted"):
                    if rest[k]:
                        t = sum(rest[k].values()) or 1
                        w.writerow([m, k] + [round(rest[k][i] / t * 100, 1) for i in (0, 1, 2)])
        # by priority per method
        with open(args.out_dir / "behavior_priority.csv", "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["method", "priority", "CRUISE", "FAST", "SLOW"])
            for m, (_o, _r, prio) in results.items():
                for p in sorted(prio):
                    t = sum(prio[p].values()) or 1
                    w.writerow([m, p] + [round(prio[p][i] / t * 100, 1) for i in (0, 1, 2)])
        print(f"saved behavior CSVs to {args.out_dir}")


if __name__ == "__main__":
    main()
