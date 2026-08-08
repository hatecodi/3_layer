#!/usr/bin/env python3
"""L3 behavior analysis: does the worker's action depend on state?
Bins L3 decisions by cumulative_delay (idx2), next_platforms_free ratio (idx7),
headway_gap (idx0), and reports CRUISE/FAST/SLOW fractions per bin.

Usage:
    python scripts/analyze_l3_behavior.py --scenario five_station_15t \
        --l2 <l2.pt> --l3 <l3.pt> --seeds 800000 --n 15
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="five_station_15t")
    ap.add_argument("--l2", required=True)
    ap.add_argument("--l3", required=True)
    ap.add_argument("--seeds", type=int, default=800000)
    ap.add_argument("--n", type=int, default=15)
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
    lc = CorridorCritic(L3_OBS_DIM, hidden_dim=128); lc.load_state_dict(l3c["l3_critic"]); lc.eval()
    for p in list(la.parameters()) + list(lc.parameters()): p.requires_grad = False
    l3 = L3WorkerTrainer(la, lc, lr=1e-3, device=dev)

    # Collect (obs, action) at each L3 decision by patching worker.select_action
    # inside _advance_trains (the env calls it per section).
    import src.env.five_station_env as envmod
    orig_advance = envmod.FiveStationEnv._advance_trains
    collect = []

    def patched(self, grants):
        orig_select = self._l3_worker.select_action if self._l3_worker else None
        if orig_select is not None:
            def wrap(obs_t):
                act, lp, v = orig_select(obs_t)
                collect.append((obs_t.detach().cpu().numpy(), int(act)))
                return act, lp, v
            self._l3_worker.select_action = wrap
        try:
            return orig_advance(self, grants)
        finally:
            if orig_select is not None:
                self._l3_worker.select_action = orig_select

    envmod.FiveStationEnv._advance_trains = patched
    for s in range(args.seeds, args.seeds + args.n):
        env = FiveStationEnv(cfg=cfg, seed=s, enable_vt=False, enable_initial_delay=True, l3_worker=l3)
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
    envmod.FiveStationEnv._advance_trains = orig_advance

    # analysis
    NAMES = {0: "CRUISE", 1: "FAST", 2: "SLOW"}

    def frac(counts):
        t = sum(counts.values()) or 1
        return {NAMES[k]: round(v / t * 100) for k, v in sorted(counts.items())}

    # 1. action vs cumulative_delay (idx 2, normalized /300)
    bins_d = defaultdict(lambda: defaultdict(int))
    for obs, act in collect:
        cd = obs[2]
        key = "delay<5" if cd < 5 / 300 else ("5-30" if cd < 30 / 300 else ("30-90" if cd < 90 / 300 else ">90"))
        bins_d[key][act] += 1
    print("== action vs cumulative_delay ==")
    for k in ["delay<5", "5-30", "30-90", ">90"]:
        if k in bins_d:
            print(f"  {k:>8}: {frac(bins_d[k])}")

    # 2. action vs next_platforms_free ratio (idx 7)
    bins_p = defaultdict(lambda: defaultdict(int))
    for obs, act in collect:
        pf = obs[7]
        key = "plat>=0.5" if pf >= 0.5 else "plat<0.5"
        bins_p[key][act] += 1
    print("== action vs next_platforms_free ratio ==")
    for k in ["plat>=0.5", "plat<0.5"]:
        if k in bins_p:
            print(f"  {k:>10}: {frac(bins_p[k])}")

    # 3. action vs headway_gap (idx 0)
    bins_h = defaultdict(lambda: defaultdict(int))
    for obs, act in collect:
        hg = obs[0]
        key = "hdw<0.2" if hg < 0.2 else ("0.2-0.5" if hg < 0.5 else "hdw>0.5")
        bins_h[key][act] += 1
    print("== action vs headway_gap ==")
    for k in ["hdw<0.2", "0.2-0.5", "hdw>0.5"]:
        if k in bins_h:
            print(f"  {k:>10}: {frac(bins_h[k])}")

    print(f"total L3 decisions: {len(collect)}")


if __name__ == "__main__":
    main()
