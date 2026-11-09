#!/usr/bin/env python3
"""Per-scale DECISION-layer cost: HMAPPO vs SA vs FlatMAPPO.

All three are measured on the SAME axis — wall-clock the decision layer needs to
run one episode — because the three methods decompose the decision differently:

    hier  (HMAPPO)   L2 = MAPPO collect_step      + L3 = l3_worker.select_action (per SECTION)
    sa    (SA L3)    L2 = MAPPO collect_step      + L3 = sa_optimize_profile     (per TRANSIT)
    flat  (FlatMAPPO) one FlatJointActor forward per pending train (platform+entry+speed
                     in a single 72-way action, so it REPLACES both L2 and L3)

Timing only the L3 layer would put FlatMAPPO at zero (it has no separate L3), so
the comparable quantity is L2+L3 per episode.  Env physics is excluded throughout.

Why HMAPPO gets to reuse the L2 policy for the SA arm: that IS the §6.6.1 protocol
("same L2, only the L3 execution layer switches"), so the SA-vs-hier difference is
attributable to the execution layer alone.

Honest expected shape: per-decision unit costs are ~scale-invariant, so all three
per-episode totals are ~O(N) — the advantage is a constant factor, NOT asymptotic.
Local search is benign on this problem; HMAPPO's case rests on avoiding the
per-transit RE-SOLVE, not on search intractability.

Usage:
    python scripts/bench_l3_decision_cost.py --scales 10 15 20 25 30 --episodes 5
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.experiment.scenario_loader import load_scenario
from src.env.five_station_env import FiveStationEnv
from src.env.observation_builder import get_obs_dim, critic_input
from src.marl.actor_critic import Layer2ActorCritic, compact_global_state
from src.marl.mappo import MAPPOTrainer
from src.marl.l3_worker import CorridorActor, CorridorCritic, L3WorkerTrainer, L3_OBS_DIM
from src.control import l3_sa as l3_sa_mod
from src.control.resource_manager import build_conflict_graph
from src.experiment.run_flat_mappo import (
    FlatJointActor, build_flat_mask, unpack_flat_action,
)

DEFAULT_L2 = ("output/8_16/training/five_station_15t_hierarchical/"
              "20260816_140621/checkpoints/best.pt")
DEFAULT_FLAT = ("output/9_4/training/flat_mappo/20260904_220534/"
                "checkpoints/best.pt")


class Timer:
    def __init__(self):
        self.total = 0.0
        self.calls = 0

    def add(self, dt):
        self.total += dt
        self.calls += 1


def timed(timer, fn):
    def wrapper(*a, **kw):
        t = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            timer.add(time.perf_counter() - t)
    return wrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=int, nargs="+", default=[10, 15, 20, 25, 30])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1,
                    help="independent episode blocks per (scale, method), averaged; "
                         "use >=3 to average out wall-clock noise")
    # "sa-online" is deliberately NOT in the default set: its compute is not
    # reported in the comparison (user decision 2026-09-20).  Pass it explicitly
    # via --methods to measure it anyway.
    ap.add_argument("--methods", type=str, nargs="+", default=["hier", "sa", "flat"])
    ap.add_argument("--l2", type=str, default=DEFAULT_L2,
                    help="L2 manager checkpoint (15t ckpt is the transfer source)")
    ap.add_argument("--flat-ckpt", type=str, default=DEFAULT_FLAT)
    ap.add_argument("--incident", action="store_true",
                    help="corridor speed-restriction arm (the paper's disturbance)")
    ap.add_argument("--seed0", type=int, default=900000)
    ap.add_argument("--out", type=Path,
                    default=Path("output/9.20/l3_decision_cost_scaling.csv"))
    args = ap.parse_args()

    dev = torch.device("cpu")
    ck = torch.load(args.l2, map_location=dev, weights_only=False)
    flat_ck = None
    if "flat" in args.methods:
        flat_ck = torch.load(args.flat_ckpt, map_location=dev, weights_only=False)

    rows = []
    print(f"{'scale':>6} {'method':>7} {'transits':>9} {'sections':>9} "
          f"{'l2_calls':>8} {'l3_calls':>8} {'l2_ms/ep':>9} {'l3_ms/ep':>9} "
          f"{'total ms/ep':>12}")

    for scale in args.scales:
        cfg = load_scenario(scenario_name=f"five_station_{scale}t")
        od = get_obs_dim(cfg)
        ac = Layer2ActorCritic(obs_dim=od, k_local=1, hidden_dim=128)
        ac.load_state_dict(ck["ac_state"]); ac.eval()
        for p in ac.parameters():
            p.requires_grad = False
        l2 = MAPPOTrainer(ac, lr=3e-4, gamma=.99, gae_lambda=.95, clip_ratio=.2,
                          ppo_epochs=5, minibatch_size=256, entropy_coef=.01,
                          value_coef=.5, max_grad_norm=1.0, total_episodes=1, device=dev)

        la = CorridorActor(obs_dim=L3_OBS_DIM, hidden_dim=128)
        cl = CorridorCritic(obs_dim=L3_OBS_DIM, hidden_dim=128)
        cg = CorridorCritic(obs_dim=L3_OBS_DIM, hidden_dim=128)
        if "l3_actor" in ck:
            la.load_state_dict(ck["l3_actor"]); cl.load_state_dict(ck["l3_critic_local"])
            cg.load_state_dict(ck["l3_critic_global"])
        l3 = L3WorkerTrainer(la, cl, cg, lr=1e-3, device=dev)

        flat_actor = None
        if flat_ck is not None:
            flat_actor = FlatJointActor(obs_dim=od, hidden_dim=128)
            flat_actor.load_state_dict(flat_ck["actor"]); flat_actor.eval()

        for method in args.methods:
            t_l2, t_l3 = Timer(), Timer()
            orig_l2 = MAPPOTrainer.collect_step
            orig_l3 = L3WorkerTrainer.select_action
            orig_sa = l3_sa_mod.sa_optimize_profile
            orig_sa_on = l3_sa_mod.sa_online_profile
            orig_dist = FlatJointActor.dist
            if method in ("hier", "sa", "sa-online"):
                MAPPOTrainer.collect_step = timed(t_l2, orig_l2)
            if method == "hier":
                L3WorkerTrainer.select_action = timed(t_l3, orig_l3)
            elif method == "sa":
                l3_sa_mod.sa_optimize_profile = timed(t_l3, orig_sa)
            elif method == "sa-online":
                l3_sa_mod.sa_online_profile = timed(t_l3, orig_sa_on)
            else:
                FlatJointActor.dist = timed(t_l3, orig_dist)

            transits = sections = 0
            rep_ms = []
            try:
              for rep in range(args.repeats):
                t0_l2, t0_l3 = t_l2.total, t_l3.total
                for ep in range(args.episodes):
                    l2.advance_episode()
                    env = FiveStationEnv(
                        cfg=cfg, seed=args.seed0 + rep * 1000 + ep, batch_window_sec=120.0,
                        enable_vt=False, enable_initial_delay=False,
                        enable_corridor_incident=args.incident,
                        l3_worker=(l3 if method == "hier" else None),
                        l3_sa=(method == "sa"),
                        l3_sa_online=(method == "sa-online"))
                    obs, info = env.reset()
                    done = False
                    while not done:
                        masks = info.get("action_masks", {})
                        if method == "flat":
                            ad, speeds = {}, {}
                            for tid in info.get("pending", []):
                                m = masks.get(tid)
                                if not m or not any(m[0]) or not any(m[1]):
                                    continue
                                obs_t = torch.tensor(obs[tid], dtype=torch.float32)
                                fmask = build_flat_mask(m[0], m[1])
                                with torch.no_grad():
                                    act = flat_actor.dist(obs_t, fmask).sample().item()
                                p, e, s = unpack_flat_action(act)
                                ad[tid] = (p, e); speeds[tid] = s
                            env._flat_speed_choices = speeds
                        else:
                            cgx = build_conflict_graph(info.get("requests", []))
                            gobs = {a: critic_input(a, obs, cgx, k_max=1) for a in obs}
                            gs = compact_global_state(info.get("metrics", {}))
                            with torch.no_grad():
                                aa = l2.collect_step(obs, masks, gobs, gs, device=dev)
                            ad = {k: (v.platform, v.entry) for k, v in aa.items()}
                        obs, r, term, trunc, info = env.step(ad, lambda_team=l2.lam_team)
                        done = term or trunc
                        # _l3_transit_events is reset each _advance_trains -> read per step
                        transits += len(info.get("l3_transit_events", []))
                    sections += len(env._l3_behavior)
                # per-repeat per-episode cost, so dispersion across repeats is
                # visible (wall-clock on a shared CPU is noisy; the 20t run once
                # came out 30% high, which a single pass cannot reveal)
                rep_ms.append(((t_l2.total - t0_l2) + (t_l3.total - t0_l3))
                              * 1000 / args.episodes)
            finally:
                MAPPOTrainer.collect_step = orig_l2
                L3WorkerTrainer.select_action = orig_l3
                l3_sa_mod.sa_optimize_profile = orig_sa
                l3_sa_mod.sa_online_profile = orig_sa_on
                FlatJointActor.dist = orig_dist

            ep_n = args.episodes * args.repeats
            row = {
                "scale": scale, "method": method, "episodes": ep_n,
                "repeats": args.repeats,
                "incident": int(args.incident),
                "transits": transits, "sections": sections,
                "l2_calls": t_l2.calls, "l3_calls": t_l3.calls,
                "l2_ms_total": f"{t_l2.total * 1000:.2f}",
                "l3_ms_total": f"{t_l3.total * 1000:.2f}",
                "decision_ms_total": f"{(t_l2.total + t_l3.total) * 1000:.2f}",
                "ms_per_episode": f"{(t_l2.total + t_l3.total) * 1000 / ep_n:.4f}",
                "ms_per_episode_std": f"{statistics.stdev(rep_ms):.4f}" if len(rep_ms) > 1 else "0",
                "ms_per_transit": f"{(t_l2.total + t_l3.total) * 1000 / max(transits, 1):.4f}",
                "ms_per_section": f"{(t_l2.total + t_l3.total) * 1000 / max(sections, 1):.4f}",
            }
            rows.append(row)
            print(f"{scale:>6} {method:>7} {transits:>9} {sections:>9} "
                  f"{t_l2.calls:>8} {t_l3.calls:>8} "
                  f"{t_l2.total * 1000 / ep_n:>9.2f} {t_l3.total * 1000 / ep_n:>9.2f} "
                  f"{(t_l2.total + t_l3.total) * 1000 / ep_n:>12.2f}")

    if rows:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nCSV: {args.out}")


if __name__ == "__main__":
    main()
