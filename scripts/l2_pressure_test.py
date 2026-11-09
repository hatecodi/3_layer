#!/usr/bin/env python3
"""Small-scale L2 pressure test: learned L2 vs heuristic L2, graded by difficulty.

Both arms run the SAME L3 executor (the learned corridor worker), so any
difference is attributable to the L2 decision layer:

  learned  arm — L2 action sampled from the trained policy
  heuristic arm — empty action_dict, so the env's built-in ``heuristic_action``
                  picks (platform, entry) for every train

Difficulty is scored objectively per seed by the heuristic arm's mean departure
delay; seeds are then split into terciles (Easy / Medium / Hard) and both arms
are reported per tier.  Both arms always run on the same seed set, so the
comparison is paired.

Usage:
    python scripts/l2_pressure_test.py --scenario five_station_15t \
        --l2 output/8_14/training/phaseA/best.pt \
        --l3 output/8_14/training/phaseB_v3/best.pt \
        --seeds 40 --seed-base 900000 \
        --out-csv output/figures/l2_pressure_15t.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_protocol as ep  # reuse the checkpoint loaders (same architecture)

from src.env.five_station_env import FiveStationEnv
from src.control.resource_manager import build_conflict_graph
from src.env.observation_builder import critic_input
from src.marl.actor_critic import compact_global_state
from src.experiment.scenario_loader import load_scenario


def run_episode(cfg, tr, l3, seed, dev, use_policy: bool, extra_train: bool,
                vt: bool = False):
    """One episode; returns arrival ontime %, mean departure delay, decision share."""
    env = FiveStationEnv(cfg=cfg, seed=seed, enable_vt=vt,
                         enable_initial_delay=True,      # origin bunching = difficulty source
                         enable_corridor_incident=False,
                         l3_worker=l3, enable_extra_train=extra_train)
    obs, info = env.reset()
    n_arr = n_ontime_arr = 0
    dep_total = 0.0
    dep_weight = 0
    pol_dec = heur_dec = 0
    for _ in range(600):
        if use_policy:
            # how many of this step's scheduling events the policy actually decides
            build_set = set(env._pending_trains(lookahead=env.batch_window_sec))
            pol_set = set(info.get("action_masks", {}))
            pol_dec += len(build_set & pol_set)
            heur_dec += len(build_set - pol_set)

            g0 = build_conflict_graph(info.get("requests", []))
            go = {a: critic_input(a, obs, g0, k_max=1) for a in obs}
            gs = compact_global_state(info.get("metrics", {}))
            aa = tr.collect_step(obs, info.get("action_masks", {}), go, gs, device=dev)
            ad = {a: (x.platform, x.entry) for a, x in aa.items()}
        else:
            ad = {}  # empty -> env's built-in heuristic for every train

        obs, _r, term, trunc, info = env.step(ad, lambda_team=0.3)
        m = info.get("metrics", {})
        n_arr += m.get("n_arrivals", 0)
        n_ontime_arr += m.get("n_ontime_arr", 0)
        dep_total += m.get("total_dep_delay", 0.0)
        dep_weight += max(m.get("n_active", 0), 1)
        if term or trunc:
            break
    tot_dec = pol_dec + heur_dec
    return {
        "arr_ontime": n_ontime_arr / max(n_arr, 1) * 100.0,
        "mean_dep": dep_total / max(dep_weight, 1),
        "status": info.get("episode_status", "?"),
        "policy_share": pol_dec / tot_dec * 100.0 if tot_dec else float("nan"),
    }


def _mean(xs):
    xs = [x for x in xs if x == x]  # drop NaN
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="five_station_15t")
    ap.add_argument("--l2", required=True)
    ap.add_argument("--l3", required=True)
    ap.add_argument("--seeds", type=int, default=40, help="number of seeds to grade")
    ap.add_argument("--seed-base", type=int, default=900000,
                    help="held-out seed base (distinct from training seeds)")
    ap.add_argument("--extra-train", action="store_true",
                    help="add a random mid-operation train (extra pressure)")
    ap.add_argument("--vt", action="store_true",
                    help="enable visitor trains — they occupy platform windows, so this is "
                         "the pressure that actually stresses L2's platform selection")
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_scenario(scenario_name=args.scenario)
    tr = ep._load_l2(cfg, args.l2, dev)
    l3 = ep._load_l3(args.l3, dev)
    print(f"scenario={args.scenario}  seeds={args.seeds} @ {args.seed_base}  "
          f"extra_train={args.extra_train}  vt={args.vt}  device={dev}")

    rows = []
    for i in range(args.seeds):
        s = args.seed_base + i
        heur = run_episode(cfg, tr, l3, s, dev, use_policy=False,
                           extra_train=args.extra_train, vt=args.vt)
        learn = run_episode(cfg, tr, l3, s, dev, use_policy=True,
                            extra_train=args.extra_train, vt=args.vt)
        rows.append({"seed": s, "difficulty": heur["mean_dep"],
                     "heur_dep": heur["mean_dep"], "learn_dep": learn["mean_dep"],
                     "heur_ontime": heur["arr_ontime"], "learn_ontime": learn["arr_ontime"],
                     "policy_share": learn["policy_share"],
                     "heur_status": heur["status"], "learn_status": learn["status"]})
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{args.seeds} seeds")

    # tercile split by the objective difficulty score
    order = sorted(range(len(rows)), key=lambda i: rows[i]["difficulty"])
    tiers = ["Easy", "Medium", "Hard"]
    per = len(rows) / 3.0
    for rank, idx in enumerate(order):
        rows[idx]["tier"] = tiers[min(int(rank / per), 2)]

    print(f"\n{'tier':8}{'n':>4}{'difficulty':>12}{'heur_dep':>10}{'learn_dep':>11}"
          f"{'Δdep':>8}{'heur_ontime':>13}{'learn_ontime':>14}{'Δontime':>9}{'pol_share':>11}")
    for t in tiers:
        g = [r for r in rows if r["tier"] == t]
        hd, ld = _mean([r["heur_dep"] for r in g]), _mean([r["learn_dep"] for r in g])
        ho, lo = _mean([r["heur_ontime"] for r in g]), _mean([r["learn_ontime"] for r in g])
        print(f"{t:8}{len(g):>4}{_mean([r['difficulty'] for r in g]):>12.1f}"
              f"{hd:>10.1f}{ld:>11.1f}{ld - hd:>+8.1f}"
              f"{ho:>13.1f}{lo:>14.1f}{lo - ho:>+9.1f}"
              f"{_mean([r['policy_share'] for r in g]):>11.1f}")

    allr = rows
    hd, ld = _mean([r["heur_dep"] for r in allr]), _mean([r["learn_dep"] for r in allr])
    ho, lo = _mean([r["heur_ontime"] for r in allr]), _mean([r["learn_ontime"] for r in allr])
    print(f"{'ALL':8}{len(allr):>4}{'':>12}{hd:>10.1f}{ld:>11.1f}{ld - hd:>+8.1f}"
          f"{ho:>13.1f}{lo:>14.1f}{lo - ho:>+9.1f}"
          f"{_mean([r['policy_share'] for r in allr]):>11.1f}")
    st = {r["learn_status"] for r in rows} | {r["heur_status"] for r in rows}
    print(f"episode statuses: {sorted(st)}")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"saved {args.out_csv}")


if __name__ == "__main__":
    main()
