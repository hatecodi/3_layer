#!/usr/bin/env python3
"""Train running diagram (列车运行图) for L3 speed-control modes.

Runs the same closure episode with frozen seed1 L2 + different L3 executors
(SA-online, BC, learned scratch, BC+PPO) and plots time-distance trajectories.
Straight segments join station departure -> next-station arrival; horizontal
segments are station dwell.  The closure window is shaded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval_protocol import (
    _load_l2, _load_l3, build_conflict_graph, compact_global_state, critic_input)
from src.env.five_station_env import FiveStationEnv
from src.experiment.scenario_loader import load_scenario

SEED = 960006
STATIONS = ["A", "B", "C", "D", "E"]
# real km (A->B 140, B->C 101, C->D 341, D->E 256)
STATION_KM = {"A": 0, "B": 140, "C": 241, "D": 582, "E": 838}

MODES = [
    ("SA-online", None, True),
    ("BC L3", "output/11_09/bc_l3_saonline.pt", False),
    ("learned L3 (scratch)",
     "output/11_09/training_closure_phaseB_seed1L2_2k/"
     "five_station_15t_hierarchical/20261109_193526/checkpoints/final.pt", False),
    ("BC + PPO",
     "output/11_09/training_l3_bc_ppo_2k/"
     "five_station_15t_hierarchical/202609?/checkpoints/final.pt", False),
]


def run_mode(cfg, tr, mode_name, l3_ckpt, sa_online, seed=SEED):
    dev = torch.device("cpu")
    worker = None if sa_online else _load_l3(l3_ckpt, dev)
    env = FiveStationEnv(
        cfg=cfg, seed=seed, enable_vt=False, enable_initial_delay=False,
        enable_corridor_incident=False, enable_corridor_closure=True,
        l3_worker=worker, l3_sa_online=sa_online, yield_entry_cap=True,
        control_coverage="base")
    obs, info = env.reset()
    closures = dict(env._corridor_closures)
    events = []
    for _ in range(600):
        rel = set(info.get("control_relevant_ids", []))
        g0 = build_conflict_graph(info.get("requests", []))
        go = {aid: critic_input(aid, obs, g0, k_max=1) for aid in obs}
        gs = compact_global_state(info.get("metrics", {}))
        aa = tr.collect_step(obs, info.get("action_masks", {}), go, gs,
                             device=dev)
        ad = {aid: (a.platform, a.entry) for aid, a in aa.items()
              if aid in rel}
        obs, _, term, trunc, info = env.step(ad, lambda_team=0.3)
        events.extend(info.get("l3_transit_events", []))
        if term or trunc:
            break
    return events, closures, info.get("episode_status", "?")


def trajectories(events):
    """{train_id: [(station, time), ...]} sorted in time."""
    by = {}
    for e in events:
        by.setdefault(e["train_id"], []).extend([
            (e["dep_station"], float(e["entry"])),
            (e["arr_station"], float(e["actual_arrival"])),
        ])
    for tid in by:
        by[tid].sort(key=lambda x: x[1])
    return by


def main():
    cfg = load_scenario(scenario_name="five_station_15t")
    dev = torch.device("cpu")
    tr = _load_l2(
        cfg,
        "output/11_09/training_closure_l2_2k_seed1/five_station_15t_hierarchical/"
        "20261109_190622/checkpoints/final.pt",
        dev)

    # Resolve the BC+PPO checkpoint by globbing the run dir (timestamp may vary).
    import glob
    bcp = glob.glob(
        "output/11_09/training_l3_bc_ppo_2k/five_station_15t_hierarchical/*/"
        "checkpoints/final.pt")
    modes = list(MODES)
    if bcp:
        modes[3] = (modes[3][0], bcp[0], False)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True, sharey=True)
    axes = axes.ravel()
    palette = plt.get_cmap("tab20")
    for ax, (name, ckpt, sa_online) in zip(axes, modes):
        torch.manual_seed(SEED)
        events, closures, status = run_mode(cfg, tr, name, ckpt, sa_online)
        traj = trajectories(events)
        # closure shading
        for cid, (t0, t1) in closures.items():
            ax.axvspan(t0, t1, color="#d62728", alpha=0.10, lw=0)
            ax.text((t0 + t1) / 2, 20, f"closure {cid}", color="#a00000",
                    fontsize=8, ha="center", va="bottom", rotation=90)
        for s in STATIONS:
            ax.axhline(STATION_KM[s], color="#555555", lw=0.8, alpha=0.6)
        ids = sorted(traj)
        highlight = {"G230", "G238", "G1380", "G1380_s1", "G1616",
                     "G1616_s3", "G5038", "G1487"}
        hcolors = {}
        hi = 0
        for tid in ids:
            train = cfg.trains[tid]
            pts = traj[tid]
            xs = [p[1] for p in pts]
            ys = [STATION_KM[p[0]] for p in pts]
            if tid in highlight:
                col = palette(hi % 20)
                hcolors[tid] = col
                hi += 1
                ax.plot(xs, ys, color=col,
                        lw=2.4 if train.priority == 1 else 1.7, alpha=0.95)
                ax.text(xs[-1], ys[-1], tid, fontsize=7, color=col,
                        ha="right", va="bottom")
            else:
                ax.plot(xs, ys, color="#999999", lw=0.8, alpha=0.25)
        handles = [plt.Line2D([], [], color=c, lw=2, label=t)
                   for t, c in hcolors.items()]
        ax.legend(handles=handles, fontsize=6.5, loc="lower right",
                  ncol=2, framealpha=0.85)
        ax.set_title(f"{name}  (seed {SEED}, {status})", fontsize=11)
        ax.set_yticks([STATION_KM[s] for s in STATIONS])
        ax.set_yticklabels(STATIONS)
        ax.invert_yaxis()
        ax.set_xlim(0, max((p[1] for tr_ in traj.values() for p in tr_),
                           default=1000) + 20)
        ax.set_ylim(-20, STATION_KM["E"] + 20)
        ax.grid(axis="x", lw=0.3, alpha=0.4)
        ax.set_xlabel("simulation time (s)")
        ax.set_ylabel("station (A top → E bottom)")
    fig.suptitle("Train running diagrams — closure episode, frozen seed1 L2",
                 fontsize=13)
    fig.tight_layout()
    out = Path("output/11_09/l3_mode_trajectories.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
