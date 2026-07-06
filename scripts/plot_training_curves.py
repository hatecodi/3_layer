"""Plot MAPPO training curves with heuristic baseline — publication style."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Heuristic baseline (100 episodes, five_station, with bounds fix)
HEURISTIC_DEP_DELAY = 25.0
HEURISTIC_ONTIME = 0.665

SMOOTH_WINDOW = 201  # episodes


def smooth(data: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def rolling_std(data: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Rolling standard deviation, same length as smooth()."""
    if len(data) < window:
        return np.full_like(data, np.std(data))
    ret = np.zeros(len(data) - window + 1)
    for i in range(len(ret)):
        ret[i] = np.std(data[i:i + window])
    return ret


def plot(csv_path: str):
    rows = []
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            if line.strip():
                values = line.strip().split(",")
                row = {}
                for h, v in zip(header, values):
                    try: row[h] = float(v)
                    except ValueError: row[h] = v
                rows.append(row)

    n_eps = len(rows)
    episodes = np.array([r["episode"] for r in rows])
    rewards = np.array([r.get("mean_reward", 0) for r in rows])
    ontime = np.array([r["ontime_rate"] for r in rows])
    dep_dly = np.array([r.get("mean_dep_delay", 0) for r in rows])

    ep_sm = smooth(episodes)
    rw_sm = smooth(rewards)
    rw_std = rolling_std(rewards)
    ot_sm = smooth(ontime)
    ot_std = rolling_std(ontime)
    dd_sm = smooth(dep_dly)
    dd_std = rolling_std(dep_dly)

    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.labelsize": 12, "axes.titlesize": 13,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    })

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    c_mappo = "#2166ac"
    c_heuristic = "#b2182b"
    lw = 1.4

    # -- Reward --
    ax = axes[0]
    ax.fill_between(ep_sm, rw_sm - rw_std, rw_sm + rw_std,
                    color=c_mappo, alpha=0.12, linewidth=0)
    ax.plot(ep_sm, rw_sm, color=c_mappo, linewidth=lw, label="MAPPO")
    ax.set_ylabel("Mean Reward")
    ax.set_xlabel("Episode")
    ax.set_title("Reward")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False)

    # -- On-Time --
    ax = axes[1]
    ax.fill_between(ep_sm, np.clip(ot_sm - ot_std, 0, 1), np.clip(ot_sm + ot_std, 0, 1),
                    color=c_mappo, alpha=0.12, linewidth=0)
    ax.plot(ep_sm, ot_sm, color=c_mappo, linewidth=lw, label="MAPPO")
    ax.axhline(y=HEURISTIC_ONTIME, color=c_heuristic, linestyle="--",
               linewidth=1.3, label=f"Heuristic ({HEURISTIC_ONTIME*100:.0f}%)")
    ax.set_ylabel("On-Time Rate")
    ax.set_xlabel("Episode")
    ax.set_ylim(0.35, 1.05)
    ax.set_title("On-Time Rate")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False)

    # -- Departure Delay --
    ax = axes[2]
    ax.fill_between(ep_sm, np.maximum(dd_sm - dd_std, 0), dd_sm + dd_std,
                    color=c_mappo, alpha=0.12, linewidth=0)
    ax.plot(ep_sm, dd_sm, color=c_mappo, linewidth=lw, label="MAPPO")
    ax.axhline(y=HEURISTIC_DEP_DELAY, color=c_heuristic, linestyle="--",
               linewidth=1.3, label=f"Heuristic ({HEURISTIC_DEP_DELAY:.0f}s)")
    ax.set_ylabel("Departure Delay (s)")
    ax.set_xlabel("Episode")
    ax.set_title("Mean Departure Delay")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False)

    fig.text(0.5, -0.02,
             f"Figure: MAPPO training ({n_eps} episodes).  "
             f"Solid lines = {SMOOTH_WINDOW}-episode moving average; "
             f"shaded bands = ±1 rolling std.  "
             f"Dashed = heuristic baseline ({HEURISTIC_DEP_DELAY:.0f}s, {HEURISTIC_ONTIME*100:.0f}%).",
             ha="center", fontsize=8, style="italic")

    plt.tight_layout()
    out_path = Path(csv_path).parent / "training_curves.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

    # Summary
    last_100 = rows[-100:]
    avg_rw = np.mean([r.get("mean_reward", 0) for r in last_100])
    avg_ot = np.mean([r["ontime_rate"] for r in last_100])
    avg_dd = np.mean([r.get("mean_dep_delay", 0) for r in last_100])
    best = max(rows, key=lambda r: r.get("mean_reward", 0))
    print(f"\nMAPPO ({n_eps}ep, last 100):  "
          f"reward={avg_rw:.1f}  ontime={avg_ot*100:.1f}%  dep_dly={avg_dd:.1f}s")
    print(f"Heuristic:  dep_dly={HEURISTIC_DEP_DELAY:.1f}s  ontime={HEURISTIC_ONTIME*100:.1f}%")
    print(f"Best ep={best['episode']:.0f}:  {float(best['ontime_rate'])*100:.1f}%  {best.get('mean_dep_delay',0):.1f}s")
    print(f"Reward std: first500={np.std(rewards[:500]):.1f}  last500={np.std(rewards[-500:]):.1f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        output_dir = Path("output/five_station")
        csvs = sorted(output_dir.glob("*/training_metrics.csv"),
                      key=lambda p: p.stat().st_mtime)
        if not csvs:
            print("No CSV found")
            sys.exit(1)
        csv_path = str(csvs[-1])
    plot(csv_path)
