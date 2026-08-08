#!/usr/bin/env python3
"""Paper-quality L3 convergence plot: dense points + rolling-mean smoothing +
multi-seed mean/std shaded band.

Usage:
    python scripts/plot_l3v3_convergence.py \
        --csv output/figures/l3v3_multiseed.csv \
        --out output/figures/l3v3_convergence_paper.png
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FONT = {"font.family": "serif", "font.size": 11, "mathtext.fontset": "dejavuserif"}
plt.rcParams.update(FONT)
COLORS = {"CRUISE": "#4477AA", "FAST": "#EE6677", "SLOW": "#228833"}
SEED_CMAP = plt.cm.viridis  # noqa


def _load(path: Path) -> dict:
    by_seed = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            s = int(r["seed"])
            by_seed[s].append({
                "episode": int(r["episode"]),
                "l3_reward": float(r.get("l3_reward", 0.0)),
                "total_reward": float(r.get("total_reward", 0.0)),
                "entropy": float(r.get("entropy", 0.0)),
                "ontime_pct": float(r["ontime_pct"]) if r.get("ontime_pct") else None,
                "mean_dep": float(r["mean_dep"]) if r.get("mean_dep") else None,
            })
    return {s: sorted(v, key=lambda x: x["episode"]) for s, v in by_seed.items()}


def _rolling(y: np.ndarray, w: int = 9) -> np.ndarray:
    if w <= 1 or len(y) < w:
        return y.astype(float)
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="valid")


def _band(by_seed: dict, key: str, episodes: np.ndarray, w: int):
    """Mean/std across seeds at each episode (NaN where a seed lacks the point)."""
    mat = np.full((len(episodes), len(by_seed)), np.nan)
    for si, (s, rows) in enumerate(sorted(by_seed.items())):
        ep_map = {r["episode"]: r[key] for r in rows if r[key] is not None}
        for i, e in enumerate(episodes):
            mat[i, si] = ep_map.get(int(e), np.nan)
    mean = np.nanmean(mat, axis=1)
    std = np.nanstd(mat, axis=1)
    return _rolling(mean, w), _rolling(std, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("output/figures/l3v3_multiseed.csv"))
    ap.add_argument("--out", type=Path, default=Path("output/figures/l3v3_convergence_paper.png"))
    ap.add_argument("--smooth", type=int, default=9, help="rolling-mean window")
    args = ap.parse_args()

    by_seed = _load(args.csv)
    n_seeds = len(by_seed)
    if n_seeds < 2:
        print(f"WARNING: {n_seeds} seed(s) — shaded band needs >=2.")

    # reward is dense (every 25); eval (ontime/dep/entropy) at its own grid
    dense_eps = sorted({r["episode"] for rows in by_seed.values() for r in rows
                        if r["l3_reward"] is not None})
    eval_eps = sorted({r["episode"] for rows in by_seed.values() for r in rows
                       if r["ontime_pct"] is not None})
    dense = np.array(dense_eps)
    ev = np.array(eval_eps)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # --- L3 reward (dense, smoothed, band) ---
    ax = axes[0, 0]
    m, s = _band(by_seed, "l3_reward", dense, args.smooth)
    x = dense[:len(m)]
    ax.plot(x, m, color="#c0392b", lw=2, label=f"mean (n={n_seeds})")
    ax.fill_between(x, m - s, m + s, color="#c0392b", alpha=0.25, label="±1 std")
    for si, (sd, rows) in enumerate(sorted(by_seed.items())):
        ys = [r["l3_reward"] for r in rows if r["l3_reward"] is not None]
        xe = [r["episode"] for r in rows if r["l3_reward"] is not None]
        ax.scatter(xe, ys, s=4, alpha=0.35, color=plt.cm.viridis(si / max(n_seeds - 1, 1)))
    ax.set_title("(a) L3 worker reward", fontsize=12)
    ax.set_xlabel("episode"); ax.set_ylabel("mean L3 reward (25-ep)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    # --- Total episode reward (dense, smoothed, band) ---
    ax = axes[0, 1]
    m, s = _band(by_seed, "total_reward", dense, args.smooth)
    x = dense[:len(m)]
    ax.plot(x, m, color="#8e44ad", lw=2, label=f"mean (n={n_seeds})")
    ax.fill_between(x, m - s, m + s, color="#8e44ad", alpha=0.25, label="±1 std")
    ax.set_title("(b) total episode reward", fontsize=12)
    ax.set_xlabel("episode"); ax.set_ylabel("mean total reward")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    # --- ontime % (eval grid, smoothed, band) ---
    ax = axes[1, 0]
    m, s = _band(by_seed, "ontime_pct", ev, max(3, args.smooth // 2))
    x = ev[:len(m)]
    ax.plot(x, m, color="#1e8449", lw=2, label=f"mean (n={n_seeds})")
    ax.fill_between(x, m - s, m + s, color="#1e8449", alpha=0.25, label="±1 std")
    ax.set_title("(c) eval ontime", fontsize=12)
    ax.set_xlabel("episode"); ax.set_ylabel("ontime (%)")
    ax.set_ylim(60, 80); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    # --- mean_dep (eval grid, smoothed, band) ---
    ax = axes[1, 1]
    m, s = _band(by_seed, "mean_dep", ev, max(3, args.smooth // 2))
    x = ev[:len(m)]
    ax.plot(x, m, color="#2874a6", lw=2, label=f"mean (n={n_seeds})")
    ax.fill_between(x, m - s, m + s, color="#2874a6", alpha=0.25, label="±1 std")
    ax.set_title("(d) eval mean departure delay", fontsize=12)
    ax.set_xlabel("episode"); ax.set_ylabel("mean dep delay (s)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
