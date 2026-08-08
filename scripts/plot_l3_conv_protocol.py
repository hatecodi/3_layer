#!/usr/bin/env python3
"""Paper-quality multi-seed L3 convergence plot: mean±std band across seeds,
rolling smoothing, plateau region marked.

Usage:
    python scripts/plot_l3_conv_protocol.py --csv output/figures/l3v3_conv_15t_ms.csv \
        --out output/figures/l3v3_conv_15t_ms.png
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


def _load(path: Path):
    by_seed = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            s = int(r["seed"])
            by_seed[s].append({k: (float(r[k]) if r[k] not in ("", "nan") else float("nan"))
                               for k in ("episode", "l3_reward", "entropy", "policy_loss",
                                         "value_loss", "ontime_pct", "mean_dep")})
    return {s: sorted(v, key=lambda x: x["episode"]) for s, v in by_seed.items()}


def _rolling(y, w=5):
    y = np.asarray(y, dtype=float)
    if w <= 1 or len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


def _band(by_seed, key, episodes, w):
    mat = np.full((len(episodes), len(by_seed)), np.nan)
    for si, (s, rows) in enumerate(sorted(by_seed.items())):
        m = {r["episode"]: r[key] for r in rows if not np.isnan(r[key])}
        for i, e in enumerate(episodes):
            mat[i, si] = m.get(int(e), np.nan)
    mean = np.nanmean(mat, axis=1)
    std = np.nanstd(mat, axis=1)
    return _rolling(mean, w), _rolling(std, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("output/figures/l3v3_conv_ms.png"))
    ap.add_argument("--smooth", type=int, default=5)
    args = ap.parse_args()

    by_seed = _load(args.csv)
    n_seeds = len(by_seed)
    eps = sorted({r["episode"] for rows in by_seed.values() for r in rows})
    ep = np.array(eps)

    panels = [
        ("l3_reward", "L3 reward", "#c0392b"),
        ("entropy", "entropy", "#8e44ad"),
        ("policy_loss", "L3 policy loss", "#2874a6"),
        ("value_loss", "L3 value loss", "#d35400"),
        ("ontime_pct", "eval ontime (%)", "#1e8449"),
        ("mean_dep", "eval mean_dep (s)", "#7d6608"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (key, title, color) in zip(axes.flat, panels):
        m, s = _band(by_seed, key, ep, args.smooth)
        x = ep[:len(m)]
        ax.plot(x, m, color=color, lw=2, label=f"mean (n={n_seeds})")
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.25, label="±1 std")
        # per-seed raw points
        for si, (sd, rows) in enumerate(sorted(by_seed.items())):
            xs = [r["episode"] for r in rows if not np.isnan(r[key])]
            ys = [r[key] for r in rows if not np.isnan(r[key])]
            ax.scatter(xs, ys, s=5, alpha=0.3, color=plt.cm.viridis(si / max(n_seeds - 1, 1)))
        # plateau region: last 25% of x range
        xm = x[-1] if len(x) else 0
        p0 = x[0] + 0.75 * (xm - x[0])
        ax.axvspan(p0, xm, color="gray", alpha=0.08)
        if key in ("ontime_pct",):
            ax.set_ylim(50, 100)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("episode"); ax.grid(True, alpha=0.3)
        if key in ("l3_reward", "ontime_pct"):
            ax.legend(fontsize=9)
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"saved {args.out} (seeds={n_seeds}, eval-every={max([0]+[ep[i+1]-ep[i] for i in range(len(ep)-1)])})")


if __name__ == "__main__":
    main()
