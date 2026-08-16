#!/usr/bin/env python3
"""Paper figures 3 & 5 from eval_protocol --out-csv data (CSV-first).

Fig 3: three-baseline arrival-ontime comparison (overall + priority P1/P2/P3).
Fig 5: incident-affected-subset arrival delay (mean + p90) — coordination value.

Usage:
    python scripts/plot_paper_compare.py \
        --csv output/figures/8.12图/data/compare.csv \
        --out-dir output/figures/8.12图
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FONT = {"font.family": "serif", "font.size": 12, "mathtext.fontset": "dejavuserif"}
plt.rcParams.update(FONT)
COLORS = {"learned": "#c0392b", "sa": "#2874a6", "cruise": "#7d6608"}


def load(path: Path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["mode"]] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("output/figures/8.12图/data/compare.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("output/figures/8.12图"))
    args = ap.parse_args()
    rows = load(args.csv)
    modes = ["learned", "sa", "cruise"]

    # ---------- Fig 3: ontime% overall + priority ----------
    metrics = ["ontime", "p1", "p2", "p3"]
    labels = ["Overall", "P1\n(Fuxing)", "P2\n(Hexie)", "P3\n(Slow)"]
    x = np.arange(len(metrics))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(modes):
        vals = [float(rows[m][k]) for k in metrics]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=m, color=COLORS[m],
                      edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("arrival ontime (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title="method")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Arrival ontime by method & train priority (incident, held-out 3×60 seeds)")
    plt.tight_layout()
    fig.savefig(args.out_dir / "fig3_ontime_priority.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out_dir / 'fig3_ontime_priority.png'}")

    # ---------- Fig 5: incident-affected arrival delay (mean + p90) ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    x2 = np.arange(len(modes))
    width2 = 0.36
    means = [float(rows[m]["aff_mean"]) for m in modes]
    p90s = [float(rows[m]["aff_p90"]) for m in modes]
    b1 = ax.bar(x2 - width2 / 2, means, width2, label="affected mean", color="#c0392b",
                edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x2 + width2 / 2, p90s, width2, label="affected p90", color="#2874a6",
                edgecolor="black", linewidth=0.5)
    for b, v in zip(list(b1) + list(b2), means + p90s):
        ax.text(b.get_x() + b.get_width() / 2, v + 10, f"{v:.0f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x2)
    ax.set_xticklabels(modes)
    ax.set_ylabel("realised arrival delay (s)")
    ax.set_title("Incident-affected transit arrival delay (coordination value)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(args.out_dir / "fig5_affected_delay.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out_dir / 'fig5_affected_delay.png'}")


if __name__ == "__main__":
    main()
