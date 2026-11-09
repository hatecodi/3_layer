#!/usr/bin/env python3
"""STEP 2 of 2: figure ONLY.  Reads the tidy CSV, computes nothing.

The CSV is produced by `scripts/aggregate_tail_by_scale.py` (which in turn reads
the per-transit rows saved by `scripts/collect_tail_eval.py`).  Splitting it this
way means the figure can be restyled or discarded without touching the analysis,
and the numbers in the paper come from the CSV, not from a plot script.

    scale, mode, subset, stat, value, ci_lo, ci_hi, ci_halfwidth,
    cross_run_std, n_episodes, n_runs

Bar  = `value` (mean of the worst 10% of the POOLED delay distribution by default).
Whisker = `ci_lo`/`ci_hi` — a 95% EPISODE-level bootstrap CI (see the aggregator's
docstring: episode is the iid unit, and neither a block std nor a cross-run std is
the right bar).
Small rotated digit at the bar base = `n_runs`, because run coverage is uneven —
30t hierarchical has ONE training run, so its bar carries no cross-run support.

Usage:
    python scripts/plot_tail_by_scale.py --csv output/9.21/tail_by_scale.csv \
        --stat tail10 --subset all --out-dir output/9.21
    python scripts/plot_tail_by_scale.py --csv ... --stat p90 --subset affected_plan
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODE_ORDER = ["learned", "flat", "sa", "sa-online", "sa-nofc", "fcfs", "cruise"]
MODE_LABEL = {
    "learned": "Hierarchical (ours)", "flat": "Flat MAPPO", "sa": "SA (oracle)",
    "sa-online": "SA (partial obs)", "sa-nofc": "SA (no forecast)",
    "fcfs": "FCFS (rule)", "cruise": "CRUISE (no coord.)",
}
MODE_COLOR = {
    "learned": "#1f77b4", "flat": "#d62728", "sa": "#2ca02c",
    "sa-online": "#98df8a", "sa-nofc": "#8c564b", "fcfs": "#7f7f7f",
    "cruise": "#bbbbbb",
}
STAT_LABEL = {
    "tail10": "worst-decile mean", "tail10abs": "worst-decile mean of |delay|",
    "p90": "p90", "p95": "p95", "max": "max", "mean": "mean",
}
SUBSET_LABEL = {"all": "all transits", "affected": "affected (realized)",
                "unaffected": "unaffected transits (dilution-free)",
                "affected_plan": "affected (planned, policy-independent)"}


def _f(v):
    if v is None or v == "":
        return None
    try:
        x = float(v)
        return None if x != x else x
    except ValueError:
        return None


def load(csv_path, stat, subset):
    rows = []
    for r in csv.DictReader(open(csv_path)):
        if r["stat"] != stat or r["subset"] != subset:
            continue
        rows.append({
            "scale": int(r["scale"]), "mode": r["mode"],
            "value": _f(r["value"]), "ci_lo": _f(r["ci_lo"]), "ci_hi": _f(r["ci_hi"]),
            "n_runs": int(r["n_runs"] or 0), "n_episodes": int(r["n_episodes"] or 0),
            "cross_run_std": _f(r["cross_run_std"]),
        })
    if not rows:
        raise SystemExit(f"no rows with stat={stat} subset={subset} in {csv_path}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=PROJECT_ROOT / "output/9.21/tail_by_scale.csv")
    ap.add_argument("--stat", default="tail10",
                    choices=["tail10", "tail10abs", "p90", "p95", "max", "mean"])
    ap.add_argument("--subset", default="all",
                    choices=["all", "affected", "unaffected", "affected_plan"])
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "output/9.21")
    ap.add_argument("--log-y", action="store_true", help="log scale (tails span 1 order of magnitude)")
    ap.add_argument("--group-width", type=float, default=0.8)
    args = ap.parse_args()

    rows = load(args.csv, args.stat, args.subset)
    scales = sorted({r["scale"] for r in rows})
    modes = [m for m in MODE_ORDER if any(r["mode"] == m for r in rows)]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(1.9 * len(scales) + 2, 4.8))
    width = args.group_width / max(len(modes), 1)
    xs = np.arange(len(scales))
    for j, mode in enumerate(modes):
        vals, los, his, runs = [], [], [], []
        for scale in scales:
            row = next((r for r in rows if r["scale"] == scale and r["mode"] == mode), None)
            v = row["value"] if row else None
            lo, hi = (row["ci_lo"], row["ci_hi"]) if row else (None, None)
            vals.append(v if v is not None else 0.0)
            los.append((v - lo) if (v is not None and lo is not None) else 0.0)
            his.append((hi - v) if (v is not None and hi is not None) else 0.0)
            runs.append(row["n_runs"] if row else 0)
        off = (j - (len(modes) - 1) / 2) * width
        ax.bar(xs + off, vals, width,
               yerr=np.array([los, his]) if (any(los) or any(his)) else None,
               capsize=2.5, color=MODE_COLOR.get(mode, "gray"),
               label=MODE_LABEL.get(mode, mode),
               error_kw={"elinewidth": 1.0, "ecolor": "black"})
        # Run coverage is uneven (30t hierarchical = 1 run): mark n_runs at the
        # base so a single-run bar is never read as cross-run supported.
        for x, nr in zip(xs + off, runs):
            if nr:
                ax.annotate(str(nr), (x, 0), ha="center", va="bottom", fontsize=6,
                            color="black", rotation=90, alpha=0.65)

    ax.set_xticks(xs); ax.set_xticklabels([f"{s}t" for s in scales])
    ax.set_xlabel("Scenario scale (trains)")
    ax.set_ylabel(f"{STAT_LABEL[args.stat]} of arrival delay (sim-s)")
    if args.log_y:
        ax.set_yscale("log")
    ax.set_title(f"Tail delay by scale — {SUBSET_LABEL[args.subset]}\n"
                 f"bar = {STAT_LABEL[args.stat]}, whisker = 95% CI (episode bootstrap); "
                 f"base digit = n training runs")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    tag = f"{args.stat}_{args.subset}"
    png = args.out_dir / f"fig_tail_by_scale_{tag}.png"
    fig.savefig(png, dpi=160)
    print(f"FIG: {png}  (from {args.csv}, stat={args.stat}, subset={args.subset})")


if __name__ == "__main__":
    main()
