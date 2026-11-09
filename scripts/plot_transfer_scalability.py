#!/usr/bin/env python3
"""Create the main zero-shot cross-scale transfer figure.

Both policies are trained at 15 trains and evaluated unchanged at each target
fleet size.  The script deliberately extracts only the hierarchical and flat
rows from the existing held-out evaluation summary, writes a self-describing
CSV, then draws a caption-free figure.  The manuscript caption can therefore
be authored independently while the numeric source remains reproducible.

Usage:
    python scripts/plot_transfer_scalability.py
    python scripts/plot_transfer_scalability.py --figure-out output/figures/transfer.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCALES = (10, 15, 20, 25, 30)
METHODS = ("learned", "flat")
LABELS = {
    "learned": "Hierarchical MAPPO",
    "flat": "Flat MAPPO",
}
COLORS = {
    "learned": "#0072B2",  # Okabe-Ito blue
    "flat": "#D55E00",     # Okabe-Ito vermillion
}


def read_transfer_rows(path: Path, source_scale: int, eval_episodes: int):
    """Read the two policy rows and attach explicit protocol provenance."""
    with path.open(newline="", encoding="utf-8") as f:
        raw = list(csv.DictReader(f))
    if not raw:
        raise ValueError(f"No rows in {path}")

    by_scale = {}
    for row in raw:
        try:
            scale = int(row["scale"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid scale row: {row}") from exc
        if scale not in SCALES:
            continue
        by_scale.setdefault(scale, {})
        for method in METHODS:
            if method not in row:
                raise ValueError(f"{path} is missing required column '{method}'")
            try:
                value = float(row[method])
            except ValueError as exc:
                raise ValueError(f"Invalid {method} value at {scale}t: {row[method]}") from exc
            by_scale[scale][method] = value

    missing = [f"{scale}t/{method}" for scale in SCALES for method in METHODS
               if method not in by_scale.get(scale, {})]
    if missing:
        raise ValueError(f"Missing transfer results: {', '.join(missing)}")

    rows = []
    for scale in SCALES:
        for method in METHODS:
            rows.append({
                "target_scale_trains": scale,
                "method": method,
                "method_label": LABELS[method],
                "arrival_ontime_pct": f"{by_scale[scale][method]:.1f}",
                "source_train_scale_trains": source_scale,
                "target_finetuning": "none (zero-shot)",
                "heldout_evaluation_episodes": eval_episodes,
                "source_summary_csv": str(path.relative_to(PROJECT_ROOT)),
            })
    return rows


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(path: Path, rows, source_scale: int):
    values = {
        method: [next(float(r["arrival_ontime_pct"]) for r in rows
                      if r["method"] == method and r["target_scale_trains"] == scale)
                 for scale in SCALES]
        for method in METHODS
    }

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.linewidth": 0.9,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    })
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for method in METHODS:
        ax.plot(
            SCALES,
            values[method],
            color=COLORS[method],
            marker="o",
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.9,
            linewidth=2.3,
            label=LABELS[method],
            zorder=3,
        )

    ax.axvline(source_scale, color="#666666", linestyle="--", linewidth=1.0, zorder=1)
    ax.text(source_scale + 0.25, 93.5, "source policy (15t)", color="#555555",
            fontsize=10, va="top")
    ax.set_xlim(9, 31)
    # A full percentage scale avoids visually exaggerating the method gap.
    ax.set_ylim(0, 100)
    ax.set_xticks(SCALES, [f"{scale}t" for scale in SCALES])
    ax.set_xlabel("Evaluation fleet size")
    ax.set_ylabel("Arrival on-time rate (%)")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, linestyle=":", zorder=0)
    ax.legend(loc="lower left", frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "output/9_4/scalability/summary_ontime.csv",
        help="held-out on-time summary containing learned and flat columns",
    )
    parser.add_argument(
        "--data-out",
        type=Path,
        default=PROJECT_ROOT / "output/9_4/scalability/transfer_15t_zero_shot_ontime.csv",
        help="self-describing transfer data extract",
    )
    parser.add_argument(
        "--figure-out",
        type=Path,
        default=PROJECT_ROOT / "output/9_4/scalability/fig_transfer_15t_zero_shot.png",
        help="caption-free PNG destination",
    )
    parser.add_argument("--source-scale", type=int, default=15)
    parser.add_argument("--eval-episodes", type=int, default=180)
    args = parser.parse_args()

    rows = read_transfer_rows(args.input, args.source_scale, args.eval_episodes)
    write_rows(args.data_out, rows)
    plot(args.figure_out, rows, args.source_scale)
    print(f"data: {args.data_out}")
    print(f"figure: {args.figure_out}")


if __name__ == "__main__":
    main()
