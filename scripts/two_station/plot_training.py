#!/usr/bin/env python3
"""Quick training curve plot — reads training_metrics.json and prints summary.

Usage:
    python scripts/plot_training.py [outputs/logs/training_metrics.json]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).resolve().parent.parent / "outputs" / "logs" / "training_metrics.json"
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    data = json.loads(path.read_text())
    n = len(data)

    # Split into bins for smoothing
    bin_size = max(1, n // 20)
    bins: Dict[str, list] = defaultdict(list)

    for ep, entry in enumerate(data):
        bins[ep // bin_size].append(entry["mean_reward"])

    print(f"\n  Learning curve ({n} episodes, bin={bin_size})\n")
    print(f"  {'Bin':>5} {'EpRange':>10} {'Reward':>8} {'DepDly':>7} {'ArrDly':>7} {'Best':>6}")
    print(f"  {'-'*47}")

    for bin_idx in sorted(bins.keys()):
        rewards = bins[bin_idx]
        avg = sum(rewards) / len(rewards)
        entries_in_bin = [
            data[ep] for ep in range(bin_idx * bin_size,
                                     min((bin_idx + 1) * bin_size, n))
        ]
        dep = sum(e["mean_dep_delay"] for e in entries_in_bin) / len(entries_in_bin)
        arr = sum(e["mean_arr_delay"] for e in entries_in_bin) / len(entries_in_bin)
        best = max(rewards)
        start = bin_idx * bin_size
        end = min((bin_idx + 1) * bin_size, n) - 1
        bar = "#" * int((avg - 2.4) / (3.6 - 2.4) * 20)
        print(f"  {bin_idx:>5} {start:>4}-{end:<4} {avg:>8.2f} {dep:>7.1f} {arr:>7.1f} {best:>6.2f}  {bar}")

    # Overall stats
    all_r = [e["mean_reward"] for e in data]
    print(f"\n  Overall: reward={sum(all_r)/n:.2f}  "
          f"best={max(all_r):.2f}  worst={min(all_r):.2f}")

    # Heuristic baseline
    print(f"  Heuristic baseline: reward=3.26  dep=37.2s  arr=16.7s")

    # Count how often we beat heuristic
    beat = sum(1 for r in all_r if r > 3.26)
    print(f"  Beat heuristic: {beat}/{n} ({beat/n*100:.1f}%)")

    print()


if __name__ == "__main__":
    main()
