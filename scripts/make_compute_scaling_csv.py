#!/usr/bin/env python3
"""Derive the wide compute-time-vs-scale table from the measured benchmark CSV.

Input  : output/9.20/l3_decision_cost_scaling.csv   (long: one row per scale x method)
Output : output/9.20/compute_time_scaling.csv       (wide: one row per scale)

Reported per scale, for each of HMAPPO / SA / FlatMAPPO:
  *_ms_per_section  — L3 decision compute per corridor section traversed.  This is
                      the per-DECISION-UNIT basis and the one that fans out
                      (HMAPPO flat, SA rising).  L2 is excluded on purpose: L2 cost
                      is per decision STEP, not per section, so including it makes
                      the unit meaningless.
  *_ms_per_episode  — whole-episode decision-layer total (L2+L3; for FlatMAPPO the
                      joint forwards).  This is the per-episode basis, where all
                      methods are ~O(N) and the curves come out near-parallel.
  sa_over_hmmapo    — ratio of the per-section costs, i.e. the constant-factor
                      advantage.  It grows only ~22% over 10-30t: do not read it
                      as an asymptotic win.

Usage:
    python scripts/make_compute_scaling_csv.py
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DISPLAY = {"hier": "HMAPPO", "sa": "SA", "flat": "FlatMAPPO"}
ORDER = ["hier", "sa", "flat"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=PROJECT_ROOT / "output/9.20/l3_decision_cost_scaling.csv")
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "output/9.20/compute_time_scaling.csv")
    args = ap.parse_args()

    rows = defaultdict(dict)
    with open(args.csv, newline="") as f:
        for r in csv.DictReader(f):
            scale = int(r["scale"])
            m = r["method"]
            rows[scale][m] = {
                "ms_per_section": float(r["l3_ms_total"]) / max(int(r["sections"]), 1),
                "ms_per_episode": float(r["decision_ms_total"]) / max(int(r["episodes"]), 1),
            }

    out_rows = []
    for scale in sorted(rows):
        d = rows[scale]
        row = {"n_trains": scale}
        for m in ORDER:
            if m not in d:
                continue
            row[f"{DISPLAY[m]}_ms_per_section"] = f"{d[m]['ms_per_section']:.4f}"
            row[f"{DISPLAY[m]}_ms_per_episode"] = f"{d[m]['ms_per_episode']:.2f}"
        if "hier" in d and "sa" in d:
            row["SA_over_HMAPPO"] = (
                f"{d['sa']['ms_per_section'] / d['hier']['ms_per_section']:.2f}")
        out_rows.append(row)

    if not out_rows:
        raise SystemExit("no rows -- is the benchmark CSV present?")

    # stable column order
    cols = ["n_trains"]
    for m in ORDER:
        cols += [f"{DISPLAY[m]}_ms_per_section", f"{DISPLAY[m]}_ms_per_episode"]
    cols.append("SA_over_HMAPPO")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)

    print(f"CSV: {args.out}")
    print(f"{'N':>4} " + " ".join(f"{c:>22}" for c in cols[1:]))
    for row in out_rows:
        print(f"{row['n_trains']:>4} "
              + " ".join(f"{str(row.get(c, '')):>22}" for c in cols[1:]))


if __name__ == "__main__":
    main()
