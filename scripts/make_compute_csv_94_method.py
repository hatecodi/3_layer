#!/usr/bin/env python3
"""Recompute the compute table with the 9_4 METHOD but MEASURED unit costs.

The 9_4 tables (output/9_4/scalability/<N>t/compute.csv) were pure arithmetic:

    method_ms      = n_transits x <hardcoded ms/transit>      (SA: 3.48)
    learned_ms     = n_sections x <hardcoded ms/section>      (    0.36)
    resolve_ratio  = sa_resolve_ms / learned_ms

Because n_sections / n_transits == 4.00 exactly (every corridor has 4 sections),
that formula pins resolve_ratio to 9.667/4 = 2.417 at EVERY scale -- the "2.4x"
carried no information about scale at all.

This script keeps the method identical (counts x unit cost) and only swaps in the
MEASURED per-scale unit costs from
    output/9.20/l3_decision_cost_scaling.csv
so resolve_ratio becomes the measured 3.61 -> 4.28.  Counts are taken from the
9_4 eval (--counts-dir); override if a different eval should supply them.

Output: output/9.20/compute_by_scale.csv  (one row per scale)

Usage:
    python scripts/make_compute_csv_94_method.py
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-csv", type=Path,
                    default=PROJECT_ROOT / "output/9.20/l3_decision_cost_scaling.csv")
    ap.add_argument("--counts-dir", type=Path,
                    default=PROJECT_ROOT / "output/9_4/scalability",
                    help="dir holding <N>t/compute.csv (supplies n_transits/n_sections)")
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "output/9.20/compute_by_scale.csv")
    args = ap.parse_args()

    # measured per-scale unit costs (ms): L3-only
    unit = defaultdict(dict)
    for r in csv.DictReader(open(args.cost_csv, newline="")):
        s, m = int(r["scale"]), r["method"]
        unit[s][m] = {
            "per_section": float(r["l3_ms_total"]) / max(int(r["sections"]), 1),
            "per_transit": float(r["l3_ms_total"]) / max(int(r["transits"]), 1),
        }

    rows = []
    for scale in sorted(unit):
        cnt = args.counts_dir / f"{scale}t" / "compute.csv"
        if not cnt.exists():
            print(f"  ! no counts for {scale}t ({cnt}) -- skipped")
            continue
        first = next(csv.DictReader(open(cnt, newline="")))
        n_tr, n_sec = int(first["n_transits"]), int(first["n_sections"])

        learned = n_sec * unit[scale]["hier"]["per_section"]
        sa = n_tr * unit[scale]["sa"]["per_transit"]
        flat = n_tr * unit[scale]["flat"]["per_transit"]
        rows.append({
            "n_trains": scale, "n_transits": n_tr, "n_sections": n_sec,
            "HMAPPO_ms": f"{learned:.1f}",
            "SA_ms": f"{sa:.1f}",
            "FlatMAPPO_ms": f"{flat:.1f}",
            "SA_over_HMAPPO": f"{sa / learned:.2f}",
            # the old 9_4 numbers, for side-by-side comparison
            "old_94_learned_ms": first["learned_ms"],
            "old_94_sa_ms": first["sa_resolve_ms"],
            "old_94_ratio": first["resolve_ratio"],
        })

    if not rows:
        raise SystemExit("no rows -- check --counts-dir")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    cols = list(rows[0].keys())
    print(f"CSV: {args.out}")
    print("  " + "  ".join(f"{c:>12}" for c in cols))
    for r in rows:
        print("  " + "  ".join(f"{str(r[c]):>12}" for c in cols))


if __name__ == "__main__":
    main()
