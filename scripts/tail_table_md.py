#!/usr/bin/env python3
"""Emit a Markdown table of `value [95% CI]` from output/9.21/tail_by_scale.csv.

The table is DERIVED from the CSV, never hand-typed, so it cannot drift from the
analysis (the figure is derived the same way, by plot_tail_by_scale.py).

Usage:
    python scripts/tail_table_md.py --stat tail10 --subset all
    python scripts/tail_table_md.py --stat p90 --subset unaffected --out docs/tail_delay_and_error_bars.md --append
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODE_ORDER = ["learned", "flat", "sa", "sa-online", "sa-nofc", "fcfs", "cruise"]
MODE_LABEL = {
    "learned": "**Hierarchical (ours)**", "flat": "Flat MAPPO", "sa": "SA (oracle)",
    "sa-online": "SA (partial obs)", "sa-nofc": "SA (no forecast)",
    "fcfs": "FCFS (rule)", "cruise": "CRUISE (no coord.)",
}
STAT_LABEL = {
    "tail10": "worst-decile mean", "tail10abs": "worst-decile mean of |delay|",
    "p90": "p90", "p95": "p95", "max": "max", "mean": "mean",
}
SUBSET_LABEL = {
    "all": "all transits", "affected": "affected only (realised overlap)",
    "unaffected": "unaffected only (dilution-free)",
    "affected_plan": "affected only (planned overlap, policy-independent)",
}


def build(csv_path, stat, subset, digits=1):
    rows = [r for r in csv.DictReader(open(csv_path))
            if r["stat"] == stat and r["subset"] == subset]
    if not rows:
        raise SystemExit(f"no rows stat={stat} subset={subset}")
    scales = sorted({int(r["scale"]) for r in rows})
    modes = [m for m in MODE_ORDER if any(r["mode"] == m for r in rows)]

    def cell(mode, scale):
        r = next((x for x in rows if x["mode"] == mode and int(x["scale"]) == scale), None)
        if not r or r["value"] == "":
            return "—"
        v, lo, hi = float(r["value"]), float(r["ci_lo"]), float(r["ci_hi"])
        return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"

    def runs(mode, scale):
        r = next((x for x in rows if x["mode"] == mode and int(x["scale"]) == scale), None)
        return (r["n_runs"] if r else "")

    lines = []
    lines.append(f"*{STAT_LABEL[stat]} of arrival delay, {SUBSET_LABEL[subset]} — "
                 f"cell = value [95% CI] in sim-s.*")
    lines.append("")
    lines.append("| method | " + " | ".join(f"{s}t" for s in scales) + " |")
    lines.append("|---|" + "---|" * len(scales))
    for m in modes:
        lines.append(f"| {MODE_LABEL.get(m, m)} | " +
                     " | ".join(cell(m, s) for s in scales) + " |")
    lines.append("")
    lines.append("Number of training runs behind each `learned` cell (the CI covers "
                 "eval sampling; cross-run spread is a separate column in the CSV): "
                 + ", ".join(f"{runs('learned', s)}× {s}t" for s in scales) + ".")
    return "\n".join(x for x in lines if x is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=PROJECT_ROOT / "output/9.21/tail_by_scale.csv")
    ap.add_argument("--stat", default="tail10")
    ap.add_argument("--subset", default="all")
    ap.add_argument("--out", type=Path, default=None, help="append/write the table to this file")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--digits", type=int, default=1)
    args = ap.parse_args()

    table = build(args.csv, args.stat, args.subset, args.digits)
    print(table)
    if args.out:
        with open(args.out, "a" if args.append else "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print(f"\n{'appended to' if args.append else 'wrote'} {args.out}")


if __name__ == "__main__":
    main()
