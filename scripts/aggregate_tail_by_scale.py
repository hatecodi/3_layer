#!/usr/bin/env python3
"""STEP 1 of 2: raw eval rows -> ONE tidy CSV of tail-delay statistics.

This script does ALL the analysis (pooling, tail statistics, bootstrap CIs) and
writes a single long-format CSV.  `plot_tail_by_scale.py` (step 2) only READS
that CSV — no statistic is ever computed inside a plotting script, so a figure
can be re-rendered or thrown away without redoing (or perturbing) the analysis.

Input :  output/9.21/tail_rows/<scale>t/<run>/per_transit.csv   (from
         scripts/collect_tail_eval.py -> scripts/eval_protocol.py)
Output:  output/9.21/tail_by_scale.csv  (long format)

    scale, mode, subset, stat, value, ci_lo, ci_hi, ci_halfwidth,
    cross_run_std, n_episodes, n_runs

  subset : all | affected | affected_plan
           - `affected`      = transit overlapped the restriction window on the
                               REALIZED trajectory -> policy-dependent subset
           - `affected_plan` = same test on the PLANNED timetable -> policy-
                               independent, so cross-policy comparisons are not
                               comparing different trains
  stat   : tail10 | tail10abs | p90 | p95 | max | mean
           - tail10 = mean of the worst 10% of the POOLED distribution (the
             requested quantity)
           - tail10abs = same on |delay|, the guard against a policy flattening
             the signed tail by running early
  ci_*   : 95% percentile interval from an EPISODE-level bootstrap (1000 draws):
           resample episodes with replacement, RE-POOL their transits, recompute
           the statistic.  Episode, not transit, is the iid unit — transits
           within an episode are correlated.  Not a block std (df=2) and not a
           run std (only exists for learned/flat, and 30t has one hier run).
  cross_run_std : std across training runs of the same statistic.  REAL SIGNAL
           about training randomness, reported alongside, never used as the CI.

Usage:
    python scripts/aggregate_tail_by_scale.py                  # all rows found
    python scripts/aggregate_tail_by_scale.py --expect-episodes 180 --strict
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TAIL_FRAC = 0.10
EPS = 1e-9


def _f(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


# ---- statistics on the pooled list -----------------------------------------

def pooled_tail(vals, frac=TAIL_FRAC):
    v = sorted(vals)
    if not v:
        return float("nan")
    k = max(1, int(round(frac * len(v))))
    return sum(v[-k:]) / k


def pooled_tail_abs(vals, frac=TAIL_FRAC):
    return pooled_tail([abs(x) for x in vals], frac)


def quantile(vals, q):
    v = sorted(vals)
    return v[min(len(v) - 1, int(len(v) * q))] if v else float("nan")


STATS = {
    "tail10": pooled_tail,
    "tail10abs": pooled_tail_abs,
    "p90": lambda v: quantile(v, 0.9),
    "p95": lambda v: quantile(v, 0.95),
    "max": lambda v: max(v) if v else float("nan"),
    "mean": lambda v: (sum(v) / len(v)) if v else float("nan"),
}
# `unaffected` matters more than it looks: the incident's ABSOLUTE damage barely
# changes with fleet size (~1.8-2.2 affected transits per episode at every scale,
# with near-identical magnitudes: mean 64.7s @10t vs 65.4s @15t), while total
# transits per episode grow 23 -> 64.  So the affected SHARE falls 7.7% -> 3.0%
# and the all-transit tail gets diluted by scale through pure mixture geometry,
# not scheduling quality: at 10t the pooled p90 (87.9) is WORSE than at 15t (80.1)
# even though 10t's UNAFFECTED p90 is BETTER (60.9 vs 68.8).  `unaffected` is the
# dilution-free view and should be reported next to `all`.
SUBSETS = ["all", "affected", "unaffected", "affected_plan"]


def group_by_episode(rows, subset):
    """-> [[values]] keyed by episode so the bootstrap resamples WHOLE episodes."""
    eps = defaultdict(list)
    for r in rows:
        if subset == "unaffected":
            if r.get("affected") != "0":
                continue
        elif subset != "all" and r.get(subset) != "1":
            continue
        v = _f(r.get("arr_delay"))
        if v is not None:
            eps[(r.get("block"), r.get("seed"))].append(v)
    return list(eps.values())


def bootstrap_pooled(ep_lists, stat, n_draws, alpha, rng):
    """95% percentile CI of `stat` over the pooled distribution, resampling
    episodes with replacement.  The SAME statistic is re-estimated on each draw,
    so the interval belongs to the reported quantity."""
    if not ep_lists:
        return float("nan"), float("nan"), float("nan"), 0
    point = stat([v for ep in ep_lists for v in ep])
    n = len(ep_lists)
    if n < 2:
        return point, point, point, n
    draws = []
    for _ in range(n_draws):
        buf = []
        for _ in range(n):
            buf.extend(ep_lists[rng.randrange(n)])
        draws.append(stat(buf))
    draws.sort()
    return (point, draws[int(n_draws * alpha / 2)],
            draws[int(n_draws * (1 - alpha / 2)) - 1], n)


def infer_scale(path):
    for part in Path(path).parts:
        if part.endswith("t") and part[:-1].isdigit():
            return int(part[:-1])
        if "five_station_" in part:
            d = "".join(c for c in part.split("_")[2] if c.isdigit())
            if d:
                return int(d)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-glob", type=str,
                    default=str(PROJECT_ROOT / "output/9.21/tail_rows/*/*/per_transit.csv"))
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "output/9.21/tail_by_scale.csv")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expect-episodes", type=int, default=None,
                    help="warn when a run has a different episode count (a leftover "
                         "smoke run mixed into the full sweep would silently bias the pool)")
    ap.add_argument("--strict", action="store_true",
                    help="drop runs whose episode count != --expect-episodes")
    args = ap.parse_args()

    import glob as _glob
    paths = sorted(_glob.glob(args.rows_glob, recursive=True))
    if not paths:
        raise SystemExit(f"no per_transit.csv matched {args.rows_glob}")

    data = defaultdict(list)          # (scale, mode, run) -> rows
    for p in paths:
        scale = infer_scale(p)
        if scale is None:
            raise SystemExit(f"cannot infer scale from {p}")
        run = Path(p).parent.name
        for r in csv.DictReader(open(p)):
            data[(scale, r["mode"], run)].append(r)

    # provenance guard: episode count per run
    kept, dropped = {}, []
    for k, rows in data.items():
        n_ep = len({(r.get("block"), r.get("seed")) for r in rows})
        if args.expect_episodes and n_ep != args.expect_episodes:
            if args.strict:
                dropped.append((k, n_ep)); continue
            print(f"  ! {k[0]}t {k[1]}/{k[2]}: {n_ep} episodes "
                  f"(expected {args.expect_episodes})")
        kept[k] = rows
    for k, n_ep in dropped:
        print(f"  - dropped {k[0]}t {k[1]}/{k[2]} ({n_ep} episodes)")
    data = kept

    scales = sorted({k[0] for k in data})
    modes = sorted({k[1] for k in data})
    rng = random.Random(args.seed)

    out_rows = []
    for scale in scales:
        for mode in modes:
            runs = [k[2] for k in data if k[0] == scale and k[1] == mode]
            if not runs:
                continue
            for subset in SUBSETS:
                ep_lists = []
                for run in runs:
                    ep_lists.extend(group_by_episode(data[(scale, mode, run)], subset))
                if not ep_lists:
                    continue
                for sname, sfun in STATS.items():
                    val, lo, hi, n_ep = bootstrap_pooled(ep_lists, sfun, args.draws, 0.05, rng)
                    per_run = []
                    for run in runs:
                        v = [x for x in (_f(r.get("arr_delay"))
                                         for r in data[(scale, mode, run)]
                                         if (r.get("affected") == "0" if subset == "unaffected"
                                             else subset == "all" or r.get(subset) == "1"))
                             if x is not None]
                        if v:
                            per_run.append(sfun(v))
                    sd = statistics.stdev(per_run) if len(per_run) > 1 else float("nan")
                    out_rows.append({
                        "scale": scale, "mode": mode, "subset": subset, "stat": sname,
                        "value": round(val, 3) if val == val else "",
                        "ci_lo": round(lo, 3) if lo == lo else "",
                        "ci_hi": round(hi, 3) if hi == hi else "",
                        "ci_halfwidth": round((hi - lo) / 2, 3) if hi == hi else "",
                        "cross_run_std": round(sd, 3) if sd == sd else "",
                        "n_episodes": n_ep, "n_runs": len(per_run),
                    })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print(f"\nCSV: {args.out}  ({len(out_rows)} rows, "
          f"{len(scales)} scales x {len(modes)} modes x {len(SUBSETS)} subsets x {len(STATS)} stats)")
    print("   plot it with:  python scripts/plot_tail_by_scale.py "
          f"--csv {args.out} --stat tail10 --subset all")


if __name__ == "__main__":
    main()
