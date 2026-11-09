#!/usr/bin/env python3
"""Training-time convergence of the AFFECTED subset + reward, from
`training_metrics.csv` (one CSV per run, no re-training).

WHY a separate analysis instead of training-time slopes: the incident fires on
only ~6/10 episodes, and each incident touches a handful of transits.  The raw
per-episode affected-delay series is therefore "mostly nan + occasional spikes",
and a per-episode difference is meaningless.  This script
  1. stratifies by `has_incident`,
  2. smooths with a rolling mean (default window 100),
  3. reports the slope over the CONVERGED tail (last --tail episodes) as the
     "change rate" the training curve is supposed to show.

Reward is available for EVERY historical run (it has always been logged), so this
also works on runs predating the affected-subset columns — those just report the
reward half and say so.

Usage:
    python scripts/analyze_train_affected.py --run output/9_21/training/five_station_15t_hierarchical/<ts>
    python scripts/analyze_train_affected.py --run <hier_ts> --run <flat_ts> \
        --out-dir output/9.21 --tag 15t_seed0
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AFF_COLS = ("has_incident", "n_aff_transits", "aff_arr_delay_mean", "aff_arr_delay_max")


def _f(row, key):
    """Float or None (handles '', 'nan', missing column)."""
    v = row.get(key)
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return None if math.isnan(x) else x


def rolling(vals, win):
    """Rolling mean over a window of *present* samples (None skipped, not zeroed);
    a window with no present sample yields None."""
    out = []
    buf = []
    for v in vals:
        if v is not None:
            buf.append(v)
        if len(buf) > win:
            buf.pop(0)
        out.append(sum(buf) / len(buf) if buf else None)
    return out


def slope(xs, ys):
    """Least-squares slope of y over x (per episode), None if <2 points."""
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if len(pts) < 2:
        return None
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    den = sum((x - mx) ** 2 for x, _ in pts)
    if den == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pts) / den


def summarize(name, rows, tail):
    """Per-run: reward + affected-delay convergence stats."""
    eps = [int(float(r["episode"])) for r in rows]
    out = {"run": name, "n_ep": len(rows)}
    phases = sorted({r.get("phase", "?") for r in rows})
    out["phases"] = "|".join(phases)

    # ---- reward (always present) ----
    has_total = _f(rows[0], "total_reward") is not None
    rkey = "total_reward" if has_total else "mean_reward"
    rew = [_f(r, rkey) for r in rows]
    out["reward_col"] = rkey
    tail_rew = rew[-tail:]
    out["reward_tail_mean"] = statistics.mean(
        [v for v in tail_rew if v is not None]) if any(v is not None for v in tail_rew) else None
    out["reward_tail_slope_per_ep"] = slope(eps[-tail:], tail_rew)
    out["reward_first200"] = statistics.mean([v for v in rew[:200] if v is not None]) \
        if any(v is not None for v in rew[:200]) else None
    out["reward_last200"] = statistics.mean([v for v in tail_rew if v is not None]) \
        if any(v is not None for v in tail_rew) else None

    # ---- affected subset (only in runs trained after 2026-09-21) ----
    if not all(c in rows[0] for c in AFF_COLS):
        out["affected"] = "MISSING (run predates the affected-subset columns)"
        return out
    out["n_inc_eps"] = sum(1 for r in rows if _f(r, "has_incident") == 1)
    out["inc_rate_pct"] = 100.0 * out["n_inc_eps"] / max(len(rows), 1)
    out["n_aff_transits_total"] = sum(int(_f(r, "n_aff_transits") or 0) for r in rows)
    aff = [_f(r, "aff_arr_delay_mean") for r in rows]
    out["aff_present_eps"] = sum(1 for v in aff if v is not None)
    # Slope over the converged tail, on the affected episodes that actually exist
    # (nan episodes are skipped, NOT interpolated — inventing values where no
    # incident happened would fabricate a trend).
    t_eps = eps[-tail:]
    t_aff = aff[-tail:]
    out["aff_tail_mean"] = (statistics.mean([v for v in t_aff if v is not None])
                            if any(v is not None for v in t_aff) else None)
    out["aff_tail_slope_per_ep"] = slope(t_eps, t_aff)
    out["aff_first200_mean"] = (statistics.mean([v for v in aff[:200] if v is not None])
                                if any(v is not None for v in aff[:200]) else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, action="append", required=True,
                    help="run dir (containing training_metrics.csv) or the CSV itself; repeatable")
    ap.add_argument("--tail", type=int, default=300,
                    help="episodes treated as the converged window for the slope")
    ap.add_argument("--win", type=int, default=100, help="rolling-mean window for the CSV")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write <run>.affected_curve.csv (rolling series) + summary here")
    ap.add_argument("--tag", type=str, default="", help="suffix for output filenames")
    args = ap.parse_args()

    summaries = []
    for run in args.run:
        run = run if run.is_absolute() else (PROJECT_ROOT / run)
        csv_path = run if run.suffix == ".csv" else run / "training_metrics.csv"
        if not csv_path.exists():
            print(f"! missing {csv_path} — skipped")
            continue
        rows = list(csv.DictReader(open(csv_path)))
        name = run.parent.name if run.suffix == ".csv" else run.name
        st = summarize(name, rows, args.tail)
        summaries.append(st)
        print(f"\n== {name} == ({csv_path})")
        print(f"  episodes={st['n_ep']} phases={st['phases']}")
        print(f"  reward[{st['reward_col']}]: first200={st['reward_first200']} "
              f"tail-t{args.tail}-mean={st['reward_tail_mean']} "
              f"slope={st['reward_tail_slope_per_ep']}")
        if isinstance(st.get("affected"), str):
            print(f"  affected subset: {st['affected']}")
        else:
            print(f"  incident episodes={st['n_inc_eps']}/{st['n_ep']} "
                  f"({st['inc_rate_pct']:.1f}%), affected transits={st['n_aff_transits_total']}")
            print(f"  aff delay: first200-mean={st['aff_first200_mean']} "
                  f"tail-t{args.tail}-mean={st['aff_tail_mean']} "
                  f"slope={st['aff_tail_slope_per_ep']} "
                  f"(present in {st['aff_present_eps']}/{st['n_ep']} eps)")

        if args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            eps = [int(float(r["episode"])) for r in rows]
            aff = [_f(r, "aff_arr_delay_mean") for r in rows] \
                if all(c in rows[0] for c in AFF_COLS) else None
            rew = [_f(r, st["reward_col"]) for r in rows]
            suffix = f"_{args.tag}" if args.tag else ""
            with open(args.out_dir / f"{name}{suffix}.affected_curve.csv", "w", newline="") as f:
                fields = ["episode", "phase", "has_incident", "n_aff_transits",
                          "aff_arr_delay_mean", "aff_arr_delay_max",
                          "aff_rolling", "reward", "reward_rolling"]
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                aff_roll = rolling(aff, args.win) if aff is not None else [None] * len(rows)
                rew_roll = rolling(rew, args.win)
                for i, r in enumerate(rows):
                    w.writerow({
                        "episode": r["episode"], "phase": r.get("phase", ""),
                        "has_incident": r.get("has_incident", ""),
                        "n_aff_transits": r.get("n_aff_transits", ""),
                        "aff_arr_delay_mean": r.get("aff_arr_delay_mean", ""),
                        "aff_arr_delay_max": r.get("aff_arr_delay_max", ""),
                        "aff_rolling": aff_roll[i] if aff is not None else "",
                        "reward": rew[i] if rew[i] is not None else "",
                        "reward_rolling": rew_roll[i] if rew_roll[i] is not None else "",
                    })

    if args.out_dir and summaries:
        suffix = f"_{args.tag}" if args.tag else ""
        p = args.out_dir / f"train_affected_summary{suffix}.csv"
        # UNION of keys, in first-seen order: a run from before the
        # affected-subset columns has strictly fewer fields, so keying off
        # summaries[0] alone throws on the newer run (and vice versa).
        fields = list(dict.fromkeys(k for s in summaries for k in s))
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for s in summaries:
                w.writerow(s)
        print(f"\nCSVs: {args.out_dir}/<run>{suffix}.affected_curve.csv, {p}")


if __name__ == "__main__":
    main()
