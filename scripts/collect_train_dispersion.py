#!/usr/bin/env python3
"""Two CSVs of already-existing data, for the error-bar discussion.

1. train_dispersion.csv — cross-RUN dispersion of the TRAINING-time metrics
   (ontime_rate, mean_dep_delay), taken from each run's `training_metrics.csv`
   (mean over the last N episodes of a completed run).  This is NOT the held-out
   metric used by the paper's tables, and the runs span several code/config eras
   — label it as an *indication of seed variance*, not as a paper number.

   Run provenance (why the coverage is uneven):
     10t: no run
     15t: 5 complete runs (one is 9000 ep, rest 3000)
     20t: 5 complete runs (3000 ep)
     25t: 2 complete runs (8_16)
     30t: 1 complete run (9_4 seed0); 220531/220533 have only checkpoints

2. affected_vs_overall.csv — per scale, the incident-affect stratification:
   the mean-delay gap between `learned` and `sa` computed on (a) AFFECTED
   transits only and (b) ALL transits.  Shows the dilution: affected transits
   are ~3-8% of the total, so the all-transit gap is 5-30x smaller.

Source: output/9_4/scalability/<N>t/{affected,delay}.csv (existing aggregates).

Usage:
    python scripts/collect_train_dispersion.py
"""
from __future__ import annotations

import csv
import glob
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "output/9.20"
TAIL = 300  # episodes used as "converged" window


def find_train_runs() -> dict:
    """{scale: [(run_id, n_episodes, csv_path)]} for COMPLETE runs only."""
    pats = {
        15: "output/five_station_15t_hierarchical/*/training_metrics.csv",
        20: "output/five_station_20t_hierarchical/*/training_metrics.csv",
        25: "output/8_16/training/five_station_25t_hierarchical/*/training_metrics.csv",
        30: "output/9_4/training/five_station_30t_hierarchical/*/training_metrics.csv",
    }
    out = {}
    for scale, pat in pats.items():
        runs = []
        for f in sorted(glob.glob(str(PROJECT_ROOT / pat))):
            rows = list(csv.DictReader(open(f)))
            if len(rows) < 1000:          # incomplete / aborted run
                continue
            runs.append((Path(f).parent.name, len(rows), Path(f)))
        if runs:
            out[scale] = runs
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 1. cross-run dispersion of training metrics ----
    runs = find_train_runs()
    per_run, agg = [], []
    for scale in sorted(runs):
        ot, dp, mr = [], [], []
        for run_id, n_ep, path in runs[scale]:
            rows = list(csv.DictReader(open(path)))[-TAIL:]
            o = statistics.mean(float(r["ontime_rate"]) for r in rows) * 100
            d = statistics.mean(float(r["mean_dep_delay"]) for r in rows)
            m = statistics.mean(float(r["mean_reward"]) for r in rows)
            per_run.append({"scale": scale, "run": run_id, "n_episodes": n_ep,
                            "ontime_rate_pct": round(o, 2),
                            "mean_dep_delay_s": round(d, 2),
                            "mean_reward": round(m, 3)})
            ot.append(o); dp.append(d); mr.append(m)
        sd = lambda v: statistics.stdev(v) if len(v) > 1 else float("nan")
        agg.append({
            "scale": scale, "n_runs": len(runs[scale]),
            "ontime_mean": round(statistics.mean(ot), 2), "ontime_std": round(sd(ot), 2),
            "ontime_min": round(min(ot), 2), "ontime_max": round(max(ot), 2),
            "dep_delay_mean": round(statistics.mean(dp), 2), "dep_delay_std": round(sd(dp), 2),
            "dep_delay_min": round(min(dp), 2), "dep_delay_max": round(max(dp), 2),
        })
    p = OUT / "train_dispersion.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys())); w.writeheader(); w.writerows(agg)
    p2 = OUT / "train_per_run.csv"
    with open(p2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_run[0].keys())); w.writeheader(); w.writerows(per_run)
    print(f"CSV: {p}\nCSV: {p2}\n")
    print("  训练期跨-run 波动（末 %d 集均值）:" % TAIL)
    for a in agg:
        print(f"    {a['scale']:>2}t (n={a['n_runs']}): ontime {a['ontime_mean']}±{a['ontime_std']} "
              f"[{a['ontime_min']}~{a['ontime_max']}] | dep_delay {a['dep_delay_mean']}±{a['dep_delay_std']} s")

    # ---- 2. affected-only vs all-transit gap (dilution) ----
    rows2 = []
    for s in (10, 15, 20, 25, 30):
        ap, dp = PROJECT_ROOT / f"output/9_4/scalability/{s}t/affected.csv", \
                 PROJECT_ROOT / f"output/9_4/scalability/{s}t/delay.csv"
        if not (ap.exists() and dp.exists()):
            continue
        A = {r["mode"]: r for r in csv.DictReader(open(ap))}
        D = {r["mode"]: r for r in csv.DictReader(open(dp))}
        if "learned" not in A or "sa" not in A:
            continue
        tot = float(A["learned"]["aff_n"]) + float(A["learned"]["unaff_n"])
        rows2.append({
            "scale": s,
            "aff_share_pct": round(100 * float(A["learned"]["aff_n"]) / tot, 2),
            "aff_mean_learned": float(A["learned"]["aff_mean"]),
            "aff_mean_sa": float(A["sa"]["aff_mean"]),
            "aff_diff": round(float(A["learned"]["aff_mean"]) - float(A["sa"]["aff_mean"]), 2),
            "all_mean_learned": float(D["learned"]["mean"]),
            "all_mean_sa": float(D["sa"]["mean"]),
            "all_diff": round(float(D["learned"]["mean"]) - float(D["sa"]["mean"]), 2),
        })
    p3 = OUT / "affected_vs_overall.csv"
    if rows2:
        with open(p3, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows2[0].keys())); w.writeheader(); w.writerows(rows2)
        print(f"\nCSV: {p3}\n")
        print("  affected-only vs overall  (learned - SA):")
        for r in rows2:
            print(f"    {r['scale']:>2}t: aff_share={r['aff_share_pct']:>4.1f}% | "
                  f"affected diff {r['aff_diff']:>+7.2f} s | all diff {r['all_diff']:>+6.2f} s")


if __name__ == "__main__":
    main()
