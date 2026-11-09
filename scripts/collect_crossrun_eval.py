#!/usr/bin/env python3
"""Cross-RUN held-out evaluation -> error bars for delay and ontime.

WHY: every eval artifact in this repo is an aggregate (mean/p50/p90 only), so no
error bar can be recovered from them.  BUT the repo already contains 2-5 trained
checkpoints per scale from earlier campaigns (Aug 13-14 for 15t/20t, 8_16 for
25t, 9_4 for 30t).  Evaluating each of them gives **cross-run dispersion at zero
training cost**.

Only the `learned` arm varies across runs.  `sa` / `sa-online` / `sa-nofc` /
`cruise` are checkpoint-independent (they differ only in the L3 execution layer),
so they need one eval per scale, not one per run.

Protocol matches the existing scalability eval: --incident --blocks 3 --n 60
--seeds 900000.  Config used = the WORKING TREE (P3 reclassified out of existence).

Outputs (into --out-dir):
    eval_per_block.csv   long: scale, run, mode, block, n, mean, aff_n, aff_mean,
                         unaff_mean, ontime
    eval_crossrun.csv    per (scale, mode): mean +/- std ACROSS RUNS of (a) ontime
                         and (b) aff_mean, averaged over blocks

Usage:
    python scripts/collect_crossrun_eval.py                 # all runs, learned arm
    python scripts/collect_crossrun_eval.py --episodes 60 --blocks 3
"""
from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# (scale, run_id, checkpoint) — every hierarchical ckpt that exists and loads.
# NOTE 15t/20260808_175357 is a 9000-episode run (others are 3000); kept and
# flagged via `budget_ep` rather than silently dropped.
RUNS = [
    # USE `final.pt`, NOT `best.pt`.  Verified 2026-09-21 on 15t: the SAME run's
    # two checkpoints differ by 27 pp of held-out arrival ontime (best.pt 40.6%
    # vs final.pt 67.4%, 1x20 seeds @900000) — `best.pt` is selected on the
    # TRAINING mean_reward, which is a different objective from the paper's
    # metric.  The existing scalability eval numbers match `final.pt`.
    # 10t has NO checkpoint of its own -> use the 15t policies under the §6.3
    # transfer protocol ("one training, retrain-free transfer to 10-30t"); the
    # same 4 uniform-budget policies therefore also give 10t its dispersion.
    (10, "15t:20260814_194055", "output/five_station_15t_hierarchical/20260814_194055/checkpoints/final.pt", 3000),
    (10, "15t:20260814_200413", "output/five_station_15t_hierarchical/20260814_200413/checkpoints/final.pt", 3000),
    (10, "15t:20260814_203600", "output/five_station_15t_hierarchical/20260814_203600/checkpoints/final.pt", 3000),
    (10, "15t:20260814_210606", "output/five_station_15t_hierarchical/20260814_210606/checkpoints/final.pt", 3000),
    (15, "20260814_194055", "output/five_station_15t_hierarchical/20260814_194055/checkpoints/final.pt", 3000),
    (15, "20260814_200413", "output/five_station_15t_hierarchical/20260814_200413/checkpoints/final.pt", 3000),
    (15, "20260814_203600", "output/five_station_15t_hierarchical/20260814_203600/checkpoints/final.pt", 3000),
    (15, "20260814_210606", "output/five_station_15t_hierarchical/20260814_210606/checkpoints/final.pt", 3000),
    (20, "20260813_231902", "output/five_station_20t_hierarchical/20260813_231902/checkpoints/final.pt", 3000),
    (20, "20260813_233319", "output/five_station_20t_hierarchical/20260813_233319/checkpoints/final.pt", 3000),
    (20, "20260813_234330", "output/five_station_20t_hierarchical/20260813_234330/checkpoints/final.pt", 3000),
    (20, "20260813_235156", "output/five_station_20t_hierarchical/20260813_235156/checkpoints/final.pt", 3000),
    (20, "20260814_000258", "output/five_station_20t_hierarchical/20260814_000258/checkpoints/final.pt", 3000),
    (25, "20260816_140832", "output/8_16/training/five_station_25t_hierarchical/20260816_140832/checkpoints/final.pt", 6000),
    (25, "20260816_154929", "output/8_16/training/five_station_25t_hierarchical/20260816_154929/checkpoints/final.pt", 6000),
    # 30t: ONLY 220529 completed (has final.pt).  220531/220533 were killed
    # mid-run and have best.pt only (220531 even stopped at ep 2999 = end of
    # Phase A, whose l3_actor is never trained) -> excluded.  So 30t: n=1.
    (30, "20260904_220529", "output/9_4/training/five_station_30t_hierarchical/20260904_220529/checkpoints/final.pt", 6000),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=900000)
    ap.add_argument("--scales", type=int, nargs="+", default=[10, 15, 20, 25, 30])
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "output/9.20")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scale, run_id, ckpt, budget in RUNS:
        if scale not in args.scales:
            continue
        ck = PROJECT_ROOT / ckpt
        if not ck.exists():
            print(f"  ! missing {ck} — skipped")
            continue
        with tempfile.TemporaryDirectory() as td:
            cmd = [sys.executable, "scripts/eval_protocol.py",
                   "--scenario", f"five_station_{scale}t",
                   "--l2", str(ck), "--l3", str(ck),
                   "--incident", "--blocks", str(args.blocks),
                   "--n", str(args.episodes), "--seeds", str(args.seeds),
                   "--learned-only", "--out-dir", td]
            r = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
            pb = Path(td) / "per_block.csv"
            if r.returncode != 0 or not pb.exists():
                print(f"  ! {scale}t/{run_id} FAILED: {(r.stderr or '').strip()[-200:]}")
                continue
            for row in csv.DictReader(open(pb)):
                if row["mode"] != "learned":
                    continue
                rows.append({"scale": scale, "run": run_id, "budget_ep": budget,
                             **row})
            o = [float(x["ontime"]) for x in rows if x["scale"] == scale and x["run"] == run_id]
            print(f"  {scale}t/{run_id}: ontime/blocks = " +
                  " ".join(f"{v:.1f}" for v in o))

    if not rows:
        raise SystemExit("no rows collected")

    pb_path = args.out_dir / "eval_per_block.csv"
    with open(pb_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nCSV: {pb_path}")

    # ---- cross-run error bars: per scale, average over blocks within a run,
    # then mean +/- std ACROSS RUNS ----
    agg = []
    for scale in sorted({x["scale"] for x in rows}):
        per_run = {}
        for x in rows:
            if x["scale"] != scale:
                continue
            per_run.setdefault(x["run"], []).append(x)
        ot, af, dl = [], [], []
        for run, blk in per_run.items():
            ot.append(statistics.mean(float(b["ontime"]) for b in blk))
            af.append(statistics.mean(float(b["aff_mean"]) for b in blk))
            dl.append(statistics.mean(float(b["mean"]) for b in blk))
        sd = lambda v: statistics.stdev(v) if len(v) > 1 else float("nan")
        agg.append({
            "scale": scale, "n_runs": len(per_run),
            "ontime_mean": round(statistics.mean(ot), 2), "ontime_std": round(sd(ot), 2),
            "ontime_min": round(min(ot), 2), "ontime_max": round(max(ot), 2),
            "overall_delay_mean": round(statistics.mean(dl), 2), "overall_delay_std": round(sd(dl), 2),
            "aff_delay_mean": round(statistics.mean(af), 2), "aff_delay_std": round(sd(af), 2),
            "aff_delay_min": round(min(af), 2), "aff_delay_max": round(max(af), 2),
        })
    cr_path = args.out_dir / "eval_crossrun.csv"
    with open(cr_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader(); w.writerows(agg)
    print(f"CSV: {cr_path}\n")
    for a in agg:
        print(f"  {a['scale']}t (n={a['n_runs']}): ontime {a['ontime_mean']}±{a['ontime_std']} "
              f"[{a['ontime_min']}~{a['ontime_max']}] | aff_delay {a['aff_delay_mean']}±{a['aff_delay_std']}")


if __name__ == "__main__":
    main()
