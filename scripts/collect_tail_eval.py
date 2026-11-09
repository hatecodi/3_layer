#!/usr/bin/env python3
"""Collect per-EPISODE eval rows for the tail-delay-by-scale figure.

WHY a new driver instead of reusing collect_crossrun_eval.py: that one keeps only
`per_block.csv` (3 values per mode) and the aggregates throw away the transits, so
no episode-level tail statistic survives.  This driver keeps `per_episode.csv`,
which is the iid unit the bootstrap CI needs.

Layout written (scale is a path component so plot_tail_by_scale.py can infer it):
    output/9.21/tail_rows/<N>t/<run_id>/per_episode.csv     (one per training run)
    output/9.21/tail_rows/<N>t/_baselines/per_episode.csv   (checkpoint-independent
                                                             modes; `learned` rows
                                                             filtered OUT to avoid
                                                             double-counting the
                                                             per-run learned arm)

Runs, provenance and the caveats that matter for reading the error bars:

  * learned (hierarchical) — 16 ckpts, `final.pt` NOT `best.pt` (the two differ by
    27 pp of held-out on-time on the same run; `best.pt` is selected on the TRAINING
    reward, a different objective).  Budgets differ: 15t/20t are 3000-ep runs,
    25t/30t are 6000-ep.  30t has n_runs=1 — its bar has NO cross-run support.
    10t has no ckpt of its own: it reuses the 15t policies (the §6.3
    "train once, transfer without retraining to 10-30t" protocol).
  * flat MAPPO — attributed from the training logs (the run dirs themselves carry
    no scenario metadata): 15t has 3 runs (1 x 4000-ep + 2 x 6000-ep), 25t has 3
    (same split), 30t has 3 (9_4).  10t/20t reuse the 15t flat ckpt by the same
    transfer protocol.
  * baselines (sa / sa-online / sa-nofc / fcfs / cruise) are checkpoint-INDEPENDENT
    — one pass per scale, then `learned` rows are dropped.

Usage:
    python scripts/collect_tail_eval.py --scales 15            # smoke one scale
    python scripts/collect_tail_eval.py                        # 10 15 20 25 30
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT_ROOT / "output/9.21/tail_rows"

# (scale, run_id, ckpt) — same provenance table as collect_crossrun_eval.py.
HIER_RUNS = [
    (10, "15t:20260814_194055", "output/five_station_15t_hierarchical/20260814_194055/checkpoints/final.pt"),
    (10, "15t:20260814_200413", "output/five_station_15t_hierarchical/20260814_200413/checkpoints/final.pt"),
    (10, "15t:20260814_203600", "output/five_station_15t_hierarchical/20260814_203600/checkpoints/final.pt"),
    (10, "15t:20260814_210606", "output/five_station_15t_hierarchical/20260814_210606/checkpoints/final.pt"),
    (15, "20260814_194055", "output/five_station_15t_hierarchical/20260814_194055/checkpoints/final.pt"),
    (15, "20260814_200413", "output/five_station_15t_hierarchical/20260814_200413/checkpoints/final.pt"),
    (15, "20260814_203600", "output/five_station_15t_hierarchical/20260814_203600/checkpoints/final.pt"),
    (15, "20260814_210606", "output/five_station_15t_hierarchical/20260814_210606/checkpoints/final.pt"),
    (20, "20260813_231902", "output/five_station_20t_hierarchical/20260813_231902/checkpoints/final.pt"),
    (20, "20260813_233319", "output/five_station_20t_hierarchical/20260813_233319/checkpoints/final.pt"),
    (20, "20260813_234330", "output/five_station_20t_hierarchical/20260813_234330/checkpoints/final.pt"),
    (20, "20260813_235156", "output/five_station_20t_hierarchical/20260813_235156/checkpoints/final.pt"),
    (20, "20260814_000258", "output/five_station_20t_hierarchical/20260814_000258/checkpoints/final.pt"),
    (25, "20260816_140832", "output/8_16/training/five_station_25t_hierarchical/20260816_140832/checkpoints/final.pt"),
    (25, "20260816_154929", "output/8_16/training/five_station_25t_hierarchical/20260816_154929/checkpoints/final.pt"),
    (30, "20260904_220529", "output/9_4/training/five_station_30t_hierarchical/20260904_220529/checkpoints/final.pt"),
]

# flat needs a --l2 ckpt only because eval_protocol loads one unconditionally; the
# flat mode never uses it for decisions.  Reuse the scale's hierarchical ckpt.
FLAT_RUNS = [
    (10, "15t:20260816_152139", "output/8_16/training/flat_mappo/20260816_152139/checkpoints/final.pt"),
    (10, "15t:20260816_154653", "output/8_16/training/flat_mappo/20260816_154653/checkpoints/final.pt"),
    (15, "20260816_140618", "output/8_16/training/flat_mappo/20260816_140618/checkpoints/final.pt"),
    (15, "20260816_152139", "output/8_16/training/flat_mappo/20260816_152139/checkpoints/final.pt"),
    (15, "20260816_154653", "output/8_16/training/flat_mappo/20260816_154653/checkpoints/final.pt"),
    (20, "15t:20260816_152139", "output/8_16/training/flat_mappo/20260816_152139/checkpoints/final.pt"),
    (20, "15t:20260816_154653", "output/8_16/training/flat_mappo/20260816_154653/checkpoints/final.pt"),
    (25, "20260816_140829", "output/8_16/training/flat_mappo/20260816_140829/checkpoints/final.pt"),
    (25, "20260816_152543", "output/8_16/training/flat_mappo/20260816_152543/checkpoints/final.pt"),
    (25, "20260816_155134", "output/8_16/training/flat_mappo/20260816_155134/checkpoints/final.pt"),
    (30, "20260904_220534", "output/9_4/training/flat_mappo/20260904_220534/checkpoints/final.pt"),
    (30, "20260904_220535", "output/9_4/training/flat_mappo/20260904_220535/checkpoints/final.pt"),
    (30, "20260904_220536", "output/9_4/training/flat_mappo/20260904_220536/checkpoints/final.pt"),
]

BASELINE_MODES = ["sa", "sa-online", "sa-nofc", "fcfs", "cruise"]


def run_eval(cmd, dest_dir: Path, drop_modes=(), dry=False):
    """Run eval_protocol into a temp dir, keep its per-episode + per-transit rows.

    per_episode.csv is the iid unit (one row per eval episode); per_transit.csv
    carries the raw delays, which is what the POOLED top-decile needs in order to
    be recomputed under episode resampling.
    """
    if dry:
        print("    " + " ".join(cmd))
        return True
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(cmd + ["--out-dir", td], cwd=PROJECT_ROOT,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"    ! FAILED: {(r.stderr or '').strip()[-300:]}")
            return False
        kept_any = False
        dest_dir.mkdir(parents=True, exist_ok=True)
        for name in ("per_episode.csv", "per_transit.csv"):
            src = Path(td) / name
            if not src.exists():
                print(f"    ! missing {name}")
                continue
            rows = list(csv.DictReader(open(src)))
            if drop_modes:
                before = len(rows)
                rows = [x for x in rows if x["mode"] not in drop_modes]
                if name == "per_episode.csv":
                    print(f"    dropped {before - len(rows)} rows of {drop_modes}")
            if not rows:
                continue
            with open(dest_dir / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
            kept_any = True
        return kept_any


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=int, nargs="+", default=[10, 15, 20, 25, 30])
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=900000)
    ap.add_argument("--sa-iters", type=int, default=300,
                    help="SA annealing budget per transit (300 = paper value; lower "
                         "values give a weaker local-experiment SA baseline)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    common = ["--incident", "--blocks", str(args.blocks),
              "--n", str(args.episodes), "--seeds", str(args.seeds),
              "--sa-iters", str(args.sa_iters)]

    ok = fail = 0
    for scale in args.scales:
        scen = f"five_station_{scale}t"

        # ---- learned (hierarchical), one eval per training run ----
        for s, run_id, ck in HIER_RUNS:
            if s != scale:
                continue
            if not (PROJECT_ROOT / ck).exists():
                print(f"  ! missing {ck} — skipped"); fail += 1; continue
            print(f"  {scale}t learned/{run_id}")
            if run_eval([sys.executable, "scripts/eval_protocol.py", "--scenario", scen,
                         "--l2", ck, "--l3", ck, "--learned-only", *common],
                        OUT_ROOT / f"{scale}t" / run_id.replace(":", "_"),
                        dry=args.dry_run):
                ok += 1
            else:
                fail += 1

        # ---- flat, one eval per training run ----
        l2_ref = next((ck for s, _, ck in HIER_RUNS if s == scale), None)
        for s, run_id, ck in FLAT_RUNS:
            if s != scale:
                continue
            if not (PROJECT_ROOT / ck).exists():
                print(f"  ! missing {ck} — skipped"); fail += 1; continue
            print(f"  {scale}t flat/{run_id}")
            if run_eval([sys.executable, "scripts/eval_protocol.py", "--scenario", scen,
                         "--l2", l2_ref, "--flat-ckpt", ck, "--flat-only", *common],
                        OUT_ROOT / f"{scale}t" / f"flat_{run_id.replace(':', '_')}",
                        dry=args.dry_run):
                ok += 1
            else:
                fail += 1

        # ---- checkpoint-independent baselines (one pass, learned dropped) ----
        if l2_ref:
            print(f"  {scale}t baselines ({'/'.join(BASELINE_MODES)})")
            if run_eval([sys.executable, "scripts/eval_protocol.py", "--scenario", scen,
                         "--l2", l2_ref, *common],
                        OUT_ROOT / f"{scale}t" / "_baselines",
                        drop_modes=("learned", "flat"), dry=args.dry_run):
                ok += 1
            else:
                fail += 1

    print(f"\n{ok} ok, {fail} failed. Rows under {OUT_ROOT}")


if __name__ == "__main__":
    main()
