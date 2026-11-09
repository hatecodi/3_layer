# Plan: 15t hierarchical MAPPO 7000-episode run + smoothed convergence plot
Date: 2026-10-29
Status: done

## What
1. `src/experiment/run_train_hierarchical.py`: add `--total-episodes N` CLI flag.
   When set, the selected phases' hard-coded episode counts (PHASES: A=3000, B=3000)
   are rescaled proportionally to sum to N (A:B stays 1:1 → 3500/3500 for N=7000).
   Default (unset) preserves existing behaviour exactly.
2. New `scripts/plot_hier_training_curve.py`: read `training_metrics.csv`
   (+ sibling `eval_ontime.csv` if present) and render smoothed training
   trajectories (rolling mean ±1 std, window 100): mean reward, mean departure
   delay, departure/arrival on-time rate, with the Phase A→B boundary marked.
   Also write a `conv_check.csv` with per-phase and last-window statistics so the
   convergence verdict is CSV-first (per user's plotting convention).
3. Run: 15t, incident disturbance, seed 0, `--phases A,B --total-episodes 7000`,
   `--eval-every 200 --eval-n 12 --eval-seed-base 900000`,
   `--out-root output/10_29/training`.
4. Plot from the produced CSV, inspect convergence, log paths in `coding_log`.

## Why
User asked for one 15-train hierarchical MAPPO training run of 7000 episodes and a
smoothed training-trajectory figure to judge convergence. Episode count was
hard-coded at 3000+3000, so a CLI override is needed (no protocol change).

## Alignment check
- [x] No hardcoded IDs: OK (episode counts only)
- [x] Observation dim: N/A
- [x] Action mask enforces bounds: N/A (untouched)
- [x] Local critic: N/A
- [x] Fallback re-verifies bounds: N/A
- [x] Config-only expansion: OK

## Pass criteria
- Smoke: `--total-episodes 2` runs two episodes (one per phase) and writes CSV.
- Full run: `training_metrics.csv` has 7000 rows, all statuses success;
  `eval_ontime.csv` has 35 held-out checkpoints.
- Figure + `conv_check.csv` produced under the run directory; convergence verdict
  based on rolling-window stationarity (last 1000 vs prior 1000 eps).
