# Plan: extend v7 L2 (Phase A) to 7000 episodes
Date: 2026-11-07
Status: done

## What
- No code change. Resume the v7 L2 run (Phase A only, SA-L3 frozen) from the
  2000-ep `final.pt` and continue to 7000 episodes.
- User decision: keep `--control-mode relevant` (v7 setting), so the overall
  on-time rate remains ~75% heuristic-driven; convergence is judged on reward /
  on-time trend plus the paired 180-ep eval against heuristic.
- Command:
  `--phases A --total-episodes 7000 --incident --seed 0`
  `--l2-ckpt output/11_06/training_v7/.../final.pt`
  `--control-mode relevant --extern-alpha 0.10 --yield-cap --yield-tail-coef 0.0`
  `--eval-every 200 --eval-n 6 --eval-seed-base 900000`
  `--out-root output/11_06/training_v7_l2_7k`
- Then plot reward / departure on-time / dep-delay with
  `scripts/plot_hier_training_curve.py` and re-run the convergence table.

## Why
The 2k v7 L2 shows a flat departure on-time rate (~0.75) and a reward curve
that rises 10.0 -> 11.0 then oscillates without a clear plateau. Stationarity
of the last 500 episodes is not evidence that the policy converged to its best
level. The older 7k run (different reward version) had Phase A on-time still
rising at ep 2000 and reaching 0.80 by ep 3500, so 2k is too short to judge.
Note: relevance gating lets the policy control only ~23.5% of decisions
(12/51 per episode), so the overall on-time metric is partly heuristic-driven;
longer training plus the existing paired eval is the way to tell whether the
learned subset keeps improving.

## Alignment check
- [x] No hardcoded IDs: N/A (no code change)
- [x] Observation dims: unchanged
- [x] Action mask enforces bounds: unchanged
- [x] Local critic: unchanged
- [x] Fallback re-verifies bounds: unchanged
- [x] Config-only expansion: unchanged

## Pass criteria
- 7000 episodes complete (status success, no deadlock/truncation).
- New convergence table: last-500 and last-1000 windows; report whether
  ontime_rate / mean_reward still trend or are flat.
- End-to-end 180-ep paired eval of the 7k L2 + SA-L3 vs heuristic and vs the
  2k L2 result, so any gain is quantified.
