# Plan: delay-dissipation time (DDT) as a recovery metric
Date: 2026-09-21
Status: done

## What

Add a **delay-dissipation time** metric — the temporal dimension of recovery that
the current metric set lacks (tail severity answers "how bad", mean/on-time rate
answer "how good on average", neither answers "how fast does the system come
back").

Definition (pre-registered cell in bold):

> After a disturbance, from `t_rec_start` onward, DDT is the elapsed time until
> **K=3 consecutive arrival events** all satisfy **|arr_delay| ≤ θ=30 sim-s**. If
> no such run occurs before episode end, the episode is **right-censored**.

Files:

1. **`src/experiment/recovery_metrics.py`** (NEW)
   - constants `DDT_THETA_SEC = 30.0`, `DDT_K = 3`,
     `DDT_GRID_THETA = (15, 30, 45)`, `DDT_GRID_K = (2, 3, 5)`
   - `dissipation_time(events, t_rec_start, theta, k, value_of=None)` → seconds
     after `t_rec_start` at which the confirming (K-th) event arrives, or `None`
     when censored. Events sorted by realised `actual_arrival`; run counter reset
     on any non-compliant event.
   - `recovery_stats(eps, ...)` → dict: `n_eps` (episodes with an anchor),
     `n_recovered`, `censor_pct`, `med`, `p25`, `p75` plus `mean_censored`
     (censoring time substituted for `None`, reported only for reference).
   - `recovery_grid(eps)` → rows over θ×K for the sensitivity table.
   - `floor_values(events, floor_idx)` → per-event excess over a reference
     execution layer, keyed `(seed, train_id, corridor)`; events with no
     counterpart are dropped (coverage returned for reporting).

2. **`scripts/eval_protocol.py`**
   - `run_episode`: record `seed` in the result dict, and compute
     `t_rec_start` — **incident arm**: end of the restriction window
     (`max(v[1] for v in env._corridor_incidents.values())`); **bunching arm**:
     release time of the last bunched train (`max(arrival_at_current)` over
     trains still at their origin with `arrival_at_current > planned_dep + 1`,
     requiring ≥2 to count as a bunch). `--no-delay` arm → `None`.
   - `report_recovery(eps, label, floor_eps=None)`: prints the pre-registered
     cell, the censoring rate, and the θ×K grid; returns a stats dict for CSV.
     With `floor_eps` (incident arm, reference = `cruise`) the same K-run rule is
     applied to the **paired excess over the CRUISE floor** — the absolute form is
     censored by construction there, because incident added delay is capped at
     `max_added_delay_sec = 80` while L3 cannot recover all of it.
   - Wire into `main()`: incident arm (`floor_eps = cruise` for every non-cruise
     mode), legacy bunching arm (`floor_eps = None`), skip the clean arm. Print
     the mode ranking for the pre-registered cell (the claim is rank stability
     across the grid, not the absolute value).
   - CSV: `recovery.csv` (per mode: `ddt_med`, `ddt_p25`, `ddt_p75`,
     `ddt_censor_pct`, `n_eps_dist`, `ddt_excess_med` in the incident arm) and
     `recovery_grid.csv` (long: mode, theta, k, med, censor_pct).

3. **`docs/notation.md`** — one-line symbol entry for DDT (paper phase: metrics
   must be citable from docs, not from scripts).

No change to `src/env/` — the anchor is derivable from state the eval script
already reads (`env._corridor_incidents`, `env._journeys`), so **zero risk to the
training paths**.

## Why

The eval protocol has no recovery-time metric; the paper's resilience claim
("bounded, quickly-dissipating delay") currently rests on the mean/p90/max of the
delay distribution, which cannot separate "slow to recover" from "uniformly
mediocre". DDT closes that gap and is the metric an L3 coordination story needs,
since L3 works by re-dispersing a platoon over the following transits.

## Alignment check
- [x] No hardcoded IDs: OK (train/corridor IDs only read, never written)
- [x] Observation 74-dim: N/A (eval instrumentation only)
- [x] Action mask enforces bounds: N/A
- [x] Local critic 444-dim: N/A
- [x] Fallback re-verifies bounds: N/A
- [x] Config-only expansion: OK

Design invariants preserved / deliberately handled:
- `|arr_delay|` not signed delay — the repo already has a fixed bug where
  systematic early arrival masked L3 recovery value (`l3_sim.py:580`); a signed
  threshold would let "run everything early" fake a zero DDT.
- Anchor at disturbance END, not onset — otherwise the metric partly measures the
  disturbance duration itself.
- Reports **median + censoring rate**, never a mean, because a time-to-event
  statistic is heavy-tailed and right-censored at episode end.
- θ=30 sits below the 80 sim-s incident cap (attainable) and is deliberately
  *distinct* from `ONTIME_THRESHOLD_SEC = 15` so DDT is not redundant with the
  on-time rate.

## Pass criteria

1. `python scripts/eval_protocol.py --scenario five_station_15t --l2 <final.pt>
   --l3 <final.pt> --incident --blocks 1 --n 8 --seeds 900000`
   runs, prints a `delay dissipation` block with the pre-registered cell + grid
   for all 6 modes, and writes `recovery.csv` / `recovery_grid.csv` with no
   `nan` medians for the cruise mode and a non-100% censoring rate for at least
   one mode.
2. Same without `--incident` (origin-bunching arm) — DDT computed on all events,
   anchored at the last bunched release; episodes with no bunch excluded from
   `n_eps_dist` rather than counted as censored.
3. `--no-delay` arm prints no DDT block (no disturbance to recover from) and
   `recovery.csv` is absent/empty — no crash.
4. Existing CSV schemas unchanged (`plot_conv_scalability.py:144` reads
   `ontime.csv` as `mode,ontime`), per_block.csv untouched.

## Outcome (all 4 criteria met)

- incident arm n=4/n=8/n=20: 6 modes out, `paired=100%`, ranking printed with CRUISE
  excluded from the ordering; `recovery.csv` + `recovery_grid.csv` written.
- bunching arm: absolute口径, ranking `sa<sa-online<sa-nofc<fcfs<learned<cruise`.
- clean arm: no DDT block, no `recovery.csv`, no crash.
- Pre-existing crash found & fixed: `round(method_ms, 1)` with `_decision_ms -> None`
  for `sa-online` killed every default-mode run (both `report_clean` and
  `report_incident`). NOT fixed (out of scope, reported): `--out-csv` raises
  `ValueError` on extra keys — `csv.DictWriter` default `extrasaction="raise"`.
- Measured: disturbed-episode count varies a lot per block (35/40/28 of 60 at
  seeds 900000/1900000/2900000) even though `prob=0.6` converges at 1000 seeds
  -> `n_eps_dist` must always be reported alongside the median.
- Nothing written to `output/`; the official 3x60 run should go to `output/9.21/`.
