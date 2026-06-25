# Plan: Batch scheduling window for intermediate-station conflicts

## Problem
Trains arrive at intermediate stations at staggered times (10-30s apart).
The current `_pending_trains()` only returns trains with `t_now >= arrival_at_current`.
Combined with time-jump logic, each train is processed alone → n_granted=1 per step → no multi-agent competition.

## Change
Add a `batch_window_sec` lookahead to `_pending_trains()` so trains arriving within
a configurable window are batched together. The resolver then handles true conflicts.

## Files changed
- `src/env/five_station_env.py`: add `batch_window_sec` param, `lookahead` to `_pending_trains()`, pass in `_build_requests()`
- `src/experiment/run_train.py`: pass `batch_window_sec` to env constructor

## Expected outcome
- `n_granted` per step increases from 1 to 2-4
- Real platform conflicts at B, C, D stations
- Policy learns coordination under resource scarcity

## Pass criteria
- Heuristic baseline: 5-station episode still completes successfully
- Trace shows multi-train batches at intermediate stations
