# Plan: L3-only training script
Date: 2026-07-14
Status: done

## What
- New script `src/experiment/run_train_l3_only.py` — loads frozen L2 Manager from checkpoint, trains only L3 Worker
- Added `get_last_stats()` to `L3WorkerTrainer` for monitoring
- Added `l3_speed_choices` tracking to `FiveStationEnv._advance_trains` and info dict

## Why
L3 observation had a bug (train_class instead of priority for speed lookup). Need to retrain L3 Worker from scratch with a frozen, well-trained L2 Manager to verify convergence.

## Alignment check
- [x] No hardcoded IDs: OK
- [x] Observation 74-dim: N/A
- [x] Action mask enforces bounds: N/A
- [x] Local critic 444-dim: N/A
- [x] Fallback re-verifies bounds: N/A
- [x] Config-only expansion: OK

## Pass criteria
Script runs, loads L2 checkpoint, trains L3, produces CSV with training metrics.
