# Plan: Fix L3 observation speed lookup — use priority not train_class
Date: 2026-07-14
Status: done

## What
`src/env/five_station_env.py:742-743` — change from `train_class` to `priority` when looking up speeds for L3 observation.

## Why
- `L3Physics.get_speeds(priority: int)` expects an int priority (1, 2, 3)
- `l3_physics.yaml` keys are `"1"`, `"2"`, `"3"` (priority strings)
- `transit()` passes `priority=self.cfg.trains[g.train_id].priority` — correct
- But L3 obs builder passes `train_class` ("G", "K") — wrong
- Since "G"/"K" don't match "1"/"2"/"3", all trains fall back to default speeds (25/30 m/s)
- Result: L3 worker observations have wrong recovery_margin, speed_ratio — policy learns from garbage

## Alignment check
- [x] No hardcoded IDs: OK
- [x] Observation 74-dim: N/A
- [x] Action mask enforces bounds: N/A
- [x] Local critic 444-dim: N/A
- [x] Fallback re-verifies bounds: N/A
- [x] Config-only expansion: N/A

## Pass criteria
L3 observation uses the same speeds as L3 transit simulation — both driven by `priority`.
