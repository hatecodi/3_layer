# Plan: Persistent resource calendar across steps
Date: 2026-07-14
Status: in_progress

## What
Add a persistent `ResourceCalendar` to `FiveStationEnv` that survives across
scheduling steps. After each step's grants, book platform+throat windows. Pass
accumulated bookings as `parked_windows` to `resolve_all` in subsequent steps.

## Why
Two trains landing on the same platform because grants from different steps
don't see each other. Example:
- t=270: K834 granted C/p2 [286, 298] → calendar discarded after step
- t=280: G5038 arrives at C, gets C/p2 [283, 336] → fresh calendar unaware of K834

## Implementation
1. `FiveStationEnv.__init__`: create `self._persistent_cal = ResourceCalendar()`
2. After `resolve_all` in `_grant()`: book each grant's resource windows into persistent calendar
3. Before `resolve_all`: dump persistent calendar to `parked_windows` dict and pass it
4. `_build_requests`: also seed `station_cal` with persistent bookings for this station
5. Clean up: no explicit cleanup needed — ResourceCalendar stores all bookings,
   conflict check handles them correctly; old windows naturally expire after passed

## Alignment check
- [x] No hardcoded IDs: OK
- [x] Observation 74-dim: N/A
- [x] Action mask enforces bounds: N/A
- [x] Local critic 444-dim: N/A
- [x] Fallback re-verifies bounds: N/A
- [x] Config-only expansion: OK
