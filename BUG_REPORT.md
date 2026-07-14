# Bug Report: Resource calendar per-step — no cross-step platform occupancy tracking

Date: 2026-07-14

## Summary

The resource calendar is created fresh each scheduling step. Platform occupancy from
grants in previous steps is not persisted, allowing two trains to occupy the same
platform with overlapping time windows.

## Root cause

`_build_requests()` (five_station_env.py:552) creates a new `ResourceCalendar` per
station per step:

```python
station_cal = ResourceCalendar(min_gap_sec=5.0)  # line 552
```

`resolve_all()` (resource_manager.py:255) creates another fresh calendar:

```python
calendar = ResourceCalendar(min_gap)  # line 255
```

Both are thrown away after the step. When a train is granted departure from station X
to station Y at step N, its platform occupancy at Y (from t_arr to t_dep) is NOT
carried into step N+1's calendar. At step N+1, other trains at Y can be assigned to
the same platform, creating overlaps.

## Reproduced violations (5 episodes, no VT, no initial delay)

### Platform overlap

```
Station C, Platform p2:
  G5038: arr=283.0  dep=336.0  (occupies p2 for 53s)
  K834:  arr=286.0  dep=298.0  (also on p2)
  Gap: -50.0s (min=5.0s) → HARD OVERLAP
```

### Throat conflicts (< 6s separation)

```
B: G230(A_p2 dep=31.0) vs G1487(C_p2 dep=36.0)  gap=5.0s
B: G5038(A_p2 dep=77.0) vs G238(A_p3 dep=79.0)  gap=2.0s
D: G230(C_p2 dep=217.0) vs G238(C_p3 dep=221.6) gap=4.6s
```

## Why ontine was 100% despite violations

`compute_metrics()` in reward.py calculates ontime as:

```python
dep_delays.append(abs(grant.W_out[0] - bounds.t_dep_plan))
n_ontime_dep = sum(1 for d in dep_delays if d <= 15.0)
```

This only checks departure time deviation from plan, NOT resource allocation
correctness. A train can be "on time" departing from a platform that another train
is already occupying.

## Code locations

| File | Line | Issue |
|------|------|-------|
| `five_station_env.py` | 552 | Fresh `ResourceCalendar` per station per step — no cross-step state |
| `resource_manager.py` | 255 | Fresh `ResourceCalendar` in `resolve_all` — same problem |
| `five_station_env.py` | 710 | `_advance_trains` moves trains but doesn't book destination platform |
| `resource_manager.py` | 54-91 | `ResourceCalendar` has no persistence mechanism |

## Fix: Persistent calendar (implemented 2026-07-14)

Added `_persistent_cal` (ResourceCalendar) as env attribute. Three injection points:

1. **`_build_requests`**: seed `station_cal` with persistent bookings for this station
2. **`_resolve_requests`**: dump persistent → `parked_windows`, pass to `resolve_all`
3. **After resolve**: book each grant's platform+throat windows into persistent calendar

### Verified
- Platform overlap at C/p2 (G5038 vs K834): **FIXED** — K834 now gets C/p3
- Throat "busy" flags in validation script are **false positives** — at A/B/D/E each
  platform has independent conflict group, so different platforms can depart simultaneously
