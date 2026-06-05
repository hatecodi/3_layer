# Plan: Fix achievable timetable

Date: 2026-06-05
Status: done

## What
Regenerate `configs/five_station/timetable.yaml` with physically achievable planned times.
All planned arrival/departure times replaced with earliest-possible times accounting for:
- Travel time from station approach to throat (30s)
- Throat clearance time (20s × 2 = 40s)
- Minimum dwell time (per station config)
- Platform clearance for pass-through trains (5s)
- Corridor transit time between stations (90s or 180s)

## Why
Current timetable has planned times that are physically impossible:
- K641 dep_plan=0 at A but earliest physical departure is 170s
- All trains accumulate cascading structural delay (90s at B, 160s at C, 230s at D, 300s at E)
- "On-time" bonus is unreachable → policy can never get positive reward
- The 306s performance plateau is ~80% structural, not policy's fault

## What changes
- `planned.arr` / `planned.dep`: all shifted to earliest achievable
- `bounds.arr_min` / `bounds.dep_max`: recalculated from new planned times + slack
- `bounds.dep_max = dep_plan + 120` (same slack as before)
- `bounds.dwell_min`: unchanged
- `candidate_platforms`: unchanged

## Alignment check
- [x] No hardcoded IDs: OK (config-driven generation)
- [x] Config-only expansion: OK (only YAML changed)
- [x] Action mask enforces bounds: OK (bounds valid by construction)
- [x] All trains have >=1 legal action at each station: YES (dep_max = dep_plan + 120 > minimum schedule)

## Pass criteria
1. Scenario loader validates without errors
2. Heuristic baseline runs and all trains complete
3. Every train has legal actions at every station
4. On-time (delay<=30s) is achievable for all trains when no corridor disturbance
