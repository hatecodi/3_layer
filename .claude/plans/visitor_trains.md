# Plan: Visitor trains with advance notification + platform-aware observation

## Problem
- B3: Platform choice has no learning signal — policy can't see which platforms are occupied
- Initial delays + corridor disturbances create unrecoverable delay, ontime rate = 0%
- Current `parked_flag` is a scalar 0/1 — policy knows "something is blocked" but not what

## Design: Visitor trains (visitor = 访客列车)

### Generation (replaces `_generate_parked_trains`)
- Target stations: B, C, D (A, E are terminals with fewer conflicts)
- Per station: 1-2 visitors, probability 0.5 per visitor slot
- Types: 60% K-class (prio=3, dwell 300-900s), 40% G-class (prio=2, dwell 120-300s)
- Platform: uniform random across station platforms
- Time: random start within [0, MAX_EPISODE_TIME - duration]
- Pre-booked into resource calendar (reuse parked_windows mechanism)

### Advance notification → observation
- Visitor info enters `_obs_context()` as `visitors: {station_id: [(platform, start, end, priority), ...]}`
- Observation "Resource" section expanded: replace `parked_flag` (1 dim) with `platform_blocked` (4 dims)
- Per-platform blocked check: visitor overlaps with [t_now, t_now + 600] wide window
- New obs: 14 (self) + 5 (next) + 2 (avail, corridor) + 4 (p1-p4 blocked) = 25 dims
- `get_obs_dim()` → 25; `DEFAULT_OBS_DIM` → 25

### Training simplifications
- Remove `initial_train_delays` (random 60-300s at origin)
- Remove `corridor_disturbances`
- Keep `batch_window_sec = 60.0`

## Expected outcome
- Policy sees "p1 blocked, p2/p3 free" → learns to pick free platforms → avoids resolver conflicts
- Heuristic still picks p1 blindly → hits conflict → fallback/held → delay penalty
- Policy should outperform heuristic by learning to avoid blocked platforms
- Ontime rate should rise (no more guaranteed initial delays)

## Files changed
- `src/env/five_station_env.py`: `_generate_parked_trains()` → `_generate_visitors()`, update `_obs_context()`, update `reset()`
- `src/env/observation_builder.py`: replace `parked_flag` with `platform_blocked` bitmask (4 dims), update `ACTOR_RES_DIM`, `ACTOR_OBS_DIM`
- `src/marl/actor_critic.py`: `DEFAULT_OBS_DIM` 22 → 25, `GLOBAL_STATE_DIM` 14 → 14 (unchanged)
- `src/experiment/run_train.py`: remove initial_train_delays, remove corridor_disturbances

## Pass criteria
- Heuristic baseline: episodes complete, visitor conflicts cause held/fallback events
- Training: reward shows learning trend (delta > 0 over 2000 episodes)
- Policy trace: platform choices correlate with visitor occupancy (avoid blocked platforms)
