# Plan: Corridor speed-restriction incident scenario (coordination-rich disturbance)

Date: 2026-08-08
Status: draft

## What

Replace/augment the origin-bunching disturbance with a **corridor speed-restriction
incident**: a random corridor gets a temporary speed limit, creating a moving
bottleneck. Trains crossing during the window are slowed, form a platoon behind
the restriction, then must re-disperse after it lifts. This creates a
COORDINATION-RICH scenario with RECOVERABLE delay — the lever L3 speed control
actually operates on — unlike origin-bunching (60-160s unrecoverable delay that
swamps every method, leaving learned ≈ SA).

## Why

The current `inject_bunched_delay` injects 60-160s delays at origin that nobody
can recover, so learned ≈ SA (the coordination value is drowned). A corridor
speed-restriction incident:
- creates delays that speed control CAN recover (platoon re-dispersion);
- forces real cross-train interaction at the bottleneck (who approaches, who
  recovers) where greedy per-transit SA is blind to the system;
- is realistic/citable (wind-caused speed restrictions — Cui et al. 2022);
- compounds with the 20/25-train scalability arm (more trains → longer platoons
  → larger coordination value).

## Changes

### 1. `src/env/disturbance.py` — new `inject_corridor_incident`
- Signature: `inject_corridor_incident(cfg, rng, corridor_ids) -> dict` returning
  `{corridor_id: (start, end, speed_factor)}` or `{}` (no incident).
- Config-driven under `time_generation.corridor_incident`:
  `enabled / prob / min_time / duration_range [100,200] / factor_range [0.4,0.6]`.
- Picks a random corridor, start time ≥ min_time, duration, speed factor.

### 2. `src/env/five_station_env.py` — wire the incident
- `__init__`: `self._corridor_incidents = {}`.
- `reset()`: generate an incident if enabled+prob (using `self.rng`).
- `_advance_trains`: for each corridor transit, compute `speed_factor`:
  `f = incident[corridor][2]` if the transit overlaps the incident window
  (`not (entry + transit < start or entry > end)`), else `1.0`. Pass to the
  L3 sim section-speed computation (worker + backward-compat paths).
- **L3 obs**: pass `speed_factor` into `build_l3_section_observation`; the
  reserved 10th slot carries `1.0 - speed_factor` (0=normal, >0=restricted) so
  the worker can SEE and react to the restriction.

### 3. `src/control/l3_sim.py` — apply restriction to section speeds
- `section_run_time(section, speed_mode, priority, speed_factor=1.0)`:
  effective speed *= speed_factor (min with section limit).
- `section_progress(..., speed_factor=1.0)` passes it through.
- Worker path in the env uses `section_progress(..., speed_factor)`; backward-compat
  single-transit path multiplies its run by the factor (or passes to transit).

### 4. `src/env/observation_builder.py` — restriction feature
- `build_l3_section_observation(..., speed_factor=1.0)`: reserved slot
  `min(max(1.0 - speed_factor, 0.0), 1.0)`.

### 5. Config — `configs/five_station/timetable.yaml` (and 15t/20t variants)
- Add `time_generation.corridor_incident` block (evidence-based, see below).

## Evidence-based parameters (railway data, 2026-08-08)

Real HSR heavy-rain rules (防灾安全监控系统 / 雨量监测, multiple sources agree):
- 1h rainfall >= 45mm (暴雨) -> speed limit **120 km/h** = **0.4 x** the P2 300 km/h
  cruise used in this sim.  >= 60mm (大暴雨) -> **45 km/h = 0.15x**.
- Normal HSR minimum tracking headway ~= 180 s (CTCS-2/3); speed restriction
  INCREASES it (wet rails -> longer braking distance -> larger spacing), which
  is the physical mechanism for capacity drop under rain.

Config defaults (sources cited in coding_log / handoff):
```yaml
corridor_incident:
  enabled: true
  prob: 0.6            # probability an incident occurs per episode
  min_time: 100        # earliest incident start (s)
  duration_range: [100, 200]
  factor_range: [0.15, 0.4]   # 大暴雨(45km/h) .. 暴雨(120km/h); HARD DATA
  safety_factor: 1.5   # moving_block_safety_m multiplier (wet-rail braking
                       # distance increase) — ESTIMATED, no direct citation;
                       # flag as such in the paper
```
The incident applies BOTH the speed-factor restriction AND the safety-distance
increase (`moving_block_safety_m * safety_factor` during the window) — that
combination is the physical model of "跟车间距增大" under rain.

### 6. Verification protocol (existing infra)
- `scripts/train_l3_convergence.py` — 3-seed L3 training with plateau check.
- `scripts/plot_l3_conv_protocol.py` — convergence figure.
- `scripts/analyze_l3_behavior.py` — does the worker react to the restriction
  (restricted corridor → different speed choice than clear)?
- `scripts/eval_protocol.py` — held-out eval, stratified by incident occurrence,
  learned vs SA vs CRUISE.

## Expected outcome

- learned (team reward) beats greedy-SA on the incident-affected subset: better
  platoon re-dispersion → lower p90/mean_dep, higher ontime.
- Behavior analysis shows the worker uses the restriction signal (e.g., no FAST
  while restricted, recover after).
- Then run 20/25-train scalability: incident + more trains → larger gap.

## Alignment check
- [ ] No hardcoded IDs: corridor from config; factor/duration from config.
- [ ] L3 obs stays 10-dim (reserved slot reused).
- [ ] L1/L2 action masks untouched.
- [ ] Config-only expansion (incident config + corridor set).
- [ ] Fallback re-verifies bounds — unchanged.

## Pass criteria
1. `python -m pytest tests/test_phase12.py tests/test_l3_physics.py -q` green.
2. A unit test: a restricted corridor produces slower transit; a train outside
   the window is unaffected.
3. 3-seed convergence: entropy→0, eval plateau (slope criterion).
4. Behavior: worker uses restriction signal (action differs when restricted).
5. held-out eval (30 seeds): learned ≥ SA on incident-affected subset (p90
   and/or mean_dep clearly better).

## Out of scope
Mid-corridor re-decision (trains decide speed DURING the incident in real time —
still one-shot at entry). Event-driven L3. L2 rolling rescheduling (the incident
is the trigger; that workstream stays deferred).
