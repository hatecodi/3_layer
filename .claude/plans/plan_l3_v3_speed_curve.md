# Plan: L3 v3 — sequential per-section speed-curve learning, headway as HARD boundary

Date: 2026-08-08
Status: in_progress

## What

Upgrade L3 from a single 3-way speed choice per corridor to a **sequential
per-section decision process** (worker picks CRUISE/FAST/SLOW for each block
section as the train traverses, inside the `_advance_trains` call). The
objective is headway management: under moving-block, the worker must time its
speed curve to keep the safe interval to the front train (no catch-up, no wasted
gap) while recovering delay and saving fuel.

**Key design decisions (user):**
- headway is a HARD boundary (moving-block safety distance always enforced by
  the simulator, never a reward term) — invariant "hard boundaries via
  constraint, not reward".
- **energy/fuel removed** from the L3 reward (simplifies convergence; objective
  is punctuality only).
- **L3 sees GLOBAL**: the worker's reward includes a global/team punctuality
  component (system delay), and the obs carries downstream congestion — so L3
  learns cross-train coordination (execution-layer rescheduling). This is what
  lets the learned worker beat a greedy per-transit SA baseline on global
  metrics (SA optimizes only its own next-station arrival deviation).
- **SA baseline**: per-corridor-transit simulated annealing over the SAME
  per-section speed space, SAME local objective (own arrival deviation, headway
  hard). It is the greedy local upper bound; the paper claim is that global-
  coordinated L3 beats greedy-SA.
- Paper framing: ONE experiment family (train-count × disturbance) serves BOTH
  claims — hierarchical rescheduling quality AND scalability (gap widens with
  traffic).

## Changes

### 1. `src/control/l3_sim.py`
- `L3CorridorState.front_time_at(corridor_id, x)` — front occupant's absolute
  time at position x (linear interp of its per-section profile); +inf when empty.
- `L3Simulator.section_run_time(sec, speed_mode, priority)` — traverse time for
  one section (v = min(mode_speed, section.limit_mps)).
- Keep `record_occupancy` (per-section entry/exit times), `calibrate_corridor_sections`,
  `_sec_bounds`, moving-block params.
- **Headway enforcement stays hard**: in the stepwise loop the sim imposes
  `wait = max(0, front_time_at(pos_k + safety) + margin - t)` and the train waits.

### 2. `src/env/five_station_env.py` — `_advance_trains`
Replace the single `l3_sim.transit()` with a compressed stepwise loop (no change
to the global time-step model):
```
corridor_entry = throat_exit            # origin throat (existing throat headway)
t = corridor_entry
for k, sec in enumerate(sections):
    pos_k = bounds[k]
    obs_k = build_l3_section_observation(...)          # includes headway gap at this section
    if worker: action_k, logprob_k, value_k = worker.select_action(obs_k)
    else:      action_k = CRUISE                       # backward-compat baseline
    run_k = section_run_time(sec, action_k, prio)
    wait_k = max(0, front_time_at(pos_k + safety) + margin - t)   # HARD boundary
    t += wait_k + run_k
    reward_k = -fuel_coef * energy(action_k) * len_k   # fuel only, no safety term
    worker.buffer.add(L3Transition(obs_k, action_k, logprob_k, reward_k,
                                   value_k, done=(k == n_sections - 1)))
arrival = t + approach + throat_in
```
- **A1 revision**: per-section transitions `done=False` between sections,
  `done=True` only at the last — sections ARE a Markov chain under the worker's
  policy, so GAE bootstraps across them correctly.
- **Terminal reward** (added on the last section's reward, or as the arrival
  outcome): ontime bonus, -late*late, -early*early_wait (arriving before the
  destination platform is free = bottleneck alignment).
- Backward-compat: `l3_worker is None` → all-CRUISE; heuristic/CRUISE eval paths
  unchanged.

### 3. `src/env/observation_builder.py`
`build_l3_section_observation(...)` (10-dim, headway-centric):
front headway gap at this section (front_time_at(pos+safety) - t), accumulated
wait, cumulative delay, section_idx/n_sections, section length ratio, remaining
distance, speed_ratio, next-station platforms free, throat_gap, reserved.

### 4. `src/env/reward.py`
- `compute_l3_section_reward(...)` → **removed/deprecated** (no energy objective).
- Terminal (per corridor): `compute_l3_terminal_reward(arrival_dev, platform_free)`
  → ontime bonus - late - early_wait (platform alignment), PLUS a **global team
  term** `-coef * system_delay` (sum of all trains' cumulative delay) so the
  worker learns cross-train coordination. λ_team small so local dominates.
- Keep `compute_l3_reward` for backward compat.

### 5. `src/marl/l3_worker.py`
No structural network change (10-dim obs, 3-way categorical). Buffer holds one
transition per section; GAE already uses the `done` flag correctly.

## Why
The single 3-way L3 collapsed to CRUISE (problem too simple, CRUISE = safe
default). Per-section sequential decisions create real cross-section credit
(speed on section k affects the wait at section k+1 via the HARD headway
boundary), so the learned worker can exceed a simple rule. Treating headway as a
hard boundary (not a reward) removes the "be conservative to avoid wait" failure
mode and aligns with the project's constraint-not-reward invariant.

## Alignment check
- [x] No hardcoded IDs: sections/speeds from config; worker shared across trains.
- [x] L2 obs unchanged; L3 obs stays 10-dim.
- [x] L1/L2 action masks/bounds untouched.
- [x] Config-only expansion retained.
- [x] Headway enforced as hard boundary (constraint, not reward) — invariant #4.
- [ ] GAE: done=False between sections, done=True at corridor end (revises A1).

## Pass criteria (validation gate)
1. `python -m pytest tests/test_phase12.py tests/test_l3_physics.py -q` green.
2. Short Phase B training (500-1000 ep): L3 speed distribution NON-degenerate
   (all of CRUISE/FAST/SLOW used, varies with state) — NOT 100% CRUISE.
3. Heuristic+CRUISE eval unchanged (~76% ontime / ~21s under delay) — no regression.
4. A trained per-section worker beats the CRUISE baseline on delay/ontime.

## Out of scope (v1.5+)
Full event-driven in-transit model (trains traverse over global steps). Per-section
`limit_mps` speed-restriction learning. L2 rolling rescheduling and the add-trains
scalability study come AFTER L3 v3 is validated.
