# Plan: L3 v2 block-section physics + L2 rolling rescheduling

Date: 2026-08-08
Status: in_progress

## What

Two workstreams, in dependency order (L3 substrate first, L2 rescheduling on top).

### Workstream A — make L3 honest physics (fixes S2/S3/S4/S5/S6 from evaluation)

A1. **L3 training signal fix (S3)** — `src/marl/l3_worker.py`, `src/env/five_station_env.py`
   - Set `done=True` on every L3Transition. The L3 Worker makes one independent
     speed decision per corridor leg; the next leg is separated by an L2 station
     decision, so legs are NOT a Markov chain under the L3 policy. GAE then
     collapses to immediate-reward advantage (correct single-step MDP treatment).
   - No buffer segmentation needed.

A2. **Throat delay visible to L2 reward (S6)** — `src/env/reward.py`, `src/env/five_station_env.py`
   - Track `self._throat_delays` per train; pass into `compute_layered_rewards`;
     add `-0.02 * throat_delay * prio_weight`. Currently only headway delay is
     propagated; C-station shared out-throat serialization is invisible to L2.

A3. **Config-driven block-section corridor + calibration (S2+S4)** — `configs/five_station/l3_physics.yaml`, `src/control/l3_sim.py`, `src/env/five_station_env.py`, `src/env/observation_builder.py`
   - New YAML `corridor_sections`: each corridor -> ordered list of sections
     `{length_m, limit_mps?}`. `limit_mps` per section = configurable speed
     restriction -> piecewise "speed curve".
   - **Calibration**: scale section lengths so that CRUISE reproduces the
     timetable reference transit (min planned transit on the corridor at P2
     cruise). Fixes systematic early-arrival (on-time dep + CRUISE -> on-time arr).
     `scale = (ref_transit - overhead) * cruise_ref / sum(length_m)`.
   - **Fixed-block headway** in `L3Simulator`: track per-section occupancy
     (entry/exit per occupant). New train's earliest corridor entry =
     `max_i( front_clear_i + block_margin - sum(run_j for j<i) )`. FIFO by
     construction (per-section last occupant). Min floor = `corridor_headway_sec`.
   - **Speed modes**: run_i = len_i / v, v = min(mode_speed, section limit_mps),
     mode_speed = cruise (P-cruise), fast (P-max), slow (P-cruise/1.2).
   - Backward-compat: if `corridor_sections` absent, fall back to current
     single-segment behavior.
   - **Honest L3 obs** (`five_station_env.py`): replace hardcoded
     `arrival_slack_ratio=0.5` and fake `nxt_free=max(total-1,1)` with real
     values (slack from planned_arr vs estimated_arr; next-station free
     platforms read from persistent_cal).

### Workstream B — L2 rolling rescheduling

B1. **Mid-episode disturbance** — `src/env/disturbance.py`, `src/env/five_station_env.py`, `configs/five_station/timetable.yaml`
   - New `inject_mid_episode_delay()`: once per episode (config prob) after a
     min time, delay a random in-transit train by `uniform(delay_min, delay_max)`
     by shifting `journey.arrival_at_current`. Drives re-scheduling.
   - Config under `time_generation.mid_delay` (`enabled/prob/min_time/delay_range`).

B2. **Stale future-booking pruning (the re-optimization lever)** — `src/env/five_station_env.py`
   - With `batch_window_sec=120`, trains are granted future station windows
     before they arrive. When a mid-episode delay shifts a train's ETA past its
     booked window, that booking is stale and blocks other trains.
   - Add `_prune_stale_bookings()` at the top of `step()`: for each persistent
     booking, if the owning train's ETA to that station > booking start + buffer,
     drop the booking; the delayed train is re-granted fresh on arrival.
   - Applies to MAPPO and heuristic equally (it lives in the env).

B3. **Hold/wait entry action** — `src/control/action_mask.py`, `src/marl/actor_critic.py`, run scripts
   - Add `"hold"` to `ENTRY_CHOICES` (entry head 6 -> 7): a held train makes no
     request this step, replans next step (reward -3.0, so policy learns sparing
     use). Gives the policy an explicit "wait for downstream to clear" lever.
   - Ripple: `ENTRY_DIM` 7, mask padding, `requested_entry_time`/`action_to_request`
     treat "hold" as a hold sentinel, `_build_requests` counts it as held.

## Why

The three-layer claim ("L3 real physics + hierarchical MAPPO", commit 2e7dd5e)
rests on an L3 that is a closed-form transit function with (a) systematic
early-arrival due to L3 transit != timetable (S2), (b) a GAE that bootstraps
across independent transitions (S3), (c) no block-section structure and a dead
120s headway config (S4), (d) throat delays invisible to L2 (S6). Workstream A
makes L3 defensible and config-driven (project invariant: resources = YAML only).
Workstream B turns L2 into a true rolling re-scheduler under mid-run disturbance,
which is the literal research question (better recovery than rule-based
rescheduling) and gives the paper its realistic experiment.

## Alignment check

- [ ] No hardcoded IDs: sections keyed by config corridor id; speeds per priority
      from existing `train_speeds`; no new magic station/train/platform.
- [ ] L2 observation unchanged (31-dim); L3 obs stays 10-dim.
- [ ] Action mask enforces Layer 1 bounds: unchanged except "hold" is always legal.
- [ ] Local critic (K+1)x31 unchanged; GlobalCritic unchanged.
- [ ] Fallback re-verifies bounds: unchanged.
- [ ] Config-only expansion: block sections added via `l3_physics.yaml`
      `corridor_sections`; mid-delay via `timetable.yaml`; no station/train added.

## Pass criteria

- A1: GAE over L3 buffer treats transitions independently (done=True); no
      cross-train bootstrap.
- A2: throat_delay appears in L2 reward breakdown (reward_breakdown shows it).
- A3: `python -c` sanity: on-time departure + CRUISE reproduces reference transit
      within 1e-6; block headway < single-segment headway when sections short.
- A3: `python -m pytest tests/test_phase12.py -v` still green (+ new unit tests
      for section headway and calibration).
- A3: heuristic baseline (no learning) runs an episode with L3 v2 without
      deadlock and reports delay/ontime.
- B1/B2: a mid-episode delay shifts a train's ETA, its stale booking is pruned,
      and another train can claim the freed platform (unit test or traced episode).
- B3: entry head = 7; a "hold" action produces a held train that replans next step.

## Out of scope (documented)
- Full accel/decel trapezoidal speed curve (approximated by per-section
  `limit_mps`). Event-driven L3 state machine (kept as departure-time computation;
  mid-corridor re-decision is future work). SUMO validation of L3 v2.
