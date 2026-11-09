# Plan: Station-level overtaking (越行) + learned yield

Date: 2026-11-06
Status: draft (awaiting approval)

## Context

The three-layer framework is bottlenecked by topology: the corridor is
double-track with no overtaking, so same-direction trains are a pure FIFO queue
and L3/L2 have no re-sequencing freedom (coding_log:2678, :2780, :3015). This is
why learned L3 ≈ per-transit SA and L2 ≈ heuristic. We add station-level
overtaking so a through train can pass a dwelling train at a station, and let L2
learn when to yield.

User-confirmed decisions:

1. Pass semantics: a train with `planned.arr == planned.dep` at an intermediate
   station is a **pass-through at that station** (global `pass_through: true`
   remains supported). No timetable edits.
2. Yield is learned by L2 by reusing the existing 6 `entry_choice` offsets
   (`plan+10/20/30/50`). No new action head, no rollout/MAPPO interface change.
3. Reward: **externality pricing** — charge the front/blocking train for the
   priority-weighted moving-block wait it imposes on followers. Mirrors
   `.claude/plans/plan_reward_externality.md`.
4. Through-platform: only P1 is `through_allowed` at A/B/C/E in
   `configs/five_station_15t/railway_resources.yaml`; D keeps both. Pass trains
   are masked to P1; stopping trains stay on P2–P4.
5. Scope: `five_station_15t` only. Do not edit other scenario configs.

Current facts that matter:

- `five_station_15t` has **no** `pass_through: true`; zero-dwell intermediate
  stops are G230@C, G238@B/C, G1487@B (G1185@C/D and G1616@E are off-route
  placeholders).
- `five_station_env.py:187-195` removes P1 from every non-pass train, so no
  overtaking is physically possible today. `DWELL_EXTEND_CHOICES=[0]`.
- Real overlap example: G1380_s1 (P1, stop) B arr=151/dep=170; G238 (P2, pass)
  B arr=155, zero dwell. G238 can depart B at 160, ahead of the P1 train.
- Grants are executed sorted by `W_out[0]` (`five_station_env.py:956`), so
  departure re-sequencing is already supported.
- L2 local reward charges only the yielder's own delay; the shared global term
  is unweighted and ~0.01–0.04, so yielding cannot be learned without a reward
  change (`reward.py:196-245`).
- Overtaking is only possible when both trains are co-pending in the same step
  (`_pending_trains(lookahead=120)`), because `_advance_trains` advances a train
  immediately on grant. This is a documented limitation, not a bug.

## Design

### 1. Per-station pass — one source of truth

In `src/experiment/scenario_loader.py`:

- Add `pass_stations: Set[str] = field(default_factory=set)` to `TrainTimetable`
  (end of field list, default keeps existing constructors valid).
- Add helpers:
  - `route_stations(train)` — same up/down origin→destination logic as
    `FiveStationEnv._get_train_route` (`five_station_env.py:244-256`); excludes
    off-route placeholder stops.
  - `pass_stations_for(train, stations)` — intermediate stations on the route
    where `abs(planned.arr - planned.dep) <= 1e-6`, `bounds.dwell_min == 0`,
    and the station is "mixed" (some but not all platforms `through_allowed`).
    The mixed gate activates the feature only for scenarios whose resource config
    reserves a through platform.
  - `is_station_pass(train, station_id)` — `train.pass_through or
    station_id in train.pass_stations`.
  - `apply_pass_platform_policy(train, stations)` — per station:
    pass train → candidates ∩ through platforms; stop train → candidates ∩
    non-through platforms when non-through exist; otherwise keep all. This
    replaces the hardcoded `'D'` exemption and the `if not train.pass_through`
    stripping loop. No station IDs hardcoded.
- Populate `train.pass_stations` at the end of `load_scenario` (after
  `_parse_resources`/`_parse_timetable`, before returning `ScenarioConfig`).
- In `FiveStationEnv.__init__`, replace lines 185-195 with a loop calling
  `pass_stations_for` (if empty) + `apply_pass_platform_policy`; do the same in
  `_build_extra_train` and remove its bespoke `if st != "D"` stripping.

### 2. Mask / schedule use per-station pass

In `src/control/action_mask.py`:

- Import `is_station_pass`.
- `derive_schedule`: replace `if train.pass_through:` with
  `if is_station_pass(train, station_id):` (pass branch: dwell 0, platform
  occupancy `THROUGH_PLATFORM_CLEARANCE_TIME`).
- `illegal_reason`: compute `is_pass` once; use it for
  `platform_not_through`, `passthrough_dwell`, and the `actual_dwell` formula.
- No changes to `ENTRY_CHOICES`, `DWELL_EXTEND_CHOICES`, `candidate_actions`,
  `action_to_request`, or the factorized action interface.

In `src/control/dispatch_bounds.py`:

- `for_train`: `pass_flag=is_station_pass(train, station_id)` so observation
  index 8 becomes per-station.

### 3. Observation (obs_dim 31 → 34)

In `src/env/five_station_env.py` `_obs_context`:

- Add `station_pending_detail`: for each station, entries
  `{tid, priority, direction, arrival (journey.arrival_at_current),
    planned_dep, is_pass}`. Keep existing `station_pending` for compatibility.

In `src/env/observation_builder.py`:

- `ACTOR_SELF_DIM` 16 → 19. Append 3 features after the 4 class one-hot dims:
  - `has_higher_prio_follower` (0/1): any same-direction pending train at this
    station with later arrival and lower priority number.
  - `prio_gap_score = max(0, self_priority - min_follower_priority) / 2`.
  - `follower_arrival_gap = clamp(min_follower_arrival - planned_dep, 0, 120) / 120`
    (1.0 when no higher-priority follower).
- Index 8 stays the pass bit but is now per-station via `DispatchBounds`.
- Update `compute_obs_dim` / docstrings. `DEFAULT_OBS_DIM` 31 → 34 in
  `src/marl/actor_critic.py`; training HP `obs_dim` comments updated. The local
  critic input becomes `(k_local+1)*34` automatically.

### 4. Externality reward (L2)

In `src/control/l3_sim.py`:

- Add `L3CorridorState.front_train_id(corridor_id)` returning
  `occs[-1].train_id` or `None`. This is exact for current physics:
  `front_time_at` uses `occs[-1]` exclusively, so a positive wait is caused by
  that occupant.
- Optional `L3Simulator.front_train_id` wrapper.

In `src/env/five_station_env.py` `_advance_trains`:

- Capture `front_id` before executing the branch (before recording the current
  transit), after `corridor_key`/`train_prio` are known.
- After `headway_delay` is known, accumulate:
  `self._leader_externality[front_id] += headway_delay * w_follower`
  (only when `headway_delay > 1e-6`, `front_id != train_id`).
  `w` = new `L2_PRIORITY_WEIGHT = {1: 2.0, 2: 1.0, 3: 0.5}` in `reward.py`.
- Enrich `_l3_transit_events` with `priority`, `front_train`, `dep_station`,
  `arr_station`, `is_pass_dep`.
- Add `_leader_externality` to reset/__init__; pass it to
  `compute_layered_rewards(externality_charges=...)`; pop charges for trains
  that received a reward this step. Pass to `compute_metrics` for diagnostics.

In `src/env/reward.py`:

- Add `L2_PRIORITY_WEIGHT`, `EXTERN_ALPHA=0.05`, `EXTERN_CAP_PER_STEP=20.0`.
- Use `L2_PRIORITY_WEIGHT` in the local reward (replace inline dict at line 212).
- `RewardBreakdown` gains `externality: float = 0.0`.
- `compute_layered_rewards(..., externality_charges=None)`:
  `w_i = L2_PRIORITY_WEIGHT[prio_i]`;
  `ext = min(EXTERN_ALPHA * ledger_i / max(w_i, 1e-6), CAP)`;
  `local -= ext`.
- Rationale: a P2 leader blocking a P1 follower with 8 s wait pays
  `0.05*8*(2/1)=0.8`; a 15 s yield costs ≈0.74 local reward (still inside the
  15 s hard on-time bonus), so yielding becomes rational. Sweep α ∈
  {0.03, 0.05, 0.08}. Cap and the 15 s bonus prevent excessive yielding.

### 5. Overtake/yield instrumentation

In `src/env/five_station_env.py`:

- Before `_advance_trains`, capture per-grant `(current_station, next_station,
  arrival, priority, pass, planned_dep, direction)`.
- `_detect_overtakes(effective_grants, pre_info)`: group by
  `(station, next_station, direction)`; event iff `arr_i < arr_j` and
  `W_out_i > W_out_j` (i arrived earlier, departed later). Tag pass/priority and
  `yield_delay = W_out_i - planned_dep_i`.
- Add an independent **yield-choice** counter: lower-priority train at a station
  with a higher-priority pending follower and an entry offset > `plan` (or
  `W_out - planned_dep > 5 s`). This catches yields that prevent an overtake;
  do not rely only on inversion events.
- Keep cumulative `overtake_total`, `yield_total`; expose per-step and totals in
  `info` (`overtake_events`, `overtake_count`, `yield_count`, ...).
- Change the `_advance_trains` sort to
  `(W_out[0], priority, train_id)` for deterministic tie-breaking.

### 6. Config

`configs/five_station_15t/railway_resources.yaml` only:

- A/B/C/E: `p1` `through_allowed: true`; `p2/p3/p4: false`.
- D: both `true` (capacity fallback).

This edit is the activation gate for per-station pass semantics.

### 7. Metrics / eval

- `Metrics`: add optional fields `pw_dep_delay`, `pw_arr_delay`,
  `overtake_count`, `yield_count`, `pass_overtake_count` (defaults keep
  construction valid). Add `priority` to transit events and `per_transit.csv`; add
  `overtakes`, `yields`, `pw_arr_delay`, `pw_dep_delay` to `per_episode.csv`; new
  `overtake.csv` in `--out-dir`.
- `scripts/eval_protocol.py` `run_episode`: accumulate env counters each step and
  return them. Baselines (FCFS/heuristic) use env-side counters, so they are
  comparable.
- Reporting: add overtake/yield counts and priority-weighted delays to
  `report_legacy` / `report_clean` / `report_incident`.

## Timing / physical analysis (why this works)

Example (real timetable, B station):

1. G1380_s1 (P1, stop) and G238 (P2, pass) are co-pending at B (arr 151 / 155,
   within the 120 s batch window).
2. Platform policy: G238 candidates `[B_p1]` (pass), G1380_s1 candidates
   `[B_p2..B_p4]` (stop) — no platform conflict; B throats are per-platform.
3. G238 pass schedule: entry 155, platform [155,160], `W_out=160`. G1380_s1
   stop schedule: entry ~165, `W_out=170`.
4. `_advance_trains` sorts by `W_out`: G238 enters B→C first. Overtake event:
   G1380_s1 arrived at B earlier but departed later.
5. If G238's departure would jam the higher-priority P1 train, the externality
   charge makes G238 choose `plan+10` (yield), giving up the pass. The policy
   learns the priority trade-off rather than passing unconditionally.

Limitations to state in docs/paper:

- Station is assumed to have a passing loop; the model treats station platform
  entry separately from the corridor, so a train can be scheduled at a station
  while another holds outside. This is an existing abstraction.
- Overtaking only occurs when both trains are co-pending (≤120 s apart).
- L3 remains FIFO per corridor; this feature adds L2 re-sequencing freedom, not
  in-corridor passing.

## Alignment check (coding-workflow)

- [x] No hardcoded station/train/platform IDs: pass platforms come from
      `through_allowed`; route from origin/destination + direction; P1 = first
      through platform in config.
- [x] Observation dim is config-derived (`get_obs_dim`); HP constants updated to
      34 for documentation.
- [x] Action mask enforces Layer 1 bounds and through-platform restriction;
      no new reward-based safety term.
- [x] Local critic input follows `(k_local+1)*obs_dim` automatically.
- [x] Fallback/retry still uses `is_legal_action` and rebuilds fresh requests.
- [x] Config-only expansion: another scenario can opt in by setting a mixed
      `through_allowed` layout; no Python changes.

## Implementation phases

**Phase 1 — physics + mask + instrumentation (obs dim still 31)**
1. `scenario_loader` helpers/field/population.
2. `action_mask`, `dispatch_bounds`, env candidate policy (`__init__`,
   `_build_extra_train`).
3. Config `through_allowed` edit.
4. `front_train_id`, externality ledger, overtake/yield detection, info fields,
   metrics.
5. Tests: pass detection, P1 masking, overtake at B, no bound/resource
   violations. Run existing tests + FCFS/heuristic smoke (old ckpts still load).

**Phase 2 — observation + reward**
6. 3 follower features, obs dim 34, DEFAULT_OBS_DIM, reward charges.
7. Unit tests: obs dims/features, externality attribution/charge math.
   Old L2 checkpoints now fail (add a clear obs-dim guard).

**Phase 3 — retrain L2**
8. `python -m src.experiment.run_train_hierarchical --env five_station_15t
   --phases A --total-episodes <2000 smoke / 7000 paper> --incident ...`
   (Phase A = L2 + SA-L3). Optionally Phase B if L3 needs adaptation.
9. Log all checkpoint/CSV paths in `coding_log` per CLAUDE.md.

**Phase 4 — eval**
10. Held-out learned vs FCFS/CRUISE/SA; report overtakes, yields,
    priority-weighted delays, existing ontime/tail/DDT.
11. Sweep `EXTERN_ALPHA` if needed; verify no excessive late-running.

## Verification

```bash
python -m pytest tests/test_phase12.py tests/test_l3_physics.py -q
python -m pytest tests/test_overtaking.py -q          # new
python scripts/eval_protocol.py --scenario five_station_15t --l2 <l2.pt> \
    --fcfs-only --n 20 --blocks 1 --incident
```

Pass criteria:

- G230/G238/G1487 are pass-only at C/B/B respectively; G1185 placeholders are not.
- Pass trains always get P1 at A/B/C/E; stop trains never do.
- At least one deterministic overtake at B (G1380_s1 / G238) with no
  `postcheck_reject`, no hard-guard violation, and no resource-window overlap.
- Externality charge appears for the blocking leader and is consumed once.
- After retraining, learned L2 can choose delayed entry (yield) when a
  higher-priority follower is present; overtake/yield counts and
  priority-weighted delay are exported.

## Critical files

- `src/experiment/scenario_loader.py`
- `src/control/action_mask.py`
- `src/control/dispatch_bounds.py`
- `src/control/l3_sim.py`
- `src/env/five_station_env.py`
- `src/env/observation_builder.py`
- `src/env/reward.py`
- `configs/five_station_15t/railway_resources.yaml`
- `scripts/eval_protocol.py`
- `tests/test_phase12.py`, `tests/test_l3_physics.py`, new `tests/test_overtaking.py`

## Risks

- **Physical abstraction**: station passing-loop/approach capacity is not modeled;
  document and keep the same-step co-pending limitation.
- **Checkpoint breakage**: obs dim 31→34 invalidates all L2/flat checkpoints;
  retraining required. Add explicit guard.
- **Reward tuning**: externality α and cap need a sweep; monitor double counting
  (follower already pays headway delay) and excessive yielding.
- **Resolver-driven yields**: C up shares P1/P2 throats; tag yield events with
  `retried`/`fallback_from` to distinguish policy-driven from resolver-driven.
- **CSV schema additions**: check plotting consumers; prefer the new
  `overtake.csv` for new counts.
