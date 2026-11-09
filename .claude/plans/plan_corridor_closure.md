# Plan: corridor closure disturbance (区间中断) to create catch-up/overtaking

Date: 2026-11-09
Status: implementation done; stratification done; 2k training launched

## Context

The current disturbance is a corridor speed restriction (slow everyone). It does
not create a fast train catching a slow one, and planned overtake overlaps
dissolve under realized moving-block delay. Goal: a temporary *closure* of a
corridor to new entries. Holding a lower-priority leader just before a
higher-priority follower's planned departure forces a genuine catch-up, and the
leader must choose to yield (or not) after reopening — the learnable coordination
case.

Scope: five_station_15t (config only; code is scenario-generic).

## What

### 1. Disturbance generator (`src/env/disturbance.py`)
- `inject_corridor_closure(cfg, rng, severity_scale=1.0) -> Dict[str, (t0,t1)]`.
- Not a random window: search *designed meets* per corridor — lower-priority
  leader with planned departure `d_L`, higher-priority follower with planned
  departure `d_F > d_L`; hold the leader for 10–30s starting just before `d_L`,
  ending before the follower's plan entry (`d_F - dwell`, so the follower is not
  held). Returns `{}` when no such pair exists. The short window isolates the
  meet; a long window holds many trains and the priority tie-break does the
  reordering instead of a learned yield.
- `cfg.time_generation["corridor_closure"]`:
  `{enabled: false, prob: 0.6, duration_range: [10,30], min_gap_sec: 15,
    follower_margin_sec: 5}`.

### 2. Environment (`src/env/five_station_env.py`)
- `enable_corridor_closure` ctor flag (None -> config `enabled`),
  `_corridor_closures: Dict[corridor_id, (t0,t1)]` generated at reset.
- `_closure_hold_until(train_id, station_id)` -> t1 when the train's next
  corridor is closed and the train is affected (t_now >= t0 or planned dep >=
  t0). Trains planned before the closure may still depart early.
- `_get_effective_bounds`: raise `arr_min` to `t1 - dwell` and extend `dep_max`
  past `t1` for held trains. Hard constraint via mask, no reward term.
- `_detect_resequencing`: tag events with `closure_related`.
- `info`: `corridor_closures`, `closure_held` (train -> hold seconds).
- Semantics documented: closure blocks NEW ENTRIES; trains already inside the
  corridor continue (no mid-section stop).

### 3. Observation (34 -> 37 dims)
Actor self-state gains 3 features (closure is new exogenous POMDP state; the
bounds clamp keeps all actions legal, so the mask cannot convey it):
- `next_corridor_closed` (0/1)
- `closure_remaining / 300` (clamped 0..300)
- `(closure_until - planned_dep) / 120` (clamped -1..1; how far past plan held)
`ACTOR_SELF_DIM 19 -> 22`, `DEFAULT_OBS_DIM 34 -> 37`. L2 checkpoints (incl. v7)
become incompatible — expected; retrain for the new disturbance.

### 4. Entry points
- `run_train_hierarchical.py`: `--closure`; thread to `env_factory`,
  `eval_hier_heldout`, and initial-delay gating.
- `scripts/eval_protocol.py`: `--closure`; thread to `run_episode`/env.
- No L3 obs change (L3 never sees a closed corridor; entry is blocked upstream).

### 5. Tests (`tests/test_closure.py`)
- Generator picks a valid meet (t1 between leader dep and follower plan entry).
- Effective bounds/mask: leader's earliest departure >= t1; follower's plan
  entry unaffected.
- Obs dim = 37 and closure features are nonzero while a closure is active.
- 2 closure episodes: success, no deadlock, no hard-guard violations.

## Why

Without a forcing event that puts a higher-priority follower close behind a
held lower-priority leader, the learned yield has almost no activation
opportunities (observed: 3 intermediate planned overtake pairs in 15 trains,
only one reliably activates). A closure creates those co-pending sets on demand
and is more interpretable than a speed restriction.

## Alignment check
- [x] No hardcoded IDs: corridor/station/train come from config + route graph.
- [x] Observation config-derived: obs_dim recomputed; 3 features documented.
- [x] Hard boundaries via mask: closure raises effective bounds; no reward term.
- [x] Local critic: `(k_local+1)*obs_dim` follows automatically.
- [x] Fallback re-verifies bounds: unchanged path; effective bounds used.
- [x] Config-only scenario expansion: new time_generation block; code generic.

### 6b. Higher policy coverage (`relevant_plus`)
The base relevance set controls only ~24% of decisions, so the training reward
is mostly the heuristic's and stays flat.  Add a `relevant_plus` coverage set:
base + (a) any train whose next corridor is closed, (b) any train sharing a
station/direction with another pending train (any priority) in the batch
window.  Measured coverage on closure episodes: ~84–88% (base ~20–26%).  Env
gets `control_coverage={"base","plus","full"}`; training/eval select it via
control mode.  Compare reward responsiveness and 180-ep paired eval against the
base-coverage run.

### 6. Affected / delay-dissipation stratification
- Transit events gain `closure_affected` (train was held at departure this step),
  `closure_affected_plan` (planned departure inside [t0,t1], policy-independent)
  and `closure_hold_sec`.  The existing `affected` / `affected_plan` fields are
  OR-ed with these, so the speed-restriction and closure arms share one schema.
- `eval_protocol.run_episode`: anchors DDT at the disturbance END — max of
  incident window ends and closure t1; episodes with no disturbance have
  `t_rec_start=None` and are excluded from DDT.
- `report_incident` now treats an episode as disturbed when it has either an
  incident or a closure; `affected.csv`, `tail.csv`, `episode.csv` and
  `recovery.csv` are written for `--closure` too.
- Held-out eval inside training applies the same relevance filter as training
  (`relevant=(control_mode=="relevant")`), so the monitor is on-distribution.

### 6c. BC warm start (if plus coverage fails)
`relevant_plus` (85%) and `relevant_plus_ramp` both end worse than the heuristic
once `p=1.0`: the policy has to learn routine scheduling and cannot match the
rule within 2k.  BC fallback: collect `(obs, platform_idx, entry_idx)` labels
from heuristic rollouts over the plus set, train the actor with masked
cross-entropy, then PPO fine-tune from that checkpoint at plus coverage.

## Pass criteria
- pytest: existing suite + new closure tests pass.
- A forced closure smoke on five_station_15t yields >=1 overtake event tagged
  `closure_related` in a small episode set, with no deadlock / no hard guard
  violation.
- The held leader's W_out >= t1; the follower's planned departure is unaffected
  when t1 < follower plan entry.

## Risks / limitations
- Entries only; trains already inside clear the corridor (no mid-section stop).
- Closure is known from reset (forecast), consistent with current incidents.
- If the closure end lands after both trains' planned departures, the existing
  `(W_out, priority, id)` tie-break reorders by rule and the learned yield has
  no effect; the generator avoids this by requiring `t1 < follower plan entry`.
- v7 L2 checkpoints (obs 34) are unloadable; old eval artifacts stay as a
  previous-version baseline.
