# Railway MARL — Implementation Specification

> Last updated: 2026-05-24
> Status: Phase 0 / 1 / 2 DONE. Next: Phase 3 (SUMO env + disturbance loop).

---

## RESEARCH QUESTION

When random delays break the nominal conflict-free timetable, can the
three-layer framework recover better than flat MAPPO and rule-based
rescheduling, while respecting hard Layer 1 dispatch boundaries?

Layer 1 is rule-based and fixed. Layer 2 is the only learned layer.
Layer 3 is rule-based. Layer 2 activates only when a disturbance is detected.

---

## NETWORK (scaled for training)

```
[A_throat_in] -- [A_p1 / A_p2 / A_p3] -- [A_throat_out]
                                                ||
                                          up_main  (A→B)
                                         down_main (B→A)
                                                ||
[B_throat_in] -- [B_p1 / B_p2 / B_p3] -- [B_throat_out]
```

Physical dimensions (training scale):

| element    | length  | speed limit |
|------------|---------|-------------|
| corridor   | 1200 m  | 30 m/s      |
| platform   | 250 m   | 15 m/s      |
| throat     | ~80 m   | 20 m/s      |
| approach   | —       | 30 m/s      |

Each station has 3 platforms, one inbound throat, one outbound throat.
Each throat has 2 conflicting switch routes.

---

## TRAINS (8 + 1 conflict-test train)

All times in simulation seconds (compressed scale).

### A→B (up direction)

| id    | class | depart | arr_plan A | dep_plan A | arr_plan B | dep_plan B | priority | notes           |
|-------|-------|--------|-----------|-----------|-----------|-----------|----------|-----------------|
| K641  | K     | 0      | -120      | 0         | 90        | 120       | 3        | dwell_min A=120s|
| D133  | D     | 120    | 60        | 120       | 210       | 240       | 1        |                 |
| G5050 | G     | 180    | 180       | 180       | 270       | 270       | 1        | pass-through    |
| K667  | K     | 300    | 240       | 300       | 390       | 420       | 3        |                 |

### B→A (down direction)

| id    | class | depart | arr_plan B | dep_plan B | arr_plan A | dep_plan A | priority |
|-------|-------|--------|-----------|-----------|-----------|-----------|----------|
| G1408 | G     | 90     | 60        | 90        | 180       | 240       | 1        |
| G5060 | G     | 240    | 210       | 240       | 330       | 390       | 1        |
| Z386  | Z     | 360    | 330       | 360       | 450       | 510       | 2        |
| K1666 | K     | 480    | 450       | 480       | 570       | 630       | 3        |

### Conflict-test train (disable by commenting in config)

| id   | class | direction | depart | notes                          |
|------|-------|-----------|--------|--------------------------------|
| K999 | K     | A→B       | 120    | conflicts D133, verifies fallback; remove before final experiments |

### Time scale reference

| parameter      | real value | training value |
|----------------|-----------|----------------|
| corridor runtime | 360 s   | 90 s           |
| A dwell default  | 600 s   | 60 s           |
| B dwell default  | 240 s   | 30 s           |
| K641 dwell at A  | 1200 s  | 120 s          |
| headway          | 180 s   | 60 s           |
| dep_max_slack    | 1200 s  | 120 s          |
| episode length   | 40000 s | < 900 s        |

---

## LAYER 1 — DISPATCH BOUNDS (rule-based, fixed)

Reads `timetable_layer1.yaml`, produces one `TrainBounds` per train per station.
Hard envelopes: violations prevented by action masking, never in reward.

```python
@dataclass
class TrainBounds:
    train_id:           str
    station_id:         str
    t_arr_plan:         float
    t_dep_plan:         float
    t_arr_min:          float   # = t_arr_plan
    t_dep_max:          float   # = t_dep_plan + dep_max_slack
    dwell_min:          float
    P_i:                list    # candidate platform IDs
    prio_i:             int
    pass_flag:          bool
    seg_runtime_bounds: dict    # {segment_id: (min_sec, max_sec)}
```

Verified Layer 1 bounds (simulation seconds):

```
K641  A: arr_min=-120  dep_max=240  dwell_min=120
      B: arr_min=90    dep_max=240  dwell_min=30
D133  A: arr_min=60    dep_max=240  dwell_min=60
      B: arr_min=210   dep_max=360  dwell_min=30
G5050 A: arr_min=180   dep_max=300  dwell_min=0   pass_flag=True
      B: arr_min=270   dep_max=390  dwell_min=0   pass_flag=True
K667  A: arr_min=240   dep_max=420  dwell_min=60
      B: arr_min=390   dep_max=540  dwell_min=30
G1408 B: arr_min=60    dep_max=210  dwell_min=30
      A: arr_min=180   dep_max=360  dwell_min=60
G5060 B: arr_min=210   dep_max=360  dwell_min=30
      A: arr_min=330   dep_max=510  dwell_min=60
Z386  B: arr_min=330   dep_max=480  dwell_min=30
      A: arr_min=450   dep_max=630  dwell_min=60
K1666 B: arr_min=450   dep_max=600  dwell_min=30
      A: arr_min=570   dep_max=750  dwell_min=60
```

---

## LAYER 2 — LOCAL COMPETITION (learned, MAPPO)

### Disturbance and trigger

Layer 2 is idle during nominal operations. Activates on disturbance detection.

```python
TRIGGER_THRESHOLD_SEC = 30   # tune in ablation

class DisturbanceInjector:
    def __init__(self, prob_per_step=0.005, delay_range=(30, 180)):
        self.prob_per_step = prob_per_step
        self.delay_range   = delay_range

    def step(self, trains, t_now):
        for train in trains:
            if random.random() < self.prob_per_step:
                delta = random.uniform(*self.delay_range)
                train.apply_delay(delta)

def layer2_trigger(train, bounds: TrainBounds, t_now: float,
                   conflict_graph) -> bool:
    predicted_arr = t_now + train.travel_time_to_throat()
    if abs(predicted_arr - bounds.t_arr_plan) > TRIGGER_THRESHOLD_SEC:
        return True
    if conflict_graph.has_active_conflict(train.id):
        return True
    return False
```

### Observation (fixed 74-dim)

```
obs_i = [self_state(13) | neighbor_state×5(40) | validity_mask(5) | resource_state(16)]
```

| block             | dim | fields |
|-------------------|-----|--------|
| self_state        | 13  | t_now, pos, speed, delay_i, t_arr_plan, t_dep_plan, t_arr_min, t_dep_max, dwell_min, prio_i, pass_flag, n_candidate_platforms, delta_from_plan |
| neighbor_state ×5 | 40  | t_arr_expected, dwell_expected, t_dep_expected, n_candidate_platforms, prio, delay, pass_flag, same_platform_flag |
| validity_mask     | 5   | 1=real neighbor, 0=padded |
| resource_state    | 16  | platform_occupied[A_p1..A_p3, B_p1..B_p3], throat_in_busy×2, throat_out_busy×2, turnout_bits×4, headway_gap_fwd, headway_gap_bwd |

Neighbors: K_max=5, sorted by `|t_arr_expected_j − t_arr_expected_i|`. Pad with zeros.

### Action space (max 60 combinations)

```
a_i = (platform_choice, entry_choice, dwell_extend)
```

| sub-action      | choices                                            |
|-----------------|----------------------------------------------------|
| platform_choice | elements of P_i                                    |
| entry_choice    | {earliest, plan, plan+1, plan+2, plan+4} (minutes) |
| dwell_extend    | {0, +1, +2} minutes; forced 0 for pass-through     |

### Action masking

```python
ENTRY_CHOICES      = ["earliest", "plan", "plan+1", "plan+2", "plan+4"]
DWELL_EXTEND_SECS  = [0, 60, 120]
THROAT_CLR         = 20    # seconds
THROUGH_CLR        = 20    # seconds for pass-through

def requested_entry_time(train_id, entry_choice, bounds, t_now, travel_time):
    physical_earliest = t_now + travel_time
    hard_floor        = max(physical_earliest, bounds.t_arr_min)
    if entry_choice == "earliest":
        return hard_floor
    offsets = {"plan": 0, "plan+1": 60, "plan+2": 120, "plan+4": 240}
    return max(bounds.t_arr_plan + offsets[entry_choice], hard_floor)

def derive_schedule(train_id, action, bounds, t_now, travel_time):
    platform, entry_choice, dwell_extend_sec = action
    t_in_s  = requested_entry_time(train_id, entry_choice, bounds, t_now, travel_time)
    t_in_e  = t_in_s + THROAT_CLR
    if bounds.pass_flag:
        t_pl_s = t_in_e
        t_pl_e = t_pl_s + THROUGH_CLR
    else:
        t_pl_s = t_in_e
        t_pl_e = t_pl_s + bounds.dwell_min + dwell_extend_sec
    t_out_s = t_pl_e
    t_out_e = t_out_s + THROAT_CLR
    return t_in_s, t_in_e, t_pl_s, t_pl_e, t_out_s, t_out_e

def is_legal(train_id, action, bounds, t_now, travel_time, resource_cfg):
    platform, entry_choice, dwell_extend_sec = action
    if platform not in bounds.P_i:                                   return False
    if bounds.pass_flag and not resource_cfg.through_allowed(platform): return False
    if bounds.pass_flag and dwell_extend_sec != 0:                   return False
    t_in_s, _, _, _, t_out_s, _ = derive_schedule(
        train_id, action, bounds, t_now, travel_time)
    if t_in_s < bounds.t_arr_min:  return False
    if t_out_s > bounds.t_dep_max: return False
    if not kinematically_feasible(train_id, t_in_s, t_out_s): return False
    return True

def compute_mask(train_id, bounds, t_now, travel_time, resource_cfg):
    actions = [(p, e, d)
               for p in bounds.P_i
               for e in ENTRY_CHOICES
               for d in DWELL_EXTEND_SECS]
    mask    = [is_legal(train_id, a, bounds, t_now, travel_time, resource_cfg)
               for a in actions]
    status  = "OK" if any(mask) else "NO_LEGAL_ACTION"
    return actions, mask, status
    # NO_LEGAL_ACTION → hold train, replan next step, do not sample policy
```

### Competition resolver

Runs deterministically after all agents submit actions.
**Design decision: corridor headway is NOT a resolver resource.**
Minimum tracking headway on up_main / down_main is enforced by
Layer 3 execution only. This avoids artificial cross-direction conflicts
at the resolver level.

```python
@dataclass
class WindowRequest:
    train_id: str
    platform: str
    W_in:     tuple   # (t_start, t_end) for inbound throat
    W_plat:   tuple   # (t_start, t_end) for platform
    W_out:    tuple   # (t_start, t_end) for outbound throat
    priority: int
    delay:    float
    # _resource_windows: {resource_id: window} — station resources only

def action_to_request(train_id, action, bounds, t_now, travel_time):
    times = derive_schedule(train_id, action, bounds, t_now, travel_time)
    t_in_s, t_in_e, t_pl_s, t_pl_e, t_out_s, t_out_e = times
    return WindowRequest(
        train_id, action[0],
        (t_in_s, t_in_e), (t_pl_s, t_pl_e), (t_out_s, t_out_e),
        bounds.prio_i, current_delay(train_id))

def windows_overlap(w1, w2, gap=20):
    return w1[0] < w2[1] + gap and w2[0] < w1[1] + gap

def requests_conflict(r1, r2, route_cfg):
    if r1.platform == r2.platform and windows_overlap(r1.W_plat, r2.W_plat):
        return True
    if route_cfg.throat_in_conflict(r1.train_id, r2.train_id) \
            and windows_overlap(r1.W_in, r2.W_in):
        return True
    if route_cfg.throat_out_conflict(r1.train_id, r2.train_id) \
            and windows_overlap(r1.W_out, r2.W_out):
        return True
    return False

def resolve(r1, r2):
    """Returns (winner, loser). Lower priority number wins.
    Tie → higher delay wins. Tie → lower train_id (lexicographic)."""
    key = lambda r: (r.priority, -r.delay, r.train_id)
    return (r1, r2) if key(r1) <= key(r2) else (r2, r1)

def assign_fallback(loser, occupied, bounds, rebuild_fn):
    # rebuild_fn(platform, shift) → WindowRequest with correct resource IDs
    # tries other platforms first, then delays on same platform
    for platform in bounds.P_i[loser.train_id]:
        if platform == loser.platform:
            continue
        slot = find_earliest_feasible(platform, loser, occupied, bounds, rebuild_fn)
        if slot is not None:
            return slot
    return delay_on_same_platform(loser, occupied, bounds, rebuild_fn)
    # Both helpers must reuse is_legal to guarantee no hard-bound violation
```

### Local critic

```python
K_MAX = 5   # tune in ablation: {3, 5, 7}

def get_local_group(agent_id, conflict_graph):
    neighbors = sorted(conflict_graph.neighbors(agent_id),
                       key=lambda j: time_overlap_score(agent_id, j))
    return [agent_id] + neighbors[:K_MAX]

def critic_input(agent_id, conflict_graph, observations):
    group    = get_local_group(agent_id, conflict_graph)
    obs_list = [observations[j] for j in group]
    while len(obs_list) < K_MAX + 1:
        obs_list.append(np.zeros(74))
    return np.concatenate(obs_list)   # always (K_MAX+1)*74 = 444-dim
```

### Reward

```
r_i = r_individual + λ_team × r_team        (λ_team start = 0.3)

r_individual = − w1 × wait_i
               − w2 × delay_i × prio_weight(prio_i)
               + w3 × completion_i
               − w4 × excessive_hold_i

r_team       = − w5 × (n_unresolved_conflicts / N_active)
               − w6 × deadlock_flag
               − w7 × (total_system_delay / N_active)
```

| term | definition |
|------|-----------|
| wait_i | seconds waiting outside allocated resource |
| delay_i | max(0, t_actual_dep − t_dep_plan) |
| completion_i | +1 on successful station passage |
| excessive_hold_i | max(0, actual_dwell − dwell_min − 60) |
| prio_weight | 2.0 / 1.0 / 0.5 for priority 1 / 2 / 3 |
| n_unresolved_conflicts | overlapping windows after resolver |
| deadlock_flag | 1 if any group has zero progress ≥ 10 steps |

No hard-boundary violation terms in reward. All enforced by masking.

---

## LAYER 3 — EXECUTION (rule-based, fixed)

### Interface from Layer 2

```python
@dataclass
class ExecutionTarget:
    train_id:             str
    allocated_platform:   str
    route_segments:       list    # authorized SUMO edge IDs in order
    t_throat_entry:       float
    t_platform_arrival:   float
    t_platform_departure: float
    t_throat_exit:        float
    v_at_throat_entry:    float
    v_at_platform:        float   # 0.0 for stopping trains
```

### Rule-based executor

```python
class RuleBasedExecutor:
    def step(self, train, target: ExecutionTarget, t_now: float) -> float:
        phase = self._phase(train, target)
        if phase == "approach_throat":
            return self._speed_for_arrival(
                train, target.t_throat_entry, target.v_at_throat_entry, t_now)
        elif phase == "throat_to_platform":
            return min(train.speed, THROAT_MAX_SPEED)
        elif phase == "braking_to_stop":
            return self._braking_profile(train, target.t_platform_arrival, t_now)
        elif phase == "dwell":
            self._door_control(train, "open")
            if t_now >= target.t_platform_departure:
                self._door_control(train, "close")
                return DEPARTURE_SPEED
            return 0.0
        elif phase == "departure_throat":
            return self._speed_for_exit(train, target.t_throat_exit, t_now)
        elif phase == "corridor":
            return self._headway_safe_speed(train)

    def _speed_for_arrival(self, train, t_target, v_target, t_now):
        d = train.distance_to_throat_entry()
        t = t_target - t_now
        if t <= 0:
            return v_target
        return min(d / t, train.v_max, kinematic_max(train.speed, d, v_target))

    def _headway_safe_speed(self, train):
        gap = train.gap_to_leading_train()
        if gap < HEADWAY_CRITICAL: return 0.0
        if gap < HEADWAY_WARNING:  return train.speed * (gap / HEADWAY_WARNING)
        return train.v_max_segment
```

### Re-trigger

```python
REPLAN_THRESHOLD_SEC = 30

def needs_replan(train, target: ExecutionTarget, t_now: float) -> bool:
    slack = target.t_throat_entry - t_now - train.travel_time_to_throat()
    return slack < -REPLAN_THRESHOLD_SEC
```

---

## CONSTRAINTS SUMMARY

| constraint                      | enforced by              |
|---------------------------------|--------------------------|
| platform single-occupancy       | action mask + resolver   |
| conflicting throat routes       | resolver                 |
| minimum dwell time              | action mask              |
| t_arr_min / t_dep_max           | action mask              |
| minimum corridor headway        | Layer 3 only (not resolver) |
| platform direction compatibility| action mask              |
| route authorization             | Layer 3 route check      |

---

## ENVIRONMENT LIFECYCLE (per decision step)

```
1.  Pull train states from SUMO via TraCI
2.  Run DisturbanceInjector.step()
3.  Check layer2_trigger for each active train
4.  For triggered trains: build 74-dim observations
5.  Build joint action masks (compute_mask)
6.  status == NO_LEGAL_ACTION → hold train, skip policy
7.  Sample legal actions from MAPPO policy
8.  Convert actions → WindowRequests (action_to_request + derive_schedule)
9.  Run deterministic resolver; assign_fallback to losers
10. Convert granted windows → ExecutionTargets
11. Check needs_replan for all active targets
12. Apply RuleBasedExecutor commands to SUMO via TraCI
13. Advance SUMO by decision_interval_sec (= 10 s)
14. Compute reward and metrics
```

---

## PROJECT STRUCTURE

```
project/
  configs/
    railway_resources.yaml
    timetable_layer1.yaml       ← actual filename in use
    scenarios/
      base.yaml
      dense.yaml
      disturbance_light.yaml
      disturbance_heavy.yaml
  sumo/
    net/two_station.net.xml     ← regenerate via netconvert after edg/nod changes
    two_station.sumocfg
    routes/base.rou.xml
    output/                     ← gitignored
  src/
    env/
      sumo_env.py               ← Gym-style env, owns lifecycle steps 1-14  [TODO]
      observation_builder.py    ← 74-dim obs, K_max padding                 [TODO]
      reward.py                 ← hybrid individual + team reward            [TODO]
      disturbance.py            ← DisturbanceInjector, layer2_trigger        [TODO]
    control/
      dispatch_bounds.py        ← DONE
      resource_manager.py       ← DONE
      action_mask.py            ← DONE
      resolver.py               ← (part of resource_manager.py)
      executor.py               ← RuleBasedExecutor, ExecutionTarget         [TODO]
    marl/
      mappo.py                  ← PPO update loop                            [TODO]
      actor_critic.py           ← shared actor (74→hidden), local critic (444→1) [TODO]
      rollout_buffer.py         ← [TODO]
  scripts/
    validate_sumo.py            ← DONE
  src/experiment/
    scenario_loader.py          ← DONE
    run_heuristic.py            ← DONE
    run_train.py                ← [TODO]
    run_eval.py                 ← [TODO]
    metrics.py                  ← [TODO]
  tests/
    test_phase12.py             ← DONE (10 tests)
```

---

## HYPERPARAMETERS

| parameter         | start value | ablation range  |
|-------------------|-------------|-----------------|
| decision_interval | 10 s        | fixed           |
| K_max             | 5           | {3, 5, 7}       |
| λ_team            | 0.3         | {0.1, 0.3, 0.5} |
| REPLAN_THRESHOLD  | 30 s        | {15, 30, 60}    |
| prio_weight high  | 2.0         | {1.5, 2.0, 3.0} |
| rollout_steps     | 1024        |                 |
| PPO epochs        | 5           |                 |
| minibatch_size    | 256         |                 |
| gamma             | 0.99        |                 |
| gae_lambda        | 0.95        |                 |
| clip_ratio        | 0.2         |                 |
| entropy_coef      | 0.01        |                 |
| learning_rate     | 3e-4        |                 |

---

## IMPLEMENTATION PHASES

### ✅ Phase 0 — SUMO sanity
- two_station.sumocfg, validate_sumo.py
- All 8 trains complete routes, platform logs valid

### ✅ Phase 1 — Scenario loader
- scenario_loader.py: parses yaml + rou.xml, validates IDs and bounds
- dispatch_bounds.py: TrainBounds per train per station

### ✅ Phase 2 — Resource manager + heuristic baseline
- resource_manager.py: ResourceCalendar, WindowRequest, conflict graph,
  resolve_pair, assign_fallback (with RebuildFn callback), resolve_all, GrantResult
- action_mask.py: derive_schedule, is_legal, compute_mask, action_to_request,
  heuristic_action
- run_heuristic.py: full schedule_station loop, verified on all 8+1 trains
- test_phase12.py: 10 tests passing

### 🔲 Phase 3 — SUMO TraCI env + disturbance loop
Build in this order:
1. `executor.py` — ExecutionTarget, RuleBasedExecutor, needs_replan
2. `disturbance.py` — DisturbanceInjector, layer2_trigger
3. `sumo_env.py` — Gym env wrapping lifecycle steps 1-14, TraCI integration
4. `observation_builder.py` — 74-dim fixed obs with K_max padding
5. Pass criterion: inject disturbances, verify Layer 2 triggers, verify
   ExecutionTarget produced, verify needs_replan causes re-allocation

### 🔲 Phase 4 — MAPPO integration
1. `rollout_buffer.py` — stores (obs, action, mask, reward, value, logprob)
2. `actor_critic.py` — shared actor, local critic (444-dim input)
3. `mappo.py` — PPO update with action masks applied to logits
4. `run_train.py`
5. Pass criterion: mean delay improves over heuristic after convergence

### 🔲 Phase 5 — Baseline comparison
Baselines: (1) rule-based FCFS, (2) flat MAPPO global critic, (3) IPPO
Metrics: mean delay, max delay, mean wait, conflict count, deadlock count,
completion rate, inference time, critic input dimension
Report mean ± std over 5 seeds.

### 🔲 Phase 6 — Ablation
K_max ∈ {3,5,7}, λ_team ∈ {0.1,0.3,0.5}, with/without resolver,
with/without local critic, with/without disturbance injection.

---

## SCALABILITY RULES (code must satisfy all 10)

1. Train IDs come from config, never hard-coded in Python
2. Platform and resource IDs come from railway_resources.yaml
3. Conflict graph built from resource windows, no station-specific branches
4. Observation always 74-dim with padding and validity mask
5. Actor is shared across all trains
6. Critic input always (K_max+1)×74 = 444-dim
7. Resource calendar keyed by resource_id, not station name
8. Route compatibility stored in config, not Python conditionals
9. New station = new net.xml + resource yaml + timetable yaml only; no code change
10. Reward formula uses aggregate IDs, no train-specific branches

---

## ACCEPTANCE CHECKLIST

- [ ] Adding a train: timetable + rou.xml only, no code change
- [ ] Adding a station: net.xml + resource yaml + timetable only, no code change
- [ ] No Python file contains a station name or train ID as a literal string
- [ ] Every policy action passes through compute_mask before sampling
- [ ] Every fallback window passes through is_legal before granting
- [ ] Every granted window is logged and replayable
- [ ] Evaluation runs on a scenario not used during training
- [ ] All metrics exported to CSV/JSON per seed
