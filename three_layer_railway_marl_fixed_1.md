# Three-Layer Railway MARL Design — Revised and Fixed

> This document supersedes the previous GPT draft.
> All seven identified issues have been resolved with concrete, implementable solutions.
> Sections that were correct in the original are retained. Fixed sections are marked.

---

## 1. Purpose

This document defines the formal design for a hierarchical MAPPO experiment on a
two-station-one-corridor minimum railway network.

The main research question:

> Can a three-layer framework with dispatch boundaries, local resource competition,
> and rule-based execution scale better than flat MAPPO while maintaining low delay,
> low waiting time, and low conflict rate?

---

## 2. Minimum Network

### 2.1 Scenario

- Station A and Station B connected by a double-track corridor.
- Up direction: A -> B, initial trains are G5050, D133, K641, and K667.
- Down direction: B -> A, initial trains are Z386, G1408, G5060, and K1666.
- Each station has 3 platform tracks.
- Departure headways are close to the minimum tracking headway.
- Current initial demand has eight trains; G5050 is marked pass-through because it is specified as non-stop.

### 2.2 Physical Layout (SUMO)

```text
[A throat_in] -- [A platforms x3] -- [A throat_out]
                                      ||
                                   up_main
                                  down_main
                                      ||
[B throat_in] -- [B platforms x3] -- [B throat_out]
```

Each station has:
- An inbound throat with configurable conflicting switch routes
- 3 platform tracks, each with a designated stopping zone
- An outbound throat with configurable conflicting switch routes

---

## 3. Three-Layer Framework

### 3.1 Layer 1 — Dispatch Center (rule-based, fixed)

Provides hard envelopes. Does not issue step-level commands.

**Output per train:**

| Field | Type | Description |
|-------|------|-------------|
| `t_arr_plan` | float | Planned arrival time |
| `t_dep_plan` | float | Planned departure time |
| `t_arr_min` | float | Earliest allowed arrival (hard floor) |
| `t_dep_max` | float | Latest allowed departure (hard ceiling) |
| `dwell_min` | float | Minimum dwell time |
| `P_i` | set | Candidate platforms |
| `prio_i` | int | Priority class (1 = highest) |
| `pass_flag` | bool | Pass-through or stopping |
| `seg_runtime_bounds` | dict | Min/max runtime per segment |

**[FIX 6] Hard boundaries are enforced by action masking, not by reward penalty.**
There is no `boundary_violation_i` term in the reward.
Any action violating a Layer 1 bound is removed from the valid action set before
the policy network is queried. The policy never sees an illegal action.

#### 3.1.1 Initial Layer 1 input for the two-station experiment

The initial experiment uses eight trains on the A-B double-track corridor. This
data should live in `timetable.yaml`, not in Python code.

| Direction | Train | Plan | Class | Priority | Station service rule |
|-----------|-------|------|-------|----------|----------------------|
| A -> B | G5050 | 07:06-08:36 | G | 1 | pass-through / non-stop |
| A -> B | D133 | 06:48-09:24 | D | 1 | normal passenger service |
| A -> B | K641 | 05:38-08:42 | K | 3 | must stop at A, minimum 20 min |
| A -> B | K667 | 07:44-10:20 | K | 3 | normal passenger service |
| B -> A | Z386 | 08:29-10:40 | Z | 2 | normal passenger service |
| B -> A | G1408 | 07:40-08:45 | G | 1 | normal passenger service |
| B -> A | G5060 | 08:02-09:31 | G | 1 | normal passenger service |
| B -> A | K1666 | 09:03-11:47 | K | 3 | normal passenger service |

Priority mapping:

```text
priority 1: G, D, C
priority 2: Z, T
priority 3: K
```

Station dwell rules:

```text
A station passenger dwell minimum: 10 min
B station passenger dwell minimum: 4 min
K641 at A: 20 min minimum
```

The Layer 1 timetable generator expands these rules into per-train fields:
`t_arr_plan`, `t_dep_plan`, `t_arr_min`, `t_dep_max`, `dwell_min`, `P_i`,
`prio_i`, `pass_flag`, and `seg_runtime_bounds`. If exact planned times are not
provided manually, they are generated from `base_departure_sec`, direction, train
order, and headway settings in the scenario config.

---

### 3.2 Layer 2 — Local Competition Layer (learned, MAPPO)

#### [FIX 7] Clarification: competition vs. cooperation

MAPPO is a cooperative algorithm. All agents share a team objective
(minimize total delay and conflict). "Competition" here refers to the
resource allocation mechanism, not the learning objective.

When two trains request the same platform at the same time, a deterministic
conflict resolver decides who gets it. MAPPO learns strategies that
cooperate at the team level — proactively choosing less-contested slots,
yielding early, and reducing overlaps — because doing so minimizes team delay.
The competition is the environmental mechanism. The cooperation is the
learning objective. They do not conflict.

---

#### [FIX 4] Observation — fixed-size with K_max neighbors

Let `K_max = 5`. Neighbors are selected by sorting conflict-graph neighbors
on `|t_arr_expected_j - t_arr_expected_i|` and taking the closest 5.
If fewer than 5 neighbors exist, pad with zeros and set the validity mask to 0.

**Observation for agent i:**

```
obs_i = [self_state | neighbor_state_1..K_max | neighbor_validity_mask | resource_state]
```

| Block | Dim | Content |
|-------|-----|---------|
| self_state | 13 | t_now, pos, speed, delay_i, t_arr_plan, t_dep_plan, t_arr_min, t_dep_max, dwell_min, prio_i, pass_flag, n_candidate_platforms, current_delta_from_plan |
| neighbor_state ×5 | 5×8=40 | t_arr_expected, dwell_expected, t_dep_expected, n_candidate_platforms, prio, delay, pass_flag, same_platform_flag |
| neighbor_validity_mask | 5 | 1 if neighbor slot is real, 0 if padded |
| resource_state | 16 | platform_occupied[p1..p3]×2 stations, throat_in_busy×2, throat_out_busy×2, turnout_conflict_bits×4, corridor_headway_gap_forward, corridor_headway_gap_backward |
| **Total** | **74** | Fixed across all agents and all timesteps |

---

#### Action Space

Use a small conditional discrete action. The policy only chooses the resource
request. Departure time is derived from the granted entry time and dwell time,
rather than controlled by a separate `delta_out`. This avoids inconsistent
actions such as "enter late but depart early".

```
a_i = (platform_choice, entry_choice, dwell_extend)
```

| Sub-action | Choices | Notes |
|------------|---------|-------|
| `platform_choice` | candidate platform/resource track | masked to P_i and resource compatibility |
| `entry_choice` | {earliest, plan, plan+1, plan+2, plan+4} min | conservative non-negative request set |
| `dwell_extend` | {0, +1, +2} min | only for stopping trains; forced to 0 for pass-through trains |

For the current SUMO network there is no independent `pass_track`; pass-through
trains use a platform track whose config has `through_allowed=true` and receive
zero dwell. If a future network adds dedicated pass tracks, they can be added to
`P_i` through config without changing the policy code.

Raw action count is at most `4 * 5 * 3 = 60`, and the valid count is usually
much smaller after platform, pass-through, kinematic, and time-bound masks.
This is intentionally more conservative than using independent in/out shifts.
It gives MAPPO fewer brittle combinations and makes convergence easier.

**[FIX 6] Action masking — implementation:**

```python
ENTRY_CHOICES = ["earliest", "plan", "plan+1", "plan+2", "plan+4"]
DWELL_EXTEND_CHOICES = [0, 60, 120]

def candidate_actions(train_i, layer1):
    for platform in layer1.P_i[train_i]:
        for entry_choice in ENTRY_CHOICES:
            for dwell_extend in DWELL_EXTEND_CHOICES:
                yield (platform, entry_choice, dwell_extend)

def requested_entry_time(train_i, entry_choice, layer1, t_now, travel_time_to_throat):
    earliest_physical = t_now + travel_time_to_throat
    earliest_allowed = max(earliest_physical, layer1.t_arr_min[train_i])

    if entry_choice == "earliest":
        return earliest_allowed

    offset_min = 0 if entry_choice == "plan" else int(entry_choice.split("+")[1])
    planned_request = layer1.t_arr_plan[train_i] + offset_min * 60
    return max(planned_request, earliest_allowed)

def derive_schedule(train_i, action, layer1, t_now, travel_time_to_throat):
    platform, entry_choice, dwell_extend = action
    pass_train = layer1.pass_flag[train_i]

    t_in_start = requested_entry_time(
        train_i, entry_choice, layer1, t_now, travel_time_to_throat)
    t_in_end = t_in_start + THROAT_CLEARANCE_TIME

    if pass_train:
        dwell = 0
        t_plat_start = t_in_end
        t_plat_end = t_plat_start + THROUGH_PLATFORM_CLEARANCE_TIME
    else:
        dwell = layer1.dwell_min[train_i] + dwell_extend
        t_plat_start = t_in_end
        t_plat_end = t_plat_start + dwell

    t_out_start = t_plat_end
    t_out_end = t_out_start + THROAT_CLEARANCE_TIME
    return t_in_start, t_in_end, t_plat_start, t_plat_end, t_out_start, t_out_end

def is_legal_action(train_i, action, layer1, t_now, travel_time_to_throat):
    platform, entry_choice, dwell_extend = action
    pass_train = layer1.pass_flag[train_i]

    if platform not in layer1.P_i[train_i]:
        return False
    if pass_train and not layer1.through_allowed[train_i][platform]:
        return False
    if pass_train and dwell_extend != 0:
        return False

    times = derive_schedule(train_i, action, layer1, t_now, travel_time_to_throat)
    t_in_start, _, _, _, t_out_start, t_out_end = times

    if t_in_start < layer1.t_arr_min[train_i]:
        return False
    if t_out_start > layer1.t_dep_max[train_i]:
        return False
    if not respects_segment_runtime_bounds(train_i, t_in_start, t_out_end, layer1):
        return False
    if not is_kinematically_feasible(train_i, times):
        return False
    return True

def compute_joint_action_mask(train_i, layer1, t_now, travel_time_to_throat):
    actions = list(candidate_actions(train_i, layer1))
    mask = [
        is_legal_action(train_i, a, layer1, t_now, travel_time_to_throat)
        for a in actions
    ]

    # If every request is illegal, do not sample from the policy.
    # Trigger deterministic replan/hold logic outside the policy instead.
    if not any(mask):
        return actions, mask, "NO_LEGAL_ACTION"
    return actions, mask, "OK"
```

The mask is applied to the full joint action before sampling. The logits of
illegal joint actions are set to `-inf` before softmax. The policy never outputs
an illegal action. If no legal action exists, the environment executes a
deterministic hold/replan transition; this transition is not treated as a
policy-selected action.

**Hard-boundary rule:** a boundary is hard only if violating actions are absent
from the policy's feasible set. If an illegal action can still be sampled and is
only punished in the reward, the boundary has become soft. Therefore
`t_arr_min`, `t_dep_max`, minimum dwell, route compatibility, and through-track
compatibility are enforced by the joint action mask, not by reward penalties.

---

#### [FIX 1] Competition Resolution Mechanism

This is the mechanism that converts agent actions into granted resource windows.
It runs deterministically after all agents in the local group submit their actions.

**Step 1 — Convert action to time-window request:**

```python
@dataclass
class WindowRequest:
    train_id:    str
    platform:    str
    W_in:        Tuple[float, float]   # [t_in_start, t_in_end]
    W_plat:      Tuple[float, float]   # [t_plat_start, t_plat_end]
    W_out:       Tuple[float, float]   # [t_out_start, t_out_end]
    priority:    int
    delay:       float

def action_to_request(train_i, action, layer1, t_now, travel_time_to_throat) -> WindowRequest:
    platform, _, _ = action
    times = derive_schedule(train_i, action, layer1, t_now, travel_time_to_throat)
    t_in_start, t_in_end, t_plat_start, t_plat_end, t_out_start, t_out_end = times

    return WindowRequest(train_i, platform,
                         (t_in_start, t_in_end),
                         (t_plat_start, t_plat_end),
                         (t_out_start, t_out_end),
                         layer1.prio_i[train_i], current_delay_i)
```

**Step 2 — Conflict detection:**

Two requests conflict if they share any resource AND their time windows overlap:

```python
def windows_overlap(w1, w2, gap=MIN_CLEARANCE_SECONDS):
    return w1[0] < w2[1] + gap and w2[0] < w1[1] + gap

def requests_conflict(r1, r2):
    same_platform     = (r1.platform == r2.platform)
    same_throat_in    = throat_in_routes_conflict(r1.train_id, r2.train_id)
    same_throat_out   = throat_out_routes_conflict(r1.train_id, r2.train_id)
    if same_platform  and windows_overlap(r1.W_plat, r2.W_plat): return True
    if same_throat_in and windows_overlap(r1.W_in,   r2.W_in):   return True
    if same_throat_out and windows_overlap(r1.W_out,  r2.W_out):  return True
    return False
```

**Step 3 — Resolution rule (deterministic, priority-weighted):**

```python
def resolve(r1, r2):
    """
    Returns (winner, loser).
    Rule:
      1. Lower prio_i number wins (1 beats 2).
      2. Tie on priority: higher delay wins (more delayed train is protected).
      3. Tie on both: lower train_id wins (deterministic tiebreak).
    """
    score_i = (r1.priority, -r1.delay, r1.train_id)
    score_j = (r2.priority, -r2.delay, r2.train_id)
    if score_i <= score_j:
        return r1, r2   # r1 wins, r2 yields
    else:
        return r2, r1   # r2 wins, r1 yields
```

**Step 4 — Yield assignment:**

The losing train does not get an empty response. It receives the
earliest feasible window on its second-choice platform (or same platform
at the next available slot):

```python
def assign_fallback(loser, occupied_windows, layer1):
    for platform in layer1.P_i[loser.train_id]:
        if platform == loser.platform:
            continue  # try other platforms first
        earliest = find_earliest_feasible(platform, loser, occupied_windows, layer1)
        if earliest is not None:
            return earliest
    # No other platform: delay on same platform
    return delay_on_same_platform(loser, occupied_windows, layer1)
```

`find_earliest_feasible` and `delay_on_same_platform` must use the same hard
feasibility checks as `is_legal_action`: no fallback may violate `t_arr_min`,
`t_dep_max`, route compatibility, minimum dwell, or kinematic feasibility. If no
bounded feasible slot exists, the train is held before the controlled area and
Layer 2 is asked to replan at the next decision step.

**What MAPPO learns:**
MAPPO does not learn the resolution rule — that is fixed and deterministic.
MAPPO learns *when to request aggressively vs. when to proactively yield*,
because an agent that always fights for the first slot will frequently be
assigned a poor fallback window. The optimal policy emerges from the
team reward pressure.

---

#### [FIX 5] Reward — credit assignment fix

Conflicts and deadlocks are joint phenomena. Assigning them to individuals
creates credit assignment problems. The fix is a hybrid individual + team reward:

Hard boundary violations are intentionally not reward terms. If a train can
violate `t_arr_min`, `t_dep_max`, route compatibility, or minimum dwell and only
receives a negative reward afterward, that constraint is soft. In this design,
such actions are removed before sampling, so the reward only optimizes among
legal schedules.

```
r_i = r_individual_i + λ_team * r_team
```

**Individual component:**

```
r_individual_i = - w1 * wait_i
                 - w2 * delay_i * prio_weight(prio_i)
                 + w3 * completion_i
                 - w4 * excessive_hold_i
```

| Term | Definition |
|------|-----------|
| `wait_i` | Time spent waiting outside an allocated resource |
| `delay_i` | max(0, t_actual_dep - t_dep_plan) |
| `completion_i` | +1 on successful station passage, 0 otherwise |
| `excessive_hold_i` | max(0, actual_dwell - dwell_min - tolerance) |
| `prio_weight(p)` | 2.0 if p=1 (high), 1.0 if p=2, 0.5 if p=3 |

**Team component (shared equally, not individual):**

```
r_team = - w5 * (n_conflicts_this_step / N_active)
         - w6 * deadlock_flag
         - w7 * (total_system_delay / N_active)
```

`n_conflicts_this_step` = number of unresolved time-window overlaps after resolution.
`deadlock_flag` = 1 if any local group has zero-progress for T_deadlock steps.

`λ_team = 0.3` is a starting value; tune during ablation.

**Rationale:** Individual terms give each agent a clear gradient for its own behavior.
Team terms penalize system-wide failure modes (conflicts, deadlocks) without
assigning them to a single agent who may not have caused them.

---

#### [FIX 2] Local Centralized Critic (MAPPO-within-groups)

Standard MAPPO uses a global centralized critic, which does not scale.
The fix: each agent has a critic that uses the concatenated observations
of all agents in its **local conflict group**, not the full system.

```python
def get_local_group(agent_i, conflict_graph, K_max=5):
    neighbors = conflict_graph.neighbors(agent_i)
    # Sort by conflict urgency (time overlap tightness)
    sorted_neighbors = sorted(neighbors,
                               key=lambda j: time_overlap_score(agent_i, j))
    group = [agent_i] + sorted_neighbors[:K_max]
    return group

def critic_input(agent_i, conflict_graph, observations):
    group = get_local_group(agent_i, conflict_graph)
    self_obs = [observations[agent_i]]
    neighbor_ids = [j for j in group if j != agent_i]
    neighbor_ids = sorted(neighbor_ids, key=lambda j: time_overlap_score(agent_i, j))
    group_obs = self_obs + [observations[j] for j in neighbor_ids]
    # Pad to K_max + 1 (self + K_max neighbors)
    while len(group_obs) < K_max + 1:
        group_obs.append(zero_observation())
    return np.concatenate(group_obs)  # fixed size: (K_max+1) * 74
```

**Properties:**
- Critic input size = `(K_max + 1) * 74 = 6 * 74 = 444` — fixed regardless of N
- The first critic slot is always the agent itself; neighbor slots are urgency-sorted
- Groups are recomputed dynamically each step as the conflict graph changes
- This is a locally centralized critic, not a globally centralized one

**Comparison:**

| Critic type | Input size | Scales with N? |
|------------|------------|---------------|
| Global (standard MAPPO) | N × 74 | No |
| Local (this design) | 6 × 74 = 444 | Yes — O(1) |
| None (IPPO) | 74 | Yes — but high variance |

---

### 3.3 Layer 3 — Execution Layer (rule-based)

#### [FIX 3] Layer 2 → Layer 3 Interface

This was missing in the original document. The interface is an
`ExecutionTarget` struct passed from Layer 2 to Layer 3 after
each resource window is granted.

```python
@dataclass
class ExecutionTarget:
    train_id:              str
    allocated_platform:    str
    route_segments:        List[str]   # authorized SUMO edge IDs in order

    # Target times (seconds, simulation clock)
    t_throat_entry:        float  # when to enter inbound throat
    t_platform_arrival:    float  # when to reach platform stopping mark
    t_platform_departure:  float  # when doors close and train begins moving
    t_throat_exit:         float  # when to clear outbound throat

    # Kinematic hints
    v_at_throat_entry:     float  # target speed when entering throat (m/s)
    v_at_platform:         float  # must be 0.0 for stopping trains
```

**Layer 3 rule-based controller:**

```python
class RuleBasedExecutor:
    def step(self, train, target: ExecutionTarget, t_now):
        phase = self.get_phase(train, target)

        if phase == "approach_throat":
            # Compute required deceleration to arrive at t_throat_entry
            # at v_at_throat_entry using trapezoidal speed profile
            return self.compute_speed_for_arrival(
                train, target.t_throat_entry, target.v_at_throat_entry, t_now)

        elif phase == "throat_to_platform":
            # Low speed, follow switch route, respect turnout speed limits
            return min(train.speed, THROAT_MAX_SPEED)

        elif phase == "braking_to_stop":
            # Brake to stop at platform mark
            return self.compute_braking_profile(train, target.t_platform_arrival)

        elif phase == "dwell":
            # Open doors, wait for t_platform_departure
            self.door_control(train, "open")
            if t_now >= target.t_platform_departure:
                self.door_control(train, "close")
                return DEPARTURE_SPEED_CMD
            return 0.0

        elif phase == "departure_throat":
            return self.compute_speed_for_exit(
                train, target.t_throat_exit, t_now)

        elif phase == "corridor":
            # Headway-safe speed on main track
            return self.headway_safe_speed(train)

    def compute_speed_for_arrival(self, train, t_target, v_target, t_now):
        d_remaining = train.distance_to(target.t_throat_entry_position)
        t_remaining = t_target - t_now
        if t_remaining <= 0:
            return v_target
        v_required = d_remaining / t_remaining  # average speed needed
        # Apply kinematic feasibility check
        v_cmd = min(v_required, train.v_max,
                    kinematic_max_speed(train.speed, d_remaining, v_target))
        return v_cmd

    def headway_safe_speed(self, train):
        gap = train.gap_to_leading_train()
        if gap < SAFE_HEADWAY_CRITICAL:
            return 0.0
        elif gap < SAFE_HEADWAY_WARNING:
            return train.speed * (gap / SAFE_HEADWAY_WARNING)
        else:
            return train.v_max_segment
```

**Key property:** Layer 3 never makes resource decisions.
It only executes what Layer 2 has granted. If the granted window
becomes infeasible due to upstream delays, Layer 2 is triggered
to re-request a new window.

**Re-trigger condition:**

```python
def needs_replan(train, target, t_now):
    slack = target.t_throat_entry - t_now - travel_time_remaining(train)
    return slack < REPLAN_THRESHOLD_SECONDS  # e.g., -30s
```

---

## 4. Full RL Specification (Revised)

### 4.1 Layer 2 Summary

| Item | Specification |
|------|--------------|
| Agent | Each train |
| Observation size | 74 (fixed) |
| Action space | Conditional discrete, at most 60 raw actions; usually fewer after masks |
| Action masking | Full joint-action mask; Layer 1 hard bounds enforced before sampling |
| Critic input | Local group concat, size 444 (fixed) |
| Reward | Hybrid: individual delay/wait + team conflict/deadlock |
| Competition | Deterministic priority-weighted resolver + fallback assignment |

### 4.2 Constraints — enforcement method

| Constraint | Enforced by |
|-----------|-------------|
| Platform single-occupancy | Action masking + resolver |
| Conflicting throat routes no overlap | Resolver |
| Minimum dwell time | Joint action masking (`dwell_extend >= 0`) |
| t_arr_min / t_dep_max | Action masking |
| Minimum tracking headway | Layer 3 headway controller |
| Platform direction compatibility | Action masking |
| No entry without route authorization | Layer 3 route authorization check |

---

## 5. Scalability Properties

With the local critic design:

| Metric | Standard MAPPO | This design |
|--------|---------------|-------------|
| Critic input size | O(N × d_obs) | O(K_max × d_obs) = const |
| Communication edges | O(N²) | O(N × K_max) = O(N) |
| Conflict resolution | Learned implicitly | Deterministic O(K²) per group |
| Group size at test time | N | ≤ K_max + 1 = 6 |

Performance is expected to degrade gracefully with N because
each agent's decision depends only on its local group,
which has bounded size regardless of total train count.

---

## 6. Remaining Design Choices (not fixed, needs tuning)

These parameters should be set by ablation:

| Parameter | Suggested start | Ablation range |
|-----------|----------------|----------------|
| K_max | 5 | {3, 5, 7} |
| λ_team | 0.3 | {0.1, 0.3, 0.5} |
| REPLAN_THRESHOLD_SECONDS | 30 | {15, 30, 60} |
| entry_choice set | {earliest, plan, plan+1, plan+2, plan+4} | compare with {earliest, plan, plan+2, plan+5} |
| prio_weight for high-priority | 2.0 | {1.5, 2.0, 3.0} |

---

## 7. What Has Not Changed from Original

- Section 2 scenario and physical layout
- Section 4 resource chain formulation (W_in, W_plat, W_out)
- Section 4.4 local conflict graph construction
- Section 7 metrics
- Section 8 project structure
- Section 9 implementation roadmap (stages 1–5)

---

## 8. Summary of All Fixes

| # | Original problem | Fix applied |
|---|-----------------|-------------|
| 1 | Competition mechanism missing | Deterministic priority-weighted resolver with fallback assignment |
| 2 | Critic scope undefined | Local group critic, input size O(K_max) not O(N) |
| 3 | Layer 2→Layer 3 interface missing | `ExecutionTarget` struct + rule-based executor with re-trigger |
| 4 | Variable neighbor count breaks NN | Fixed K_max=5 with zero-padding and validity mask |
| 5 | conflict_i/deadlock_i credit assignment | Hybrid reward: individual terms + shared team terms |
| 6 | Hard boundary in reward (contradiction) | Removed from reward; enforced by action masking only |
| 7 | Competition vs cooperation contradiction | Clarified: mechanism is competitive, learning objective is cooperative |

---

## 9. Implementable Experiment Plan

### 9.1 Core requirement

The experiment must be **configuration-driven**. Adding trains, stations,
platforms, throat routes, or corridor sections must not require changes to
training code, environment code, policy code, or reward code. Only SUMO network
files and scenario/config files may change.

The Python system should discover the active railway problem from:

- SUMO network topology (`.net.xml`)
- SUMO route and train demand files (`.rou.xml` or generated trip files)
- A railway resource config (`railway_resources.yaml`)
- A timetable and dispatch-bound config (`timetable.yaml`)
- A training config (`train_config.yaml`)

No station name, train ID, platform ID, or number of trains should be hard-coded
inside Python modules.

---

### 9.2 Recommended project structure

```text
project/
  configs/
    railway_resources.yaml
    timetable.yaml
    train_config.yaml
    scenarios/
      two_station_base.yaml
      two_station_dense.yaml
      three_station_generalization.yaml
  sumo/
    nets/
      two_station.net.xml
      three_station.net.xml
    routes/
      base.rou.xml
      dense.rou.xml
  src/
    env/
      sumo_env.py
      topology.py
      resource_manager.py
      action_mask.py
      observation_builder.py
      reward.py
    control/
      dispatch_bounds.py
      resolver.py
      executor.py
    marl/
      mappo.py
      actor_critic.py
      rollout_buffer.py
    experiment/
      run_train.py
      run_eval.py
      metrics.py
      scenario_loader.py
  outputs/
    checkpoints/
    logs/
    eval_reports/
```

---

### 9.3 Configuration schema

The railway resource config describes resources, not algorithms:

```yaml
stations:
  - id: A
    platforms:
      - id: A_p1
        through_allowed: true
      - id: A_p2
        through_allowed: true
      - id: A_p3
        through_allowed: true
    pass_tracks: []
    throat_in_routes:
      - id: A_in_1
        edges: [A_in_edge_1, A_sw_1]
        conflicts: [A_in_2]
    throat_out_routes:
      - id: A_out_1
        edges: [A_sw_3, A_out_edge_1]
        conflicts: [A_out_2]

  - id: B
    platforms:
      - id: B_p1
        through_allowed: true
      - id: B_p2
        through_allowed: true
      - id: B_p3
        through_allowed: true
    pass_tracks: []
    throat_in_routes:
      - id: B_in_1
        edges: [B_in_edge_1, B_sw_1]
        conflicts: [B_in_2]
    throat_out_routes:
      - id: B_out_1
        edges: [B_sw_3, B_out_edge_1]
        conflicts: [B_out_2]

corridors:
  - id: A_B_up
    from_station: A
    to_station: B
    edges: [AB_up_1, AB_up_2]
    min_headway_sec: 120
  - id: B_A_down
    from_station: B
    to_station: A
    edges: [BA_down_1, BA_down_2]
    min_headway_sec: 120
```

The timetable config describes trains and Layer 1 bounds:

```yaml
priority_rules:
  high: {classes: [G, D, C], priority: 1}
  medium: {classes: [Z, T], priority: 2}
  low: {classes: [K], priority: 3}

station_dwell_min_sec:
  A: 600
  B: 240

time_generation:
  base_departure_sec:
    A_to_B: 0
    B_to_A: 0
  headway_sec:
    A_to_B: 180
    B_to_A: 180
  corridor_runtime_sec:
    A_to_B: 360
    B_to_A: 360
  dep_max_slack_sec: 600

trains:
  - id: G5050
    class: G
    direction: A_to_B
    origin: A
    destination: B
    stop_pattern:
      A: stop
      B: stop
    candidate_platforms:
      A: [A_p1, A_p2, A_p3]
      B: [B_p1, B_p2, B_p3]

  - id: D133
    class: D
    direction: A_to_B
    origin: A
    destination: B
    stop_pattern: {A: stop, B: stop}
    candidate_platforms:
      A: [A_p1, A_p2, A_p3]
      B: [B_p1, B_p2, B_p3]

  - id: K641
    class: K
    direction: A_to_B
    origin: A
    destination: B
    stop_pattern: {A: stop, B: stop}
    special_dwell_min_sec:
      A: 1200  # locomotive change at A
    candidate_platforms:
      A: [A_p1, A_p2, A_p3]
      B: [B_p1, B_p2, B_p3]

  - id: Z186
    class: Z
    direction: B_to_A
    origin: B
    destination: A
    stop_pattern: {B: stop, A: stop}
    candidate_platforms:
      B: [B_p1, B_p2, B_p3]
      A: [A_p1, A_p2, A_p3]

  - id: G1408
    class: G
    direction: B_to_A
    origin: B
    destination: A
    stop_pattern: {B: stop, A: stop}
    candidate_platforms:
      B: [B_p1, B_p2, B_p3]
      A: [A_p1, A_p2, A_p3]

  - id: G5060
    class: G
    direction: B_to_A
    origin: B
    destination: A
    stop_pattern: {B: stop, A: stop}
    candidate_platforms:
      B: [B_p1, B_p2, B_p3]
      A: [A_p1, A_p2, A_p3]
```

Layer 1 expands the compact timetable config as follows:

```python
def train_priority(train_class, priority_rules):
    for rule in priority_rules.values():
        if train_class in rule["classes"]:
            return rule["priority"]
    return 99

def dwell_min_for(train_id, station_id, train_cfg, station_dwell_min_sec):
    base = station_dwell_min_sec[station_id]
    special = train_cfg.get("special_dwell_min_sec", {}).get(station_id)
    return max(base, special or 0)

def expand_layer1_train(train_cfg, station_id, generated_times, resource_cfg):
    arr_plan = generated_times[train_cfg["id"]][station_id]["arr"]
    dep_plan = generated_times[train_cfg["id"]][station_id]["dep"]
    dwell_min = dwell_min_for(
        train_cfg["id"], station_id, train_cfg, station_dwell_min_sec)

    return {
        "t_arr_plan": arr_plan,
        "t_dep_plan": dep_plan,
        "t_arr_min": arr_plan,
        "t_dep_max": dep_plan + dep_max_slack_sec,
        "dwell_min": dwell_min,
        "P_i": train_cfg["candidate_platforms"][station_id],
        "prio_i": train_priority(train_cfg["class"], priority_rules),
        "pass_flag": train_cfg["stop_pattern"][station_id] == "pass",
        "seg_runtime_bounds": runtime_bounds_for(train_cfg, resource_cfg),
    }
```

When adding a new train, add one `trains` entry. When adding a new station, add
one `stations` entry and update the timetable. The code should rebuild agents,
resources, conflict graphs, masks, and metrics automatically.

---

### 9.4 Environment design

The environment should follow a Gym-style interface:

```python
obs, info = env.reset(scenario_config)
obs, reward, terminated, truncated, info = env.step(action_dict)
```

Where:

- `action_dict` maps `train_id -> discrete_action_index`.
- `obs` maps `train_id -> fixed_size_observation`.
- `info["action_masks"]` maps `train_id -> valid_action_mask`.
- `info["execution_targets"]` records granted windows for debugging.
- `info["metrics"]` records delay, wait, conflicts, deadlocks, throughput.

The environment lifecycle per decision step:

1. Pull train states from SUMO through TraCI.
2. Build active train set from SUMO IDs and timetable entries.
3. Build fixed-size observations for all active trains.
4. Build joint action masks from Layer 1 bounds and physical feasibility.
5. Sample legal actions from MAPPO.
6. Convert actions to time-window requests through `derive_schedule`.
7. Resolve conflicts using the deterministic resource manager.
8. Convert granted windows to `ExecutionTarget`.
9. Apply rule-based speed/route commands to SUMO.
10. Advance SUMO for `decision_interval_sec`.
11. Compute reward and metrics.

---

### 9.5 Code-level scalability rules

The following rules make train/station expansion possible without code changes:

1. Use train IDs from SUMO and timetable config, not fixed agent names.
2. Use station/platform/resource IDs from `railway_resources.yaml`.
3. Build the conflict graph from resource usage windows, not station-specific if-statements.
4. Build observations with padding and masks, so tensor size is independent of train count.
5. Use a shared actor network for all trains.
6. Use the local critic input `(self + K_max neighbors)`, not global concatenation.
7. Use resource calendars keyed by resource ID: `resource_id -> occupied_windows`.
8. Store all route compatibility in config, not in Python conditionals.
9. Treat new stations as more resources in the same resource manager.
10. Keep reward formulas aggregate and ID-agnostic.

---

### 9.6 Experiment phases

#### Phase 0: SUMO sanity baseline

Goal: verify that the railway network and routes are physically valid before RL.

Tasks:

- Build the two-station SUMO network.
- Define 6 trains: 3 up, 3 down.
- Run fixed timetable execution with no learning.
- Verify every train can complete its route.
- Log arrival time, departure time, platform occupancy, and headway.

Pass criteria:

- No SUMO route error.
- No train is stuck forever.
- Platform and throat occupancy logs match expected resource use.

#### Phase 1: Rule-based resource manager only

Goal: verify that time windows, hard masks, conflict detection, fallback, and
execution targets work before MAPPO is introduced.

Tasks:

- Replace policy actions with a deterministic heuristic:
  - choose earliest legal entry
  - choose first legal platform
  - choose minimum dwell
- Run the resource resolver.
- Execute granted targets in SUMO.
- Stress test dense headways.

Pass criteria:

- No granted window violates Layer 1 bounds.
- No two trains occupy the same platform window.
- No conflicting throat routes overlap.
- If no legal action exists, the train is held and replanned.

#### Phase 2: MAPPO integration on the minimum network

Goal: train the proposed Layer 2 policy on the two-station network.

Tasks:

- Implement shared actor and local critic.
- Use action masks during rollout and PPO update.
- Store action log-prob only for legal sampled actions.
- Train against the hybrid reward.
- Save checkpoints and evaluation metrics.

Starting training settings:

| Item | Start value |
|------|-------------|
| decision_interval_sec | 10 |
| rollout_steps | 1024 |
| PPO epochs | 5 |
| minibatch_size | 256 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_ratio | 0.2 |
| entropy_coef | 0.01 |
| value_coef | 0.5 |
| learning_rate | 3e-4 |
| K_max | 5 |

Pass criteria:

- Mean total delay improves over the heuristic baseline.
- Conflict rate after resolution remains zero or near zero.
- Deadlock rate is lower than the heuristic baseline under dense demand.

#### Phase 3: Baseline comparison

Goal: show whether the three-layer design is useful.

Compare:

1. Fixed rule-based dispatch only.
2. Flat MAPPO with global critic on the same network.
3. IPPO without centralized critic.
4. Proposed local-critic hierarchical MAPPO.

Metrics:

- mean delay per train
- max delay per train
- mean waiting time before station entry
- platform conflict count before and after resolution
- unresolved conflict count
- deadlock count
- completion rate
- throughput
- inference time per decision step
- critic input dimension

Report mean and standard deviation over at least 5 random seeds.

#### Phase 4: Generalization test without code changes

Goal: prove that adding trains or stations only changes config/SUMO files.

Tests:

| Test | Change allowed | Code change allowed? |
|------|----------------|----------------------|
| More trains | Add entries in timetable/routes | No |
| More platforms | Add platform IDs in resource config | No |
| Three stations | Add SUMO net, station config, timetable | No |
| Different stop pattern | Change timetable only | No |
| Different priority mix | Change timetable only | No |

Pass criteria:

- The same trained/evaluation scripts run unchanged.
- Observation tensor shape remains fixed.
- Actor output dimension remains fixed.
- Critic input dimension remains fixed at `(K_max + 1) * obs_dim`.
- Resource manager creates calendars for new resources automatically.

#### Phase 5: Ablation study

Goal: identify which design choices matter.

Ablations:

- `K_max`: 3, 5, 7
- team reward weight: 0.1, 0.3, 0.5
- entry choice set: small vs medium
- with vs without local critic
- with vs without deterministic fallback
- with vs without pass-track action

Primary claim is supported only if the proposed method improves delay or
deadlock rate while preserving hard feasibility.

---

### 9.7 Minimal implementation order

Implement in this order:

1. Scenario/config loader.
2. SUMO wrapper and train state extractor.
3. Topology/resource parser.
4. Resource calendar and conflict checker.
5. `derive_schedule` and joint action mask.
6. Rule-based fallback resolver.
7. Execution target generator.
8. Rule-based SUMO executor.
9. Observation builder with padding and masks.
10. Reward and metric logger.
11. Heuristic baseline.
12. MAPPO rollout buffer.
13. Shared actor and local critic.
14. PPO update with action masks.
15. Evaluation and report scripts.

Do not start MAPPO training before Phase 1 passes. Most failures in this
experiment will come from invalid resource accounting or SUMO execution drift,
not from the neural network.

---

### 9.8 Acceptance checklist

The experiment is considered implementable when all of the following are true:

- Adding a train requires only SUMO route/timetable config changes.
- Adding a station requires only SUMO network/resource/timetable config changes.
- Python code contains no station-specific or train-specific branches.
- Every policy action is checked by the joint action mask before sampling.
- Every fallback window is checked by the same hard feasibility logic.
- Every granted window is logged and can be replayed.
- Evaluation can run with a checkpoint on a scenario not used in training.
- Metrics are exported to CSV/JSON for every seed and scenario.
