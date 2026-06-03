# Railway MARL — Implementation Specification v4.1


> **Last updated: 2026-06-03 (V4.1: L2-first, SUMO-config-driven, L3-future-compatible)**
> **Status:** Phase 0 / 1 / 2 DONE. Phase 3 skeleton exists. **Next: finish L2 learning environment and MAPPO.**
> **Revision scope:** Preserves the original three-layer goal while making SUMO topology configurable, keeping Layer 1 hard boundaries non-negotiable, and reserving a clean Layer 3 learning interface for later work.

---

## REVISION SUMMARY

This revision incorporates all improvements from the architecture review and subsequent critic-fusion refinement:

1. **L2-first learning scope** – Current implementation trains only Layer 2. Layer 3 remains rule-based in the main path, but consumes a stable `ExecutionTarget` interface so it can later be replaced by a learned Worker.
2. **SUMO/config-driven topology** – Python must not contain business branches for specific station IDs, train IDs, platform IDs, SUMO edge IDs, route IDs, or the number of stations. Extending the SUMO environment must require only `sumo/` and `configs/` changes plus model retraining when observation topology changes.
3. **Hard-boundary safety** – Layer 1 envelopes are never softened by reward or degraded fallback. If no legal fallback exists, the train is held/replanned or the episode is marked infeasible; no out-of-bound plan is granted.
4. **Dual-Critic with Loss-Level Fusion** – Two independent value heads (Local Safety, Global Efficiency) are fused only in the loss functions, preserving standard MAPPO stability while injecting a global efficiency signal.
5. **Layered Reward Alignment** – Rewards are decomposed into local and global components for the Phase 4 baseline; station critic remains an ablation.
6. **Decision/control step split** – Layer 2 dispatch decisions use `decision_interval_sec = 10`; Layer 3/SUMO monitoring uses `control_step_sec = 1`.
7. **Corridor Conflict Design** – Corridor headway is **not** a resolver resource, consistent with the double-track physical layout.

All completed Phase 0–2 modules are **backward‑compatible**; only targeted additions are required in upcoming phases.

---

## RESEARCH QUESTION

When random delays break the nominal conflict‑free timetable, can the
three‑layer framework recover better than flat MAPPO and rule‑based
rescheduling, while respecting hard Layer 1 dispatch boundaries?

Layer 1 is rule‑based and fixed. Layer 2 is the only learned layer.
Layer 3 is rule‑based. Layer 2 activates only when a disturbance is detected.

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

## TRAINS (8 + 1 conflict‑test train)

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

### Conflict‑test train (disable by commenting in config)

| id   | class | direction | depart | notes                          |
|------|-------|-----------|--------|--------------------------------|
| K999 | K     | A→B       | 120    | conflicts D133, **Phase 2‑3 development/debug only**; remove before final experiments |

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

**Episode Termination (NEW)**
An episode ends under any of the following conditions:

1. **SUCCESS**: All scheduled trains have completed their full routes.
2. **TRUNCATED**: Simulation time exceeds `MAX_EPISODE_TIME = 900` s.
3. **DEADLOCK**: An irrecoverable deadlock is detected (all active agents have `NO_LEGAL_ACTION` for ≥ `DEADLOCK_THRESHOLD = 15` consecutive steps).

A terminal reward of 0 is assigned at termination; GAE bootstraps from value estimates at truncated episodes. Deadlocked episodes are assigned a large negative penalty and the episode is terminated.

---

## LAYER 1 — DISPATCH BOUNDS (rule‑based, fixed)

(Unchanged from original. Reads `timetable_layer1.yaml`, produces `TrainBounds`.)

---

## LAYER 2 — LOCAL COMPETITION (learned, MAPPO)

### Disturbance and trigger

Layer 2 is idle during nominal operations. Activates on disturbance detection. **A deactivation mechanism is also defined (NEW)** to return trains to nominal control once disturbances are resolved.

```python
TRIGGER_THRESHOLD_SEC = 30   # tune in ablation
DISTURBANCE_PROB_PER_STEP = 0.03  # (REVISED from 0.005) to ensure sufficient training samples

class DisturbanceInjector:
    def __init__(self, prob_per_step=DISTURBANCE_PROB_PER_STEP, delay_range=(30, 180)):
        self.prob_per_step = prob_per_step
        self.delay_range   = delay_range

    def step(self, trains, t_now):
        for train in trains:
            if random.random() < self.prob_per_step:
                delta = random.uniform(*self.delay_range)
                train.apply_delay(delta)

class ControlStatus(Enum):
    LAYER2_ACTIVE = "layer2_active"
    LAYER2_MONITORING = "layer2_monitoring"
    NOMINAL = "nominal"

def layer2_control_status(train, bounds: TrainBounds, t_now: float,
                          conflict_graph) -> ControlStatus:
    """Determines whether Layer 2 should be active for this train."""
    predicted_arr = t_now + train.travel_time_to_throat()
    if abs(predicted_arr - bounds.t_arr_plan) > TRIGGER_THRESHOLD_SEC:
        return ControlStatus.LAYER2_ACTIVE
    if conflict_graph.has_active_conflict(train.id):
        return ControlStatus.LAYER2_ACTIVE
    # Train is recovering; keep monitoring if there are pending Layer 2 targets
    if train.has_pending_layer2_targets():
        return ControlStatus.LAYER2_MONITORING
    return ControlStatus.NOMINAL
```

**When status is NOMINAL, the train uses its original timetable (*not* Layer 2 policy).** This prevents trains from being "orphaned" after disturbance recovery.

### Observation (fixed 74‑dim for current 2‑station network)

```
obs_i = [self_state(14) | neighbor_state×5(40) | validity_mask(5) | resource_state(15)]
```

**Dimension note:** Fixed at 74 for the current 2‑station topology. When adding
stations, only YAML and net files change; resource_state grows and the model must
be retrained — no Python source code changes required.

| block             | dim    | fields |
|-------------------|--------|--------|
| self_state        | 14     | t_now, pos, speed, delay_i, t_arr_plan, t_dep_plan, t_arr_min, t_dep_max, dwell_min, prio_i, pass_flag, remaining_slack_to_dep_max, n_candidate_platforms, delta_from_plan |
| neighbor_state ×5 | 40     | t_arr_expected, dwell_expected, t_dep_expected, n_candidate_platforms, prio, delay, pass_flag, same_platform_flag |
| validity_mask     | 5      | 1=real neighbor, 0=padded |
| resource_state    | 15     | platform_occupied[A_p1,A_p2,A_p3,B_p1,B_p2,B_p3](6), throat_in_busy[A,B](2), throat_out_busy[A,B](2), turnout_bits[A_in,A_out,B_in,B_out](4), headway_gap_fwd(1) |
| **Total**         | **74** | fixed |

Neighbors: K_max=5. Select first from direct resource‑conflict graph neighbors
(trains competing for same platform / throat / turnout route). Fill remaining
slots by |t_arr_expected_j - t_arr_expected_i| ascending. Pad zeros if under 5.

### Action space (max 45 combinations in the Phase 4 baseline)

```
a_i = (platform_choice, entry_choice, dwell_extend)
```

| sub‑action      | choices                                            |
|-----------------|----------------------------------------------------|
| platform_choice | elements of P_i                                    |
| entry_choice    | {earliest, plan, plan+1, plan+2, plan+4} (minutes) |
| dwell_extend    | {0, +60, +120} seconds; forced 0 for pass-through |

The finer `{0,+30,+60,+90,+120}` dwell grid is reserved for Phase 6 ablation.

### Action masking

(Unchanged from original; only updated dwell enum.)

### Competition resolver

**Corridor headway: NOT a resolver resource (design decision retained).**
`up_main` and `down_main` are physically independent tracks. Trains in opposite
directions never share a track, so there is no corridor‑level resource conflict
to resolve. Introducing opposing‑direction checks here would generate spurious
conflicts (e.g. G5050 A→B vs G1408 B→A contending for the same "slot" when
they are on different tracks). Minimum tracking headway between same‑direction
trains is enforced solely by the Layer 3 headway controller.

```python
def requests_conflict(r1, r2, route_cfg):
    # Station resources only — corridor is NOT included here
    if r1.platform == r2.platform and windows_overlap(r1.W_plat, r2.W_plat):
        return True
    if route_cfg.throat_in_conflict(r1.train_id, r2.train_id) \
            and windows_overlap(r1.W_in, r2.W_in):
        return True
    if route_cfg.throat_out_conflict(r1.train_id, r2.train_id) \
            and windows_overlap(r1.W_out, r2.W_out):
        return True
    return False
```

**Fallback safety (V4.1):**
When no legal fallback exists within `t_dep_max`, the resolver must not grant any out-of-bound plan. The train is held for replan, and the environment records `hard_bound_infeasible_count`. If this condition persists until the deadlock/infeasible threshold, the episode terminates as infeasible.

### Critic architecture — Dual Critic with Loss‑Level Fusion

The baseline critic architecture uses **two independent value heads**, fused
only in the loss functions. This preserves standard MAPPO training stability
while injecting a global efficiency signal.

#### Local Safety Critic

- Input: compact group observation `(K_local+1) × obs_dim`
  - `K_local = 3` by default (agent + 3 most conflicting neighbours)
  - Total input dim: `4 × 74 = 296` (fixed regardless of total train count)
- Output: scalar `V_local(s_i)`
- Training target: GAE returns computed on `r_local`

#### Global Efficiency Critic

- Input: lightweight global state vector (~14 dim)
  ```
  [N_active, mean_delay, max_delay, total_conflicts,
   n_trains_A, plat_occ_A, throat_busy_A,
   n_trains_B, plat_occ_B, throat_busy_B,
   total_system_delay, deadlock_flag, episode_progress, disturbance_count]
  ```
  **No agent‑identity information is included**, ensuring the critic learns
  system‑level statistics only.
- Output: scalar `V_global(s)`
- Training target: GAE returns computed on `r_global`

#### Fusion at the Loss Level

Both critics are trained independently, each with its own TD target. The
fusion happens **only in the policy and critic loss functions**:

```python
# ---- Advantage computation (standard GAE, no fusion here) ----
A_local,  returns_local  = GAE(r_local,  V_local,  gamma, gae_lambda)
A_global, returns_global = GAE(r_global, V_global, gamma, gae_lambda)

# ---- Policy loss ----
A_combined = A_local + lam_team * A_global
L_policy = -(log_probs * A_combined).mean()

# ---- Critic loss ----
L_V = F.mse_loss(V_local_pred, returns_local) \
    + lam_team * F.mse_loss(V_global_pred, returns_global)
```

**Key property:** `lam_team` only scales the loss contributions; it does **not**
mix advantage estimates. Each GAE computation is self‑contained and stable.

#### Curriculum Schedule for `lam_team` (unchanged from reward section)

```python
def get_lambda(episode, total):
    progress = episode / total
    if progress < 0.3:  return 0.05
    elif progress < 0.6: return 0.05 + 0.25 * (progress-0.3)/0.3
    else:               return 0.30
```

Early in training the local critic dominates, allowing agents to learn boundary
compliance first. As `lam_team` increases, the global efficiency signal grows,
naturally shifting the policy toward system‑level coordination — **without
requiring agents to “unlearn” selfish behaviour**.

#### Comparison with Alternative Designs

| Design | Status | Rationale |
|--------|--------|-----------|
| Fusion at advantage level (`A = (1-α)A_local + αA_global`) | Rejected | Non‑standard, gradient attribution unclear, unstable during curriculum |
| Single local group critic (Phase 4 ablation) | Ablation baseline | Simpler, but lacks explicit global signal; used to quantify dual‑critic gain |
| Three‑layer critic (Station Critic) | Phase 6 ablation | Adds intra‑station coordination; optional extension |

### Reward (REDESIGNED with alignment and curriculum)

```
r_i = r_local + λ_station × r_station + λ_team × r_global
```

**r_local** (aligned with Local Safety Critic):
```
r_local = − w1 × wait_i
          − w2 × delay_i × prio_weight(prio_i)
          + w3 × completion_i
          − w4 × excessive_hold_i
          + w_bound × bound_slack_i   (NEW)
```
`bound_slack_i` measures remaining margin to `t_dep_max`; positive when the train is comfortably within bounds, negative when approaching the hard limit.

**r_station** (Phase 6 ablation only):
```
r_station = − w5 × station_conflicts/station_active
            + w6 × station_throughput
```

**r_global** (aligned with Global Efficiency Critic):
```
r_global = − w7 × (total_system_delay / N_active)
           − w8 × deadlock_flag
```

**Initial weights** (NEW, previously unspecified):
| weight | value | range |
|--------|-------|-------|
| w1     | 0.01  | 0.005–0.05 |
| w2     | 0.02  | 0.01–0.1 |
| w3     | 5.0   | 1.0–10.0 |
| w4     | 0.01  | 0.005–0.05 |
| w_bound| 0.02  | 0.01–0.1 |
| w5–w8  | 0.02  | 0.01–0.1 |

**λ curriculum** (NEW):
```python
def get_adaptive_lambda(episode, total_episodes):
    """
    Returns (lambda_station, lambda_team).
    Starts from small nonzero values to avoid a fully selfish phase
    that could entrench anti‑social behaviour before team signal arrives.
    """
    progress = episode / total_episodes
    if progress < 0.3:
        # Warm‑up: small but nonzero team signal from the start
        return 0.0, 0.05
    elif progress < 0.6:
        lam_team = 0.05 + 0.25 * (progress - 0.3) / 0.3   # 0.05 → 0.30
        return 0.0, lam_team
    else:
        lam_station = 0.1 * (progress - 0.6) / 0.4         # 0.0 → 0.10
        return lam_station, 0.30
```

**Reward regularisation** (NEW): During the first 200 episodes, `r_global` is linearly warmed up to prevent early training instability.

---

## LAYER 3 — EXECUTION (rule‑based, fixed; with layered safety guard)

Layer 3 remains rule‑based for the main experiments. **A learning‑based Worker variant is noted as an optional Phase 6 ablation**, following standard H‑MAPPO Manager‑Worker paradigms with intrinsic reward and reward hiding. This is not part of the core specification.

### Multi-layer safety guard (V4.1)

L1 (action mask): Prevents selection of hard-boundary violations.
L2 (execution monitor): Checks at every Layer 3 control step if slack to `t_throat_entry` drops below warning/critical thresholds; triggers emergency replan if necessary.
L3 (hard guard): If departure becomes physically impossible within `t_dep_max`, hold/replan or terminate infeasible. It must not execute an out-of-bound grant in the main experiment.

### Re‑trigger

```python
REPLAN_THRESHOLD_SEC = 30
def needs_replan(train, target, t_now):
    slack = target.t_throat_entry - t_now - train.travel_time_to_throat()
    return slack < -REPLAN_THRESHOLD_SEC
```

---

## ENVIRONMENT LIFECYCLE (per decision step)

(Updated to include termination checks and Layer 2 deactivation.)

```
1.  Pull train states from SUMO via TraCI
2.  Run DisturbanceInjector.step()
3.  For each active train, determine Layer 2 control status
4.  If NOMINAL: apply nominal timetable; skip to step 11
5.  For Layer 2 ACTIVE trains: build observations
6.  Build joint action masks
7.  status == NO_LEGAL_ACTION → hold train, skip policy
8.  Sample legal actions from MAPPO policy (only for controlled trains)
9.  Convert actions → WindowRequests
10. Run deterministic resolver (with fallback degradation)
11. Convert granted windows → ExecutionTargets
12. Check needs_replan for all active targets
13. Apply RuleBasedExecutor commands to SUMO (with safety guard)
14. Advance SUMO by decision_interval_sec (= 10 s)
15. Compute reward and metrics
16. Check episode termination conditions; if terminal, reset or end
```

---

## HYPERPARAMETERS (Updated)

| parameter         | start value | ablation range  | notes |
|-------------------|-------------|-----------------|-------|
| decision_interval | 10 s        | fixed           | Layer 2 dispatch decision interval |
| control_step      | 1 s         | fixed           | Layer 3/SUMO execution monitor interval |
| K_max (neighbour) | 5           | {3, 5, 7}       | neighbours now selected from conflict graph, not time‑sorted |
| K_local (critic)  | 3           | {3, 5}          | NEW: for local critic input |
| λ_team            | curriculum  | {0.1,0.2,0.3} final | NEW: curriculum schedule |
| λ_station         | curriculum  | {0.05,0.1,0.15} final | NEW |
| REPLAN_THRESHOLD  | 30 s        | {15, 30, 60}    | |
| disturbance_prob  | 0.03        | {0.02, 0.05}    | REVISED from 0.005 |
| DEADLOCK_THRESHOLD| 15          | {10, 20}        | NEW |
| rollout_steps     | 1024        |                 | |
| PPO epochs        | 5           |                 | |
| minibatch_size    | 256         |                 | |
| gamma             | 0.99        |                 | |
| gae_lambda        | 0.95        |                 | |
| clip_ratio        | 0.2         |                 | |
| entropy_coef      | 0.01        |                 | |
| learning_rate     | 3e‑4        |                 | |

---

## PROJECT STRUCTURE (Updated)

```
project/
  configs/
    ...
  src/
    env/
      sumo_env.py               ← Will include termination, Layer 2 status
      observation_builder.py    ← Updated neighbour selection, bound_slack
      reward.py                 ← Redesigned layered rewards with curriculum
      disturbance.py            ← Updated prob, Layer 2 status
    control/
      dispatch_bounds.py        ← OK (no change)
      resource_manager.py       ← Add fallback degradation
      action_mask.py            ← OK (dwell enum remains [0, 60, 120])
      resolver.py               ← (integrated with resource_manager)
      executor.py               ← Add safety guard layers
    marl/
      mappo.py                  ← Dual‑critic loss‑level fusion, curriculum
      actor_critic.py           ← Actor unchanged; LocalCritic + GlobalCritic heads
      rollout_buffer.py         ← Stores r_local and r_global
  ...
```

---

## IMPLEMENTATION PHASES (Revised)

### ✅ Phase 0–2 — DONE (no rework needed)

### 🔲 Phase 3 — SUMO TraCI env + critical fixes

**Before starting training, the following P0 fixes must be implemented:**

1. `sumo_env.py`: Add episode termination logic (SUCCESS/TRUNCATED/DEADLOCK)
2. `disturbance.py`: Add `layer2_control_status()` and NOMINAL path
3. `disturbance.py`: Configurable disturbance probability; training configs may use higher probabilities for sample efficiency
4. `reward.py`: Implement layered rewards with specified weights and bound_slack
5. `resource_manager.py` / `resolver.py`: Preserve no-out-of-bound fallback semantics and report held/infeasible trains
6. (Optional) Adjust `decision_interval` to 5 s if throat transit issues observed

### 🔲 Phase 4 — MAPPO integration

Implement the **dual‑critic loss‑level fusion** architecture as the Phase 4 baseline:

- `actor_critic.py`: Add separate `LocalCritic` and `GlobalCritic` heads.
  `LocalCritic` input dim = (K_local+1)*obs_dim = 4*74 = 296.
  `GlobalCritic` input dim = 14 (compact global state).
- `mappo.py`: Compute GAE independently for each critic. Fuse in `L_policy`
  and `L_critic` using `lam_team` weight (no advantage fusion).
- `rollout_buffer.py`: Store both `r_local` and `r_global` per step.
- Use curriculum λ schedule described above.

**Ablation baseline:** Phase 4 also runs a **single local group critic** (input dim 444, no global head) to isolate the benefit of the explicit global head. This is a standard comparison, not the main experiment.

### 🔲 Phase 5 — Baseline comparison

Baselines: (1) rule‑based FCFS, (2) flat MAPPO global critic, (3) IPPO, **plus (4) FCFS + rule fallback (NEW)** as a non‑learning strong baseline.

### 🔲 Phase 6 — Ablation and learned Layer 3

Add: with/without curriculum lambda, with/without bound_slack reward, K_max variants, finer dwell grid, station critic, and learned Layer 3 Worker. The learned Worker must implement the same execution-controller interface and consume `ExecutionTarget` rather than requiring Layer 2 to emit SUMO-specific details.

---

## CONSTRAINTS SUMMARY (Updated)

| constraint                      | enforced by              |
|---------------------------------|--------------------------|
| platform single‑occupancy       | action mask + resolver   |
| conflicting throat routes       | resolver                 |
| corridor same‑direction headway | Layer 3 only (not resolver — double‑track, no opposing conflict) |
| minimum dwell time              | action mask              |
| t_arr_min / t_dep_max           | action mask              |
| minimum corridor headway        | Layer 3 safety guard     |
| hard boundary infeasible condition | hold/replan or terminate; never grant out-of-bound plan |
| platform direction compatibility| action mask              |
| route authorization             | Layer 3 route check      |

---

## ACCEPTANCE CHECKLIST (Updated)

- [ ] All trains have clear control status transitions (ACTIVE/MONITORING/NOMINAL)
- [ ] Episodes terminate correctly on success, truncation, or deadlock
- [ ] Disturbances occur at least once per episode on average
- [ ] Rewards are not dominated by a single term
- [ ] Critic inputs are aligned with the corresponding reward components
- [ ] Curriculum schedule switches from individual to team focus
- [ ] No fallback grants a Layer 1 hard-bound violation
- [ ] Scalability: observation dimension is configurable, not hard‑coded
- [ ] SUMO topology changes are expressible through `sumo/` and `configs/` without Python source edits

---

## REWORK ASSESSMENT FOR CURRENT PROGRESS

**Your current status:** Phase 0/1/2 completed. Phase 3 (SUMO env + disturbance loop) is the next step.

**What needs to be changed (rework):**

| Module | Change Required | Effort |
|--------|-----------------|--------|
| `dispatch_bounds.py` | No change | None |
| `resource_manager.py` / `resolver.py` | Add fallback degradation | Light |
| `action_mask.py` | Keep baseline dwell enum `[0, 60, 120]` | None |
| `disturbance.py` | Add Layer 2 status and configurable disturbance injection | New module |
| `sumo_env.py` | Replace one-step heuristic skeleton with L2 action-driven env | Moderate |
| `observation_builder.py` | Update neighbour selection, add bound_slack, keep 74-dim layout | Light |
| `reward.py` | Add local/global reward components and metrics | Light |
| `executor.py` (to be built) | Add safety guard layers | Already planned |
| `mappo.py`, `actor_critic.py` (Phase 4) | Dual‑critic with loss‑level fusion | Not yet started |

**Conclusion:** No completed Phase 0-2 module requires redesign. The important correction is to keep all hard-bound handling fail-closed, make executor/SUMO route lookup config-driven, and replace the Phase 3 skeleton with a true L2 action-driven training environment.
