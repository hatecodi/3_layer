# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Before you do anything else

1. Read `coding_log` to see recent changes and current project status.
2. Read `.Codex/skills/coding-workflow.md` — this is the mandatory workflow discipline.
3. Write a plan to `.Codex/plans/` before making code changes.
4. Append to `coding_log` after every completed task.

## Project overview

Three-layer hierarchical MARL framework for railway dispatch on a two-station double-track corridor. Research question: can a three-layer design (dispatch bounds → MAPPO scheduling → rule-based execution) scale better than flat MAPPO while maintaining low delay and zero hard-constraint violations?

- **Layer 1** (rule-based, fixed): dispatch bounds from `configs/timetable_layer1.yaml` — hard time envelopes per train
- **Layer 2** (learned, MAPPO): scheduling within Layer 1 bounds — platform, entry timing, dwell decisions
- **Layer 3** (rule-based, fixed): SUMO TraCI execution of granted schedules

## Key commands

```bash
# Run unit tests (10 tests for Phase 1+2)
python -m pytest tests/test_phase12.py -v

# Run heuristic baseline (Phase 2 — no learning)
python -m src.experiment.run_heuristic

# Validate config loading
python -m src.experiment.scenario_loader

# SUMO sanity check (Phase 0)
python scripts/validate_sumo.py

# SUMO executor baseline (Phase 3 — heuristic grants → SUMO via TraCI)
python -m src.experiment.run_executor_baseline
```

Requires SUMO installed with `SUMO_HOME` environment variable set (TraCI import falls back to `$SUMO_HOME/tools`).

## Architecture

### Config-driven design (no hardcoded IDs)

All train IDs, station names, platform IDs, and corridor references come from config files — never hardcoded in Python. Adding a train = YAML + `.rou.xml` only. Adding a station = net.xml + resource YAML + timetable YAML only.

### Data flow (per decision step)

1. `sumo_env.py` pulls train states → `disturbance.py` injects delays → decides Layer 2 activation
2. Triggered trains get 74-dim observations via `observation_builder.py`
3. `action_mask.py` builds joint action mask (Layer 1 hard bounds enforced pre-sampling)
4. Policy selects `(platform, entry_choice, dwell_extend)` → `action_to_request()` → `WindowRequest`
5. `resource_manager.py` resolves conflicts (deterministic priority-weighted resolver, fallback assignment)
6. Resolved grants → `ExecutionTarget` → Layer 3 `executor.py` → SUMO TraCI

### Key modules

| Module | Status | Purpose |
|--------|--------|---------|
| `src/experiment/scenario_loader.py` | DONE | Loads + validates all config files |
| `src/control/dispatch_bounds.py` | DONE | Layer 1 per-train bounds wrapper |
| `src/control/action_mask.py` | DONE | Action space, `derive_schedule`, `is_legal_action`, mask |
| `src/control/resource_manager.py` | DONE | `ResourceCalendar`, conflict resolver, `assign_fallback` |
| `src/control/executor.py` | DONE | Rule-based SUMO executor via TraCI |
| `src/control/execution_target.py` | DONE | Layer 2 → Layer 3 interface dataclass |
| `src/env/sumo_env.py` | PARTIAL | Gym-style env (scheduling level — no live SUMO yet) |
| `src/env/disturbance.py` | DONE | `DisturbanceInjector`, `ControlStatus` (nominal/L2 active/monitoring) |
| `src/env/observation_builder.py` | DONE | 74-dim fixed obs with K_max=5 neighbor padding |
| `src/env/reward.py` | DONE | Hybrid individual + team reward, `Metrics` dataclass |
| `src/marl/actor_critic.py` | DONE | `MaskedActor` (74→hidden→action), `LocalCritic` ((K+1)×74→1), `GlobalCritic` |
| `src/experiment/heuristic_planner.py` | DONE | Reusable heuristic plan builder |
| `src/experiment/run_heuristic.py` | DONE | CLI baseline |
| `src/experiment/run_executor_baseline.py` | DONE | CLI: heuristic grants → SUMO execution |

### What needs to be built next

- `src/marl/mappo.py` — PPO update loop with action masks
- `src/marl/rollout_buffer.py` — stores (obs, action, mask, reward, value, logprob)
- `src/experiment/run_train.py` / `run_eval.py` — training and evaluation scripts
- `src/experiment/metrics.py` — CSV/JSON export

### Key design invariants

1. **Hard boundaries via masking, not reward**: `t_arr_min`, `t_dep_max`, dwell_min, route compatibility enforced by action mask (illegal actions get -inf logits). No boundary-violation terms in reward.
2. **Local critic**: critic input is `(K_max+1)×74 = 444` dim, not global — O(1) scaling with train count.
3. **Fallback must re-verify bounds**: `assign_fallback` uses `RebuildFn` callback that creates fresh `WindowRequest` with correct resource IDs and re-checks all Layer 1 bounds.
4. **Corridor headway is Layer 3 only**: corridor is NOT in `_resource_windows` — the resolver only owns station resources (platform, in-throat, out-throat). Minimum tracking headway enforced by execution-layer speed control.
5. **NO_LEGAL_ACTION → hold/replan**: if all actions masked, train is held and replanned next step; never sample from policy.

### Network (training scale)

Two stations A/B connected by 1200m double-track corridor, 3 platforms each, 2 conflicting throat routes each. 8 scheduled trains + 1 conflict-test train (K999). All physical and time parameters scaled from real values (~4-10× compression).

### Conflict resolution rule (deterministic)

Priority (1 beats 3) → higher delay wins tie → train_id lexicographic wins. Losers get earliest feasible fallback on alternate platform, or delayed on same platform.
