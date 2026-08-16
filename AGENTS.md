# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Before you do anything else

1. Read `coding_log` to see recent changes and current project status.
2. Read `.claude/skills/coding-workflow.md` — this is the mandatory workflow discipline.
3. Write a plan to `.claude/plans/` before making code changes.
4. Append to `coding_log` after every completed task.

## Project overview

Three-layer hierarchical MARL framework for railway dispatch on a five-station (A–E) double-track corridor, scaled to 15 trains (`configs/five_station_15t/`). Research question: can a three-layer design (dispatch bounds → MAPPO scheduling → learned corridor speed control) scale better than flat MAPPO while maintaining low delay and zero hard-constraint violations?

- **Layer 1** (rule-based, fixed): dispatch bounds from `configs/{scenario}/timetable.yaml` — hard time envelopes per train
- **Layer 2** (learned, MAPPO): scheduling within Layer 1 bounds — platform + entry timing (dwell fixed at min)
- **Layer 3** (learned, per-corridor worker): sequential per-section speed control (CRUISE/FAST/SLOW) with moving-block headway as a hard boundary — `l3_worker.py` + `l3_sim.py`

## Key commands

```bash
# Run unit tests (15 tests for Phase 1+2)
python -m pytest tests/test_phase12.py -v

# Validate config loading
python -m src.experiment.scenario_loader

# Hierarchical three-phase training (L2 → L3 → joint)
python -m src.experiment.run_train_hierarchical --env five_station

# Train L3 worker only (frozen L2) — current L3-only workflow
python -m src.experiment.run_train_l3_only --ckpt <l2.pt> --episodes 2000

# L3 convergence protocol (multi-seed) + paper figure
python scripts/train_l3_convergence.py --l2 <l2.pt> --out output/figures/l3v3_conv.csv
python scripts/plot_l3_conv_protocol.py --csv output/figures/l3v3_conv.csv

# L2/L3 schedule safety audit
python scripts/validate_l3_schedule.py

# Heuristic baseline (no learning)
python -m src.experiment.run_heuristic

# Legacy two-station SUMO executor baseline (Phase 3 era — not the main path)
python -m src.experiment.run_executor_baseline
```

SUMO + `SUMO_HOME` are only needed for the legacy two-station executor baseline.

## Architecture

### Config-driven design (no hardcoded IDs)

All train IDs, station names, platform IDs, and corridor references come from config files — never hardcoded in Python. Adding a train = YAML + `.rou.xml` only. Adding a station = net.xml + resource YAML + timetable YAML only.

### Data flow (per decision step)

1. `five_station_env.py` pulls train states → `disturbance.py` injects bunched origin delay → scheduling loop starts
2. Triggered trains get ~29-dim observations via `observation_builder.py`
3. `action_mask.py` builds factored `(platform, entry)` action mask (Layer 1 hard bounds enforced pre-sampling)
4. Policy selects `(platform_idx, entry_idx)` → `action_to_request()` → `WindowRequest`
5. `resource_manager.py` resolves conflicts (deterministic priority-weighted resolver, fallback assignment) → `GrantResult`
6. Granted trains transit the corridor: L3 worker (`l3_worker.py`) picks CRUISE/FAST/SLOW per block section in `l3_sim.py` (moving-block headway is a hard boundary); the actual arrival feeds the next station's scheduling

### Key modules

| Module | Status | Purpose |
|--------|--------|---------|
| `src/experiment/scenario_loader.py` | DONE | Loads + validates all config files |
| `src/control/dispatch_bounds.py` | DONE | Layer 1 per-train bounds wrapper |
| `src/control/action_mask.py` | DONE | Action space, `is_legal_action`, factored (platform, entry) mask |
| `src/control/resource_manager.py` | DONE | `ResourceCalendar`, conflict resolver, `assign_fallback` |
| `src/control/l3_sim.py` | DONE | L3 corridor physics — per-section speed profiles, moving-block headway, throat headway |
| `src/control/l3_sa.py` | DONE | SA per-transit baseline (eval only) |
| `src/control/executor.py` | LEGACY | Rule-based SUMO executor via TraCI — two-station baseline only |
| `src/control/execution_target.py` | LEGACY | L2 → L3 interface dataclass — two-station baseline only |
| `src/env/five_station_env.py` | DONE | Five-station scheduling env — L2/L3 loop, team reward, min_gap safety |
| `src/env/disturbance.py` | DONE | Bunched origin-delay injection, deadlock detection |
| `src/env/observation_builder.py` | DONE | Config-driven ~29-dim L2 obs + 10-dim L3 section obs |
| `src/env/reward.py` | DONE | Layered reward, `compute_l3_terminal_reward`, episode team reward |
| `src/marl/actor_critic.py` | DONE | `Layer2ActorCritic` (obs→platform/entry), local + global critic |
| `src/marl/l3_worker.py` | DONE | `CorridorActor`/`CorridorCritic` + `L3WorkerTrainer` (PPO), 10-dim obs |
| `src/marl/mappo.py` | DONE | MAPPO update loop with action masks |
| `src/marl/rollout_buffer.py` | DONE | RolloutBuffer with GAE |
| `src/experiment/run_train_hierarchical.py` | DONE | Three-phase training (L2 → L3 → joint) |
| `src/experiment/run_train_l3_only.py` | DONE | Freeze L2, train L3 worker only |
| `src/experiment/run_flat_mappo.py` | DONE | Flat MAPPO baseline (no L3) |
| `src/experiment/run_heuristic.py` | DONE | CLI heuristic baseline |
| `src/experiment/run_executor_baseline.py` | LEGACY | CLI: heuristic grants → SUMO execution (two-station) |

### What needs to be built next

- `src/experiment/run_eval.py` — consolidated held-out evaluation (currently `scripts/eval_protocol.py`)
- `src/experiment/metrics.py` — consolidated CSV/JSON export helpers
- Corridor speed-restriction incident disturbance (暴雨限速) — makes L3 coordination matter (see coding_log + plan_corridor_incident_scenario.md)

### Key design invariants

1. **Hard boundaries via masking, not reward**: `t_arr_min`, `t_dep_max`, dwell_min, route compatibility enforced by action mask (illegal actions get -inf logits). No boundary-violation terms in reward.
2. **Local critic**: critic input is `(k_local+1)×obs_dim` (~29 obs, k_local=1 → 58 dims), not global — O(1) scaling with train count.
3. **Fallback must re-verify bounds**: `assign_fallback` uses `RebuildFn` callback that creates fresh `WindowRequest` with correct resource IDs and re-checks all Layer 1 bounds.
4. **Corridor headway is Layer 3 only**: corridor is NOT in `_resource_windows` — the resolver only owns station resources (platform, in-throat, out-throat). Minimum tracking headway enforced by the L3 sim's moving-block headway (never a reward term).
5. **NO_LEGAL_ACTION → hold/replan**: if all actions masked, train is held and replanned next step; never sample from policy.

### Network (training scale)

Five stations A–E on a double-track corridor, 4 platforms each (D has 2). 15 trains in `configs/five_station_15t/` (11 base + 4 scaled copies from `scripts/generate_scaled_timetable.py`) plus config-driven visitor trains. Time compression: real seconds / 25 = sim seconds.

### Conflict resolution rule (deterministic)

Priority (1 beats 3) → higher delay wins tie → train_id lexicographic wins. Losers get earliest feasible fallback on alternate platform, or delayed on same platform.
