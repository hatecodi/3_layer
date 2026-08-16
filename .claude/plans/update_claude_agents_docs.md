# Update CLAUDE.md + AGENTS.md to current code state

## Why

Both files describe the two-station / L3-rule-SUMO era. Verified stale:
- `configs/timetable_layer1.yaml`, `src/env/sumo_env.py`, `scripts/validate_sumo.py`
  — files do NOT exist.
- `ExecutionTarget → executor.py → SUMO TraCI` data-flow step — dead code (only
  legacy `run_executor_baseline.py` uses it).
- obs "74-dim, K_max=5" → actual ~29 dims, k_local=1.
- `.Codex/` referenced by AGENTS.md — directory does not exist.

## Edits

### CLAUDE.md
1. Project overview: "two-station double-track" → five-station A–E (+ 15t scaling
   via `configs/five_station_15t/`). Layer 1 source: `configs/timetable_layer1.yaml`
   → `configs/{scenario}/timetable.yaml`. Layer 3: rule-based SUMO → learned
   per-section speed worker (L3 v3, `l3_worker.py` + `l3_sim.py`).
2. Key commands: drop `scripts/validate_sumo.py`; fix test count 10 → 15; add
   current commands (`run_train_hierarchical`, `run_train_l3_only`,
   `scripts/train_l3_convergence.py`, `scripts/validate_l3_schedule.py`); mark
   `run_executor_baseline` legacy.
3. Data flow: `sumo_env.py` → `five_station_env.py`; step 6 `ExecutionTarget →
   SUMO TraCI` → L3 worker per-section speed decision with moving-block headway.
4. Module table: drop `sumo_env.py`; mark `executor.py`/`execution_target.py`
   legacy (two-station only); add `five_station_env.py`, `l3_worker.py`,
   `l3_sim.py`, `l3_sa.py`, `run_train_hierarchical.py`, `run_train_l3_only.py`,
   `run_flat_mappo.py`.
5. obs dims in invariants: (K_max+1)×74=444 → ~29 obs, k_local=1, critic (k_local+1)×29.
6. "To be built next": mappo.py/rollout_buffer.py done; replace with actual next
   (run_eval.py, metrics.py, corridor-incident scenario).
7. Network section: two-station → five_station_15t summary.

### AGENTS.md
Mirror CLAUDE.md content edits; fix workflow references `.Codex/` → `.claude/`
(real dir), test count 10 → 15.

## Verify
- `python -m pytest tests/test_phase12.py -v` still listed as a valid command.
- No remaining references to `sumo_env.py`, `timetable_layer1.yaml`,
  `validate_sumo.py`, or `.Codex/` in either file.
