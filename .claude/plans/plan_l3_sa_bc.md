# Plan: SA-BC baseline for the L3 corridor worker

Date: 2026-11-09
Branch: l3-sa-bc
Status: in_progress

## What

1. Add an optional BC recorder to the env: when the L3 executor is **SA-online**
   (partial observation, no global forecast) and `record_l3_bc=True`, record the
   10-dim per-section observation and the SA-online speed mode for every section.
2. New `scripts/bc_pretrain_l3.py`:
   - roll out the **frozen seed1 L2** (base relevance, closure, yield-cap) with
     SA-online as the L3 executor, collecting `(obs_10d, mode)` labels;
   - train `CorridorActor` with cross-entropy;
   - save a 2-critic checkpoint (`l3_actor`, `l3_critic_local`,
     `l3_critic_global`) loadable by eval/Phase B.
3. Evaluate the BC-only L3 end-to-end on 180 closure episodes (same seeds,
   seed1 L2) and compare with:
   - current learned L3 (Phase B final)
   - SA-online (teacher)
   - full SA reference
   Focus: Pareto of on-time vs tail (tail10/p90/max), since the current learned
   L3 trades on-time for tail.

## Why

Learned L3 is -3.4pt on-time vs SA-online but better on tail/mean. Pure BC will
approach the teacher, so this baseline quantifies how much tail is lost when
starting from SA-online. It is the reference for later mixed-BC/tail-aware PPO.

## Alignment check
- [x] No hardcoded IDs; config driven.
- [x] L3 obs unchanged (10-dim); no env physics change.
- [x] Hard safety (mask/moving block) unchanged.
- [x] BC is offline initialization only; PPO semantics untouched.
- [x] Checkpoint format matches the 2-critic loader.

## Pass criteria
- BC val mode accuracy high (>0.9) and BC-only eval close to SA-online.
- Quantified Pareto: ontime and tail10/p90/max of BC vs SA-online vs current
  learned L3.
- No deadlock / no postcheck rejects in the BC eval.
