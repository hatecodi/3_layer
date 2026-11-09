# Plan: L3 learned worker on frozen base L2 (closure scenario)

Date: 2026-11-09
Status: in_progress

## What

1. Freeze **seed1** base L2 (`training_closure_l2_2k_seed1/.../final.pt`).
2. Run Phase B (L2 frozen, learned L3) for 2000 episodes with the closure
   disturbance, relevant L2 gating, yield cap, alpha=0.10.
3. Evaluate end-to-end on 180 closure episodes (same seeds 900000):
   - learned L3
   - **SA-online** (`--sa-online-only`): partial-observation, per-section greedy
     from current state only — NO global/joint prediction, the fair match for
     the learned worker
   - SA-nofc (optional): no incident forecast
   - full SA (300 iters, oracle forecast) as reference only
4. Check training stability (success, deadlock, reward/arr_ontime curves).

## Why

Full-information SA is an offline oracle (joint annealing over the whole
profile, sees the front occupant's full trajectory). Comparing the learned L3
worker to it understates the learned layer. SA-online is documented as the
apples-to-apples baseline (same 10-dim partial observability as the worker).
Learned L3 < full SA is expected; the question is whether it is competitive
with no-global-prediction SA.

## Alignment check
- [x] No hardcoded IDs: scenario/config driven.
- [x] L3 obs unchanged (closure blocks entry upstream; headway/throat effects
      already in the 10-dim per-section obs).
- [x] Hard boundaries unchanged (mask/moving block).
- [x] L2 frozen in Phase B; relevant gating still applied.
- [x] Config-only scenario expansion.

## Pass criteria
- Phase B 2000/2000 success, no deadlock, stable arr_ontime.
- Paired eval learned L3 vs SA-online on the same 180 closure seeds; report
  ontime/mean/tail10/p90/max, closure overtakes, affected/DDT.
- No claim against full SA until SA-online comparison is in.
