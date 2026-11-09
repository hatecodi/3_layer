# Plan: Phase B resume (continue learned-L3 training from 2k checkpoint)
Date: 2026-11-07
Status: done

## What
1. `src/experiment/run_train_hierarchical.py`
   - add CLI arg `--l3-ckpt PATH` (optional)
   - thread it into `train_hierarchical(..., l3_ckpt=None)`
   - after constructing `CorridorActor` / both `CorridorCritic`s, load
     `state["l3_actor"]`, `state["l3_critic_local"]`, `state["l3_critic_global"]`
     from the checkpoint (same key layout written by `final.pt`/`best.pt`)
   - print loaded episode number for traceability
2. Run a short smoke (20 eps) to verify load + relevance/cap still active.
3. Launch Phase B continuation: `--phases B --total-episodes 4000`
   (`--l2-ckpt` = frozen v7 seed0 L2 final, `--l3-ckpt` = phaseB 2k final),
   incident, relevant, yield-cap, tail-coef 0, seed 0, eval-every 500 / eval-n 6.
4. End-to-end 180-ep eval of the resumed final/best checkpoint with learned L3,
   paired against `output/11_06/eval_v4_heur_sa_180` and v7-L2+SA-L3.
5. Append coding_log with results + save locations (per project rule).

## Why
Phase B (2k) finished, but the learned L3 is undertrained: end-to-end ontime
63.2% vs 75% for SA-L3 (180 paired incident eps), with only ~2 L3 updates per
episode. The user chose to extend Phase B to 4-6k episodes rather than restart.
Resume support avoids discarding the 2k-episode L3 progress, since the trainer
currently only loads L2 checkpoints.

## Alignment check
- [x] No hardcoded IDs: N/A (checkpoint plumbing only)
- [x] Observation dims: unchanged (L3 obs dim fixed at L3_OBS_DIM=10)
- [x] Action mask enforces bounds: unchanged
- [x] Local critic: unchanged
- [x] Fallback re-verifies bounds: unchanged
- [x] Config-only expansion: unchanged

## Pass criteria
- Smoke run loads L3 without shape errors and logs `Loaded L3 from ...`.
- 4k continuation completes, `final.pt` saved; held-out arr_ontime recovers
  toward the SA-L3 level (target >= 0.60 on training held-out, up from 0.455).
- Final 180-ep paired eval reported (ontime, mean, tail10, p90, max, pw-delay),
  coding_log updated with all save paths.
