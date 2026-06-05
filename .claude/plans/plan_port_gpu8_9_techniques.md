# Plan: Port GPU_8_9 techniques to five-station MAPPO

Date: 2026-06-04
Status: done

## Source
GPU_8_9 project (D:\sumo_pract\GPU_8_9) — MADDPG + SUMO high-speed rail delay management.
8 trains, 8 stations, discrete action space.

## What to port (priority order)

### P0: Dynamic progressive delay penalty
**File:** `src/env/reward.py`
Current: `-0.05 * dep_delay * prio_weight` (linear, same marginal cost for 100s vs 500s)
Target: `-0.05 * dep_delay * prio_weight * (1 + delay // 300)`
Every 300s (5 min) of delay doubles the penalty. Severe delays hurt more.

### P1: BatchNorm on critic inputs
**File:** `src/marl/actor_critic.py`
Add `BatchNorm1d` to LocalCritic and GlobalCritic first layer.
Dimension differences from GPU_8_9:
- Their NormCritic: 96-dim → BN → FC(128)
- Our LocalCritic: 230-dim → BN → FC(128)
- Our GlobalCritic: 14-dim → BN → FC(128) (14-dim BN might not help much, but consistent)

Key: use `affine=False` (no learnable scale/shift), same as their design.
This normalises heterogeneous input scales (time secs, delay secs, priority 1-3,
platform bits 0/1, conflict bits 0/1) without adding parameters.

### P2: Exploration entropy decay schedule
**File:** `src/experiment/run_train.py`
Current: fixed `entropy_coef = 0.001`
Target: linear decay `0.01 → 0.001` over first 30% of episodes, then flat at 0.001.
More exploration early, more exploitation later.

## What NOT to port (and why)

| Technique | Reason to skip |
|-----------|---------------|
| Gumbel-Softmax | MAPPO uses Categorical logprob, not deterministic policy gradient |
| Safety distance penalty | L3 concern, not L2 scheduling |
| OU Noise | MAPPO uses stochastic policy, not deterministic + noise |
| Their action masking | We already have compute_joint_action_mask + masked_fill(-inf) |

## Alignment check
- [x] No hardcoded IDs: OK (config-driven coefficients)
- [x] Observation config-driven: OK
- [x] Action mask enforces bounds: OK
- [x] Local critic input = (K+1)×obs_dim: OK
- [x] No hardcoded dimensions: OK (BatchNorm adapts to input dim)

## Pass criteria
1. Dynamic penalty: trains with 500s delay get ~3× penalty of 100s delay
2. BatchNorm: critic loss converges faster (fewer episodes to stable value)
3. Entropy schedule: early episodes explore more, later episodes exploit more
