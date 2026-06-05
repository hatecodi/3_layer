# Plan: 完成 Phase 3 — disturbance, observation, reward, env L2 loop
Date: 2026-06-03
Status: done

## 现状

- `disturbance.py` ✅：DisturbanceInjector + ControlStatus 三态机
- `observation_builder.py` ✅：74-dim 固定观测
- `reward.py` ✅：分层 reward，Metrics 含出发+到达延误
- `sumo_env.py` ❌：骨架仅单步（max_steps=1），无列车生命周期

## What

重写 `sumo_env.py` 的核心逻辑，使其成为完整的多步调度环境：

1. **列车生命周期**：reset 时所有列车进入 pending 集合；每次 step 后 granted 列车移出
2. **NOMINAL 路径**：未触发 L2 的列车直接用 timetable plan（t_arr_plan/t_dep_plan + 第一候选站台 + dwell_min），不走 policy
3. **多步决策**：held/replan 列车在下个 step 重试；新 step 可能注射新 disturbance 导致新触发
4. **Episode 终止**：SUCCESS（全部 scheduled）/ TRUNCATED（超时）/ DEADLOCK（连续无合法动作超过阈值）
5. **死锁检测**：所有 pending 列车 mask 全为 NO_LEGAL_ACTION 且持续 DEADLOCK_THRESHOLD=15 步

## Alignment check
- [x] No hardcoded IDs: 已确认
- [x] Observation 74-dim: 不变
- [x] Action mask enforces bounds: 不变
- [x] Fallback re-verifies bounds: 不变
- [x] Config-only expansion: 不变

## Pass criteria
- 8 列车单步全部 NOMINAL → 1 步 episode SUCCESS
- 注入 disturbance 触发 L2 → 多步 episode
- deadlock 条件触发 → episode 终止为 deadlock
- python -m pytest tests/test_phase12.py 仍 15/15
