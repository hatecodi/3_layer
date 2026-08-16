# Plan: L2 reward 加实际到站准点项(跳出发车准点平台)

Date: 2026-08-13
Status: draft

## Context

硬准点奖励后,L2 干净训练 reward 收敛(-12.6→-9.4,斜率≈0)但发车准点只到 0.61,低于 heuristic 0.63;learned ≈ heuristic(噪声内)。平台原因是:**L2 reward 只优化发车准点,而论文 KPI 是到站准点(L3 决定)**。L2 无法通过发车选择影响"到站是否准点"→ 学不到与 L3 的协调(发车要早到足以消化走廊 headway)。

## What

1. `src/env/reward.py`:
   - 新增 `ARR_DELAY_COEF = 0.02`。
   - `compute_layered_rewards` 加参数 `realized_arr_delays: Optional[Dict[str,float]]`,在 granted 分支加:
     `local += -ARR_DELAY_COEF * realized_arr * prio_weight`(realized_arr = 本次过轨实际到站延误 max(0, arr_delay))。
   - 保留硬准点奖励 + 连续 exp。
2. `src/env/five_station_env.py` step(): 从 `self._l3_transit_events` 构建 `{train_id: max(0, arr_delay)}`,传给 `compute_layered_rewards`。
3. 干净重训 L2(20t, 无 --delay/--vt):`run_train_hierarchical --env five_station_20t --phases A`。
4. 重评估 learned vs heuristic(到站准点为主指标)。

## Alignment check
- [x] 无硬编码 ID
- [x] 观测/动作不变
- [x] 硬边界仍由 mask 强制
- [x] 只改 reward 项 + env 传参 + 重训

## Verification
1. 43 测试通过。
2. 干净训练收敛且**到站准点率随训练上升**(不只是发车准点)。
3. eval:learned ARR ontime ≥ heuristic(论文 KPI 翻正)。

## Files
- `src/env/reward.py`, `src/env/five_station_env.py`
- 新训练 run + eval

## Pass criteria
- learned L2 到站准点率 ≥ heuristic(ARR 45.8% 基准),或至少发车+到站双双 ≥ heuristic。
- 若仍不超:诚实报告走廊 headway 物理封顶,L2 价值有限,转 re-dispatch pivot。
