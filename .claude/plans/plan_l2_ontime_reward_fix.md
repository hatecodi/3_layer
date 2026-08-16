# Plan: L2 reward 硬准点奖励修复 + 干净无干扰重训

Date: 2026-08-13
Status: draft

## Context

Phase A L2 训练(20t, --delay)reward 从 -17.2 爬到 -15.5 但发车准点率平在 0.55 不涨。评估发现 learned L2 在发车(54.9%)和到站(40.3%)准点上都低于 heuristic(59.7%/44.0%)。

根因:L2 reward 的准点奖励是 `ONTIME_BONUS * exp(-dep_delay/60)` **连续衰减**——策略把延误从 40s 压到 20s(reward 涨)但从不跨过 15s 准点门槛(ontime 平)。reward 优化"少延误"而非"准点",与评估指标(15s 硬阈值, `ONTIME_THRESHOLD_SEC=15.0`, `n_ontime_dep = d ≤ 15`)错位。

## What

1. `src/env/reward.py`:
   - 新增 `ONTIME_HARD_BONUS = 2.0`。
   - 在 `local`(granted 分支)加硬准点项:`if dep_delay <= ONTIME_THRESHOLD_SEC: local += ONTIME_HARD_BONUS * prio_weight`。
   - 保留连续 exp 奖励(窗内平滑梯度),硬触发让 15s 门槛成为显式目标。
2. **干净无干扰重训** L2(20t, 无 --delay/--vt):
   `python -m src.experiment.run_train_hierarchical --env five_station_20t --phases A`
3. 重评估 learned vs heuristic(干净设置, eval 不带 --delay)。

## Alignment check
- [x] 无硬编码 ID:reward 改动通用
- [x] 观测/动作不变
- [x] 硬边界仍由 action mask 强制(reward 不碰边界)
- [x] 只改 reward 项 + 重训,不碰 resolver/env

## Verification
1. reward 修改后 43 测试通过(硬奖励项不改其它行为)
2. 干净训练收敛:reward 上升且发车准点率上升(>0.55, 向 heuristic 的 ~60% 靠近)
3. eval:learned ≥ heuristic 发车准点(不再落后)

## Files
- `src/env/reward.py`
- 新训练 run 目录 + eval

## Pass criteria
- 干净训练 reward 收敛且发车准点率随训练上升
- learned L2 在干净 20t 上 ≥ heuristic 发车准点(反向结论翻正)
