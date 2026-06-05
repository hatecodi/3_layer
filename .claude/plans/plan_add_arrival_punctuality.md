# Plan: 添加到达正点/晚点统计
Date: 2026-06-03
Status: in_progress

## What
- `reward.py`: `Metrics` 增加 `total_arrival_delay`、`mean_arrival_delay`；`compute_layered_rewards` 的 local reward 增加到达延误惩罚

## Why
当前只有出发延误 `max(0, t_out_start - t_dep_plan)`，缺少到达延误 `max(0, t_in_start - t_arr_plan)`。用户定义了正晚点规则 — 到达晚于 arr_plan 即为晚点 — 需要在 reward 中体现。

## Alignment check
- [x] No hardcoded IDs: N/A（纯计算逻辑）
- [x] Observation 74-dim: N/A
- [x] Action mask enforces bounds: N/A
- [x] Local critic 444-dim: N/A
- [x] Fallback re-verifies bounds: N/A
- [x] Config-only expansion: N/A

## Pass criteria
- `Metrics` 包含 arrival delay 相关字段
- local reward 包含到达延误项
- 现有 10 个测试仍然通过
