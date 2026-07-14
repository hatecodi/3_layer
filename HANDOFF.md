# Handoff — 2026-07-15 00:30

## 项目状态

三层铁路 MARL 框架，所有核心功能完成。最后一次完整训练的 clean + delay 两个版本均已完成。

## 训练产物

| 版本 | 路径 | 结果 |
|------|------|------|
| **Clean** | `output/five_station_hierarchical/20260714_231614/` | Phase C: dly=1.3s, ont=100% |
| **Delay** | `output/five_station_hierarchical/20260714_232845/` | Phase C: dly 1.2-45s（取决于 episode 是否触发 delay） |

CSV 已同步到 `output/figures/` 和 `output/figures/VT_rank/`。

## 最终 L3 reward

```python
# src/env/reward.py — compute_l3_reward()
total_delay = headway_delay_sec + throat_delay_sec
late = max(arrival_deviation_sec, 0.0)
effective_recovery = min(recovery_used_sec, total_delay + late)
fuel_penalty = 0.003 * recovery_used_sec
return (
    -0.02 * headway_delay_sec    # corridor headway penalty
    - 0.02 * throat_delay_sec    # throat headway penalty
    - 0.01 * late                # late arrival penalty (early = no penalty)
    + 0.01 * effective_recovery  # recovery reward (only when delay exists)
    - fuel_penalty               # small cost for FAST (makes CRUISE default)
)
```

## Clean 环境 L3 速度分布

CRUISE=22, FAST=39, SLOW=11 — 三种模式均被使用。

## 关键修复（本次会话）

1. L3 obs speed lookup: `train_class` → `priority`
2. `dwell_min` 全部 5s
3. Intermediate station `t_arr_plan` floor（防早发）
4. Persistent platform calendar（跨 step 站台安全）
5. L3 reward: `abs(arrival_dev) → max(arrival_dev, 0)`（早到不罚）
6. L3 reward: `fuel_penalty = 0.003 * recovery`（FAST 有代价）
7. L3 obs 加 `throat_gap`（10 维）
8. L3 physics 加 `throat_headway_sec=6.0`
9. CSV 字段一致性修复

## 未完成

- VT 扰动训练未收敛（建议降低 VT 强度或 curriculum）
- VT_rank 对比数据用旧 L2 跑的，需要用新 L2 ckpt 更新
- 画图脚本可能需要更新路径

## 下一 session 建议

1. 用新版 L2 ckpt 更新 VT_rank 对比数据
2. 调低 VT 强度（prob 0.25, slots=1）尝试收敛
3. 或者接受无扰动为主结果，VT 作为 robustness discussion

## 关键命令

```bash
# 安全审查
python scripts/validate_l3_schedule.py \
  --l2-ckpt output/five_station_hierarchical/20260714_231614/checkpoints/best.pt \
  --l3-ckpt output/five_station_hierarchical/20260714_231614/checkpoints/best.pt \
  --n-episodes 10

# 训练
python -m src.experiment.run_train_hierarchical --env five_station          # clean
python -m src.experiment.run_train_hierarchical --env five_station --delay  # +initial delay

# VT 鲁棒性对比
python scripts/compare_vt.py
```
