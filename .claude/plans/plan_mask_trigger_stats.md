# Plan: 掩码触发次数统计（论文"构造性安全保证"证据）

Date: 2026-09-18
Status: in_progress

## What
统计动作掩码在真实决策路径上**触发**的次数与按约束类型的分解，用于论文论证
"硬边界由掩码构造性保证、零违反"，而非靠奖励惩罚。

改动（3 个文件）：
1. `src/control/action_mask.py`
   - 新增 `illegal_reason(train, station_id, action, t_now, travel_time_to_throat, cfg, bounds_override=None) -> Optional[str]`：
     返回 `None`（合法）或短原因码（`no_station` / `platform_not_candidate` / `unknown_platform` /
     `platform_not_through` / `passthrough_dwell` / `arr_min` / `dep_max` / `dwell_min`）。
   - `is_legal_action` 改为薄包装 `return illegal_reason(...) is None`（**行为不变**）。
   - `compute_joint_action_mask(..., reason_counter=None)`：可选 sink，逐个非法动作累加原因计数，
     避免二次枚举。
2. `src/env/five_station_env.py`
   - `__init__`/`reset`：`self._mask_stats = defaultdict(int)`（reset 清零）。
   - `_policy_action(..., mask_stats=None)`：把 sink 透传给 `compute_joint_action_mask`；
     同时累计 `decisions`(决策机会) / `candidates`(枚举候选动作数) / `masked_out`(被掩掉数) /
     `no_legal_action`(全掩 → hold)。
   - `_build_requests`：调用点传入 sink；并统计 `postcheck_reject`（构造后仍越界的**冗余**二次校验拒绝数，
     预期恒 0 —— 掩码完整性的自洽检查）。
   - 在 episode 末 info 中暴露 `"mask_stats": dict(self._mask_stats)`。
3. `scripts/eval_protocol.py`
   - `run_episode` 累加每步 `info["mask_stats"]` 并在返回 dict 中带出。
   - main 汇总 per-mode，打印摘要并写 `mask.csv`（列：decisions, candidates, masked_out,
     masked_pct, no_legal_action, postcheck_reject + 各 reason 列）。

## Why
论文第 6.6 节需要"掩码不是摆设"的定量证据。零训练成本：掩码统计不依赖学习结果，
在既有 ckpt 上跑现有 eval 协议即可产出。用户已同意先做统计、暂不做 no-mask 消融
（理由：no-mask 没有中性对照定义，易被判 straw man）。

## 统计口径（论文需明确写入）
- **决策机会 (decision opportunity)**：某列车在某站被策略要求选择动作的一次事件
  （`_build_requests` 中对每辆 pending 车调用 `_policy_action` 一次）。
- **候选动作**：`candidate_actions` 枚举的 (platform × entry × dwell=0) 组合数。
- **被掩掉**：`is_legal_action == False` 的候选动作数（掩码在采样前置 `-inf`，策略永不采到）。
- **NO_LEGAL_ACTION**：全部候选被掩 → 列车 hold、下一步重规划（**安全兜底，不是违反**）。
- 口径不含 `_retry_action` 内部的换台重试枚举（那是救援路径，另行说明）。

## Alignment check
- [ ] No hardcoded IDs: OK（改动不引入任何硬编码 ID/站名/车号）
- [ ] Observation dim: N/A（不改观测；仍为配置驱动的 31 维）
- [ ] Action mask enforces bounds: **OK — 正是本次要度量的对象；`is_legal_action` 行为逐条等价改写**
- [ ] Local critic dim: N/A（不改网络）
- [ ] Fallback re-verifies bounds: OK（不改 `RebuildFn`；额外统计其冗余拒绝数）
- [ ] Config-only expansion: OK（统计随车数/站数自动变化，无硬编码）

## Pass criteria
1. `python -m pytest tests/ -q` 全绿（掩码/解析器测试不回归）。
2. `illegal_reason` 与 `is_legal_action` 在既有测试用例上判定完全一致。
3. 5 个规模（10..30t）在 `--no-delay` clean 协议下产出 `mask.csv`；
   `postcheck_reject` 恒 0（掩码完整性自洽），`masked_pct` 显著 > 0（掩码确实在起作用）。
4. `no_legal_action` 次数与既有 `infeasible` 计数一致（同一条路径的两种记法）。
