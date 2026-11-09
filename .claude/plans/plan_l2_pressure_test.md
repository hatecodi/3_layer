# Plan: 小规模 L2 压力测试（learned L2 vs 启发式 L2，按难度分档）

Date: 2026-09-18
Status: done（脚本 `scripts/l2_pressure_test.py` + 三组结果 CSV；见 coding_log 2026-09-18）

## What
新增 `scripts/l2_pressure_test.py`（只读评估脚本，不改环境/训练代码）：

- 对同一批固定种子，各跑两次：**learned L2**（策略采样）与 **启发式 L2**（`env.step({})`，
  环境内建 `heuristic_action`）。
- **两臂共用同一个 L3 执行器**（同一个 learned L3 ckpt）→ 差异只归因于 L2 决策层。
- **难度评分**：用启发式臂在该种子上的平均发车延误作为客观难度分。
- 按难度分**三分位分档**（Easy/Medium/Hard），分档报告两臂的 {平均发车延误, 到站准点率}。
- 输出：逐种子 CSV + 分档汇总打印。

难度来源：默认开启源发成束晚点（`enable_initial_delay=True`），不同种子随机性不同 →
天然形成难度梯度。可选 `--extra-train` 加压。

## Why
用户要一个 L2 压力梯度的小规模数据（"有数据就行"），用于支撑 L2 定位
（站台选择 + 路权分配决策层；规则固定、策略空间小，但压力下策略价值显现）。

## 与历史产物的区别（重要）
已有 `output/figures/VT_rank/vt_difficulty_data.csv` 与 `output/pic_of_3/difficulty_tiers.json`，
但**架构陈旧（2026-06-29）且疑似 in-sample**，不可引用。
本脚本在当前架构（obs 31 / learned L3 / 现奖励）上、用 held-out 种子区间重做，
且**同一批种子配对比较**（两法跑同一批实例），避免 in-sample 问题。

## Alignment check
- [ ] No hardcoded IDs: OK（站名/车号均来自 config）
- [ ] Observation dim: OK（用 `get_obs_dim(cfg)` 实测 31，不写死）
- [ ] Action mask enforces bounds: OK（不修改掩码；两臂共用同一掩码逻辑）
- [ ] Local critic dim: OK（`k_local=1`，与 eval_protocol 一致）
- [ ] Fallback re-verifies bounds: OK（不改回退逻辑）
- [ ] Config-only expansion: OK（`--scenario` 切换规模，无硬编码）

## Pass criteria
1. 产出逐种子 CSV（含 tier 列）与分档汇总；
2. 两臂在同一批种子上配对，`Hard` 档差值与 `Easy` 档差值可直接比较；
3. 规模：默认 40 种子 × 2 臂，15t 下单臂约 1–2 s → 总时长数分钟；
4. 报告策略决策占比（复用 `info["mask_stats"]` 的 decisions 与实测口径）作为机制说明。
