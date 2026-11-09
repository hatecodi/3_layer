# Plan: 规模 × 方法的"尾部延误"柱状图（带误差棒）
Date: 2026-09-21
Status: in_progress

## 目标
一张图：x = 规模(10/15/20/25/30t)，分组 = 方法(learned/sa/sa-online/sa-nofc/fcfs/cruise/flat)，
y = **尾部延误 = 最差 10% transit 的延误均值**，带误差棒。

## 现状：画不出来（已实测）
- `output/9_4/scalability/15t/delay.csv` = `mode,n,mean,p50,p90,p95,max` —— 每 mode **一个数**，
  `n=6480` 是 180 集 × 36 transit **pooled**，block 结构没留。
- 唯一有 per-block 粒度的 `output/9.20/eval_per_block.csv` 列 = `n,mean,aff_n,aff_mean,
  unaff_n,unaff_mean,ontime` —— **无 p90**。
- `tail.csv`(`sev_rate/sev_mean/sev_max`) 同样只在聚合层。
→ **尾部量在 per-block / per-episode 层根本不存在**，误差棒无源。

## 误差棒的正确单位（这次的关键设计）
`p90` 是**分位数**，3 个 block 的 std 不是它的抽样误差（df=2，噪声极大）。真正 iid 的
单位是**回合**，所以：
- 每个 (scale, mode) 逐回合落盘统计量 → `per_episode.csv`；
- 图上的误差棒 = **按回合 bootstrap**（1000 次重采样，2.5/97.5 分位）。180 集/规模足够。
- learned 臂**另叠**跨 run 离散度（15t/20t 4–5 个 ckpt、25t 2 个、30t **1 个** → 30t 只能靠
  bootstrap，图上标 `n_runs`）。
- 每回合的尾部量定义为**该回合内最差 10% transit 的延误均值**，再对回合取均值 ——
  估计量明确（"每回合各自尾部的平均"），且能干净地从回合级 bootstrap。
  （不同于"pooled 全样本 top-decile"，会在脚本 docstring 里写明。）

## 改动

### 1. env：`affected_plan`（策略无关的受影响标记）
`src/env/five_station_env.py` —— transit 事件 append 处（~1266-1274）加一个 key：
```python
inc = self._corridor_incidents.get(corridor_key)
affected_plan = False
if inc:
    p_entry = train_plan[cur].dep
    p_exit = p_entry + float(cfg.time_generation["corridor_runtime_sec"].get(corridor_key, 90.0))
    affected_plan = (p_entry < inc[1]) and (p_exit > inc[0])
```
**为什么**：现有 `affected` 由 `sf_k != 1.0` 决定（`five_station_env.py:1062-1063`），即按**实际
轨迹**判定 → "受影响车"这个**子集随策略变化**，跨策略比它是在比不同的车。`affected_plan` 只
用计划时刻表 → 策略无关。两者**都记**，可对照。
纯埋点：只往事件 dict 加 key，不动物理、不改 reward、不改 mask。

### 2. `scripts/eval_protocol.py`
- `_block_stats` 加尾部列（全车 + affected）：`tail10`, `tail10abs`, `p90`, `max`,
  `aff_tail10`, `aff_max`。`per_block.csv` 加列是安全的 —— 消费者
  `collect_crossrun_eval.py` 用 `rows[0].keys()` 生成 fieldnames；`plot_conv_scalability.py`
  读的是 `ontime.csv`（不动）。
  - `tail10abs`（按 |delay| 取最差 10%）= 防"整体跑早"抵消的护栏；新增列而非改定义。
- **新增 `per_episode.csv`**（bootstrap 的数据源）：每 (mode, block, seed) 一行，
  列 = `mode,block,seed,n,mean,tail10,p90,max,aff_n,aff_tail10,aff_max,ontime`。

### 3. `scripts/plot_tail_by_scale.py`（新）
- 读各 run 的 `per_episode.csv` → 按 (scale, mode) 汇总；
- 按**回合 bootstrap** 出 CI → 写 `output/9.21/tail_by_scale.csv`（CSV 先行）；
- 画分组柱状图 + 误差棒 → `output/9.21/fig_tail_by_scale.png`；
- learned 臂叠跨-run 散点/第二层棒并标注 `n_runs`；10t/20t 的 flat 走 15t 迁移口径。

### 4. 跑法（纯 eval，不训练）
沿用 `collect_crossrun_eval.py` 的 RUNS 清单（16 个 learned ckpt：10t×4(15t迁移)、15t×4、
20t×5、25t×2、30t×1）+ 每规模 1 次 checkpoint-无关模式(6 modes)。
`--incident --blocks 3 --n 60 --seeds 900000`（与既有论文口径一致）。
规模：learned 16 run ≈ 1–1.5h；模式 eval 30 次 ≈ 1–2h。

## Alignment check
- [x] 无硬编码 ID：OK（只读计划时刻表）
- [x] 观测维度：N/A（未动 observation）
- [x] 掩码强制边界：N/A
- [x] 局部 critic：N/A
- [x] **不改 env 物理/奖励/掩码**：只加事件 dict 的一个 key
- [x] 仅配置扩展：OK

## Pass criteria
1. `per_episode.csv` / `per_block.csv` 出现新列；旧列语义与数值不变。
2. `affected_plan` 与 `affected` 对照：`affected_plan=1` 的条数 ≥ `affected=1`（计划口径不会因
   策略挪开实际时刻而"漏标"），且两者不一致的比例有报告（= 策略把车挪出窗口的比例）。
3. 5 个规模 × 6–7 方法都出柱；每个柱有 CI；learned 臂标出 `n_runs`；30t 显式标注 n_runs=1。
4. `tail10abs` 与 `tail10` 并列报出 —— 若二者结论冲突（早到抵消），图上必须同时呈现。
