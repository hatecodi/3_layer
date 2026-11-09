# Plan: 训练期"受影响车辆延误"逐集埋点 + 15t 单种子重跑
Date: 2026-09-21
Status: in_progress (埋点 DONE + 冒烟验证通过; 两个 6000 集 run 在跑)

## What

**问题**：`training_metrics.csv` 只有**全车聚合**（`mean_dep_delay` 是发车侧、`ontime_rate`/`arr_ontime`
是准点率），**没有 affected/unaffected 拆分，也没有逐车到达延误**。所以"受影响车辆延误随 episode
的变化"在已有 8.14–8.16 数据里**取不到**（reward 侧取得到，见下）。需要加埋点并重跑。

### 1. 埋点（两个训练脚本，口径必须一致）

`src/experiment/run_train_hierarchical.py`（累加点 ~245-250，ep_summary ~328）与
`src/experiment/run_flat_mappo.py`（累加点 ~321-325，ep_summary ~349）各加 4 列：

| 列 | 来源 | 说明 |
|---|---|---|
| `has_incident` | `bool(env._corridor_incidents)`（reset 后立即取） | 该回合是否**真的注入**了限速（~6/10） |
| `n_aff_transits` | `info["l3_transit_events"]` 中 `affected=True` 的条数 | 受影响样本量 |
| `aff_arr_delay_mean` | 同上事件的 `arr_delay` 均值；无受影响 transit 时 `nan` | 主量 |
| `aff_arr_delay_max` | 同上事件的 `arr_delay` 最大值；无则 `nan` | 尾部 |

- 事件源：`five_station_env.py:477` 的 `info["l3_transit_events"]`，**逐 step**（`_l3_transit_events`
  在 `_advance_trains` 开头清空），事件含 `arr_delay` + `affected`；append 是**无条件**的
  （`five_station_env.py:1266-1274`），CRUISE / SA / learned 三种 L3 都记 → **flat 也能用**。
- `affected` 的定义沿用现有口径（transit 窗口与限速窗口重叠），不新增定义。
- 与已有列的关系：新增列**只加不改**，旧列语义不变（`plot_conv_scalability.py` 读的是
  `arr_ontime`，不受影响）。两脚本 `fieldnames` 均从 `episode_metrics[0].keys()` 动态生成
  （hier:409、flat:384），加键即自动进表。

### 2. 变化率（记录原始序列，下游算）

**不**在训练里算斜率：incident 仅 ~6/10 触发、每回合受影响 transit 只有几条，
逐集曲线是"大量 nan/0 + 偶发尖峰"，直接求差分没有意义。所以**记原始逐集值**，
另加 `scripts/analyze_train_affected.py` 产出：
  - 按 `has_incident` 分层的逐集曲线
  - 滚动均值（窗口 100 集）+ 分 phase 的线性斜率（= 用户要的"变化率"）
  - reward 同口径（`total_reward` / `mean_reward` 变化率）—— 这部分**已有数据也能算**，
    新脚本对 8.14/8.16 老 run 同样可用
  - 输出 `output/9.21/train_affected_<run>.csv` + 一张图（CSV 先行，符合既有习惯）

### 3. 重跑（单种子 6000 集验证通路）

对齐 8_16/9_4 的既有协议（`plan_30t_retrain_multiseed.md`）：
`--incident`（限速 0.6）、关闭 origin bunching / VT / 初始延误。

```
python -m src.experiment.run_train_hierarchical --env five_station_15t --incident \
    --phases A,B --seed 0 --out-root output/9_21/training \
    --eval-every 200 --eval-n 12 --eval-seed-base 900000
python -m src.experiment.run_flat_mappo --scenario five_station_15t --incident \
    --episodes 6000 --seed 0 --out-root output/9_21/training \
    --eval-every 200 --eval-n 12 --eval-seed-base 900000
```

env 种子 = 集索引（hier:217 / flat:265 均为 `seed=ep`），全局计数，故两脚本的 phase B
跑在同一批种子上 —— 与既有 crossrun 比较口径一致。

## Why

论文的时间维度论据目前只有奖励曲线（`plot_conv_scalability`），而"分层是否更快把受扰车辆
拉回正常"只能由**受影响子集**的延误收敛来说明。全车均值会把受影响 transit 稀释掉
（既有实测：affected 只占 3–8%，见 `collect_train_dispersion.py` 的 affected_vs_overall），
所以必须分开记。

## Alignment check
- [x] 无硬编码 ID：OK（只读 `affected`/`arr_delay`）
- [x] 观测维度：N/A（未动 observation）
- [x] 掩码强制边界：N/A（未动 action_mask）
- [x] 局部 critic：N/A
- [x] fallback 复检：N/A
- [x] 仅配置扩展：OK
- [x] **不改 env 行为**：只读取已有 `info["l3_transit_events"]`，训练动态零影响

## Pass criteria

1. 20 集 smoke（两脚本各一次）跑通，CSV 出现 4 个新列，且
   `n_aff_transits > 0` 的回合其 `aff_arr_delay_mean/max` 与 held-out eval 的
   `affected` 量级一致（15t 实测 affected 均值 ~40–90 s）。
2. `has_incident=False` 的回合 `n_aff_transits == 0` 且均值列为 nan（不是 0 —— 0 会污染下游均值）。
3. 旧列与旧 run 的兼容性：`analyze_train_affected.py` 对 `20260814_194055`（无 arr_ontime、
   无新列）不崩，只输出 reward 变化率并提示"affected 列缺失"。
4. 正式单种子 6000 集：hier 与 flat 各 1 run 完整跑完（status 全 success），
   `training_metrics.csv` 6000 行 + 4 新列，`output/9.21/train_affected_*.csv` 产出，
   保存位置写进 `coding_log`（CLAUDE.md 要求）。
