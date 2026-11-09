# Plan: clean 场景扩展性评估 (10t→30t) → `output/clean/`

## 背景 / 目标

现有扩展性曲线 (`output/9_4/scalability/`) 全部是 **incident** 臂；**clean** 臂只有 15t 一份
(`output/8_14/clean/`，且只有 4 张表、无优先级准点/算力指标)。

用户要：clean 场景下 **10/15/20/25/30t** 各规模的**各项指标**评估数据，存到 `output/clean/`。

## 决定(用户确认)

1. **指标口径对齐 incident 扩展性**：每规模 5 张 CSV — `ontime.csv`(含 p1/p2/p3)、`delay.csv`、
   `tail.csv`、`compute.csv`、`episode.csv`。（`affected.csv` 不写：clean 无限制窗口，该列恒为 0，无意义。
   `inc_eps/aff_eps` 同样不适用，clean 的 `episode.csv` 只保留 succ/deadlock/trunc。）
2. **6 种 mode**：`learned / sa / sa-online / sa-nofc / cruise / flat`。
3. **协议同扩展性**：15t 免重训迁移 ckpt (L2 `output/8_14/training/phaseA/best.pt`,
   L3 `output/8_14/training/phaseB_v3/best.pt`, flat `output/8_14/training/flat_mappo/best.pt`)；
   `--no-delay`(关闭 origin bunching 与 corridor incident，完全干净时刻表)；
   `--seeds 900000 --n 60 --blocks 3`(与 incident 曲线同种子，逐点可比)。

## 改动(代码)

`scripts/eval_protocol.py` — 非 incident 分支补上优先级准点 + E1 算力(eval only，不动训练/环境):

- 新增 `report_clean(eps, mode)`：复刻 `report_incident` 的
  - 优先级到站准点 P1/P2/P3(all 口径，无 affected 细分)；
  - realized arrival delay 全分布 (n/mean/p50/p90/p95/max) + 尾端 severe；
  - E1 re-solve compute (`n_transits` / `n_sections` / `method_ms` / `sa_resolve_ms` / `learned_ms` / `resolve_ratio`)；
  返回值键集合与 `report_incident` 对齐(去掉 aff_* 与 inc_eps/aff_eps)。
- main() 非 incident 分支：仍打印 bunched/non-bunched 分层(clean 下恒 0)，但 `stats_by_mode` 改取 `report_clean`。
- `--out-dir` 非 incident 分支改为写 5 张表：
  `ontime.csv[ontime,p1,p2,p3] / delay.csv / tail.csv / compute.csv / episode.csv[succ,deadlock,trunc]`，
  与 incident 臂字段名兼容 → 下游可直接合并 clean vs incident。

## 执行步骤

1. 改 `scripts/eval_protocol.py`（上述）。
2. Smoke：`--scenario five_station_30t --cruise-only --no-delay --n 10` 确认 0 deadlock/0 trunc。
3. 5 个规模 × 6 mode 全量跑（顺序，各规模 ~1-2 min，参照 9-04 incident 重跑耗时）：
   ```
   python scripts/eval_protocol.py --scenario five_station_<N>t \
     --l2   output/8_14/training/phaseA/best.pt \
     --l3   output/8_14/training/phaseB_v3/best.pt \
     --flat-ckpt output/8_14/training/flat_mappo/best.pt \
     --no-delay --n 60 --blocks 3 --seeds 900000 \
     --out-dir output/clean/<N>t | tee output/clean/<N>t.log
   ```
   N ∈ {10,15,20,25,30}。（注意：clean 与 incident 的 mode 集合不同时才需 flat ckpt；6 mode 全要 flat。）
4. 合并 `output/clean/summary_ontime.csv`（列：scale, learned, sa, sa-online, sa-nofc, cruise, flat），
   汇总脚本用一段一次性 python（不新增常驻脚本）。
5. （可选）出图 `output/clean/scalability_clean.png`：clean vs incident 两臂 ontime 曲线并排。
6. `coding_log` 追加：改动 + 全部产出路径 + 关键数字；`HANDOFF.md` 若涉及则同步。

## 验证

- 每规模 5 张 CSV 存在且 mode 行齐全(6 行)；`episode.csv` 中 deadlock=0、trunc=0 且 succ=180。
- `summary_ontime.csv` 5 个规模点齐全，learned > cruise(clean 无扰动的协调价值证据)。
- `python -m pytest tests/ -q` 不回归(eval 脚本改动不触及训练/env)。

## 风险 / 备注

- clean 下 `enable_initial_delay=False`，bunched 恒为 0 → `report_legacy` 的分层行退化为同一份数据，
  这是预期；新 `report_clean` 不再需要 bunched 分层。
- clean 与 incident 的差异应只在"无扰动"，故 ckpt 的 `--no-delay` 使用与文档一致(in-distribution learned 对比)。
- 30t 无 P3 车(9-04 已规范化)，ckpt 迁移无外推问题。
