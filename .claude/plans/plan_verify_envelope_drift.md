# Plan — 核实 §3.6.3 的两个数字（包络漂移率 / 放宽救回率）

## 背景

`docs/paper_outline_ch3_ch4.md` §3.6.3 引用了两个"实测"数字，并被写进了
`docs/ch3_problem_formulation_paper.md` §3.6.3（已标 ⟨待核⟩）：

1. **70–76% 的决策点**动态包络与静态时刻表包络不同
2. 真正"放宽并救回一个原本非法动作"的比例 **0（clean/常态 incident）～2.3%（极端压力）**，
   且随强度单调上升（0 → 0.8% → 2.3%）
3. 附带：漂移"多数方向是收紧"，平均 `dep_max` 位移 **−27～−42 s**

**仓库中不存在产出这些数字的脚本或中间数据**（已全库搜索）。即当前**不可复现**。

## 定义（按代码实现固定口径）

- **静态包络** $\mathcal{B}^{\text{static}}$：`cfg.trains[i].bounds[station]`（时刻表展开，预先计算）
- **动态包络** $\mathcal{B}^{\text{dyn}}$：`env._get_effective_bounds(i, station, t_now)`
  —— 中途站 `arr_min = ref_time = max(actual_arr, t_now, planned_arr)`，
  `dep_max = max(ref + dwell + slack, ref + overhead + dwell)`；
  始发站仅在 `actual_arr > planned_dep` 时平移（`five_station_env.py:1292-1333`）
- 二者接口一致（`.arr_min/.dep_max/.dwell_min`），可逐字段比

**指标**（每个决策点 × 每辆待决策车，容差 `TOL = 1 s`）：

| 指标 | 定义 |
|---|---|
| `drift` | `|Δarr_min| > TOL` 或 `|Δdep_max| > TOL` |
| `relax` | `Δdep_max > TOL` 或 `Δarr_min < -TOL`（放宽） |
| `tighten` | `Δdep_max < -TOL` 或 `Δarr_min > TOL`（收紧） |
| `gained` | 动态包络下的合法动作数 **>** 静态包络下的合法动作数 |
| **`rescued`** | 静态下**无**合法动作（全掩码），动态下**有** ≥1 个 |

`gained` / `rescued` 由 `compute_joint_action_mask(..., bounds_override=B)` 两次求掩码后比较。

## 实现

`scripts/measure_envelope_drift.py`（新增）：
- 复用 `eval_protocol` 的 learned 驱动（L2 策略 + L3 worker），逐决策点在 `env.step` **之前**取
  静态与动态包络
- 参数：`--scenario`、`--episodes`、`--arm {clean,incident}`、`--severity`、`--l2/--l3`
- 输出 `output/9.20/envelope_drift.csv`（每 arm 一行）+ 终端表

## 判定标准

| 结果 | 结论 |
|---|---|
| 三档（clean / incident / incident+severity2）的 drift 落在 70–76%，rescued 落在 0–2.3% 且单调 | 数字**成立**，删掉 ⟨待核⟩ 标记 |
| 数量级相符但区间不同 | 数字**需按新测量改写**，同步 3 处文档 |
| 明显不符 | **撤回该论点**或改为定性表述 |

## 不做

- 不改 `five_station_env.py`（纯测量）
- 不重复 `eval_protocol` 已有的指标输出
