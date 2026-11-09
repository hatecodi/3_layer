# Plan — 车队级联合 SA 的 time-to-target 规模曲线 (Fig: 指数搜索 vs 恒定决策)

## 目标

产出一张图回答"**为什么大规模协调必须靠学习而不能靠搜索**"：

- **B 曲线**（主图）：车队级联合 SA 的 **time-to-target**（跑到固定质量阈值所需墙钟）vs 列车数 N，
  log 纵轴。预期随 N 快速增长。
- **A 曲线**（内嵌小图）：联合解空间基数 `3^(4N)` vs N，log 纵轴，理论参照。
- **对照平线**：学习式 L3 的决策耗时（每区间一次前向），与 N 无关。

用途：支撑 §6.5（逐区间反应式 vs 逐趟全解）与 §6.6（执行层基线比较），把 §6.6 从
"逐趟 oracle 的规模反转"扩展成"全局联合搜索不可解 → 协调必须学习"。

## 硬约束（必须写进正文，否则图不可信）

1. **现有 `src/control/l3_sa.py` 测不出指数曲线** —— 它是逐趟、`3^4=81`，与 N 无关。
   本图需要**新的车队级联合 SA**，不是复用 `l3_sa.py`。
2. **固定迭代预算的联合 SA 耗时是 O(N)，不是指数**。因此测的是 time-to-target
   （迭代数本身会爆），不是"固定预算跑完的时间"。
3. 该联合 SA 是**集中式全信息 oracle**，比逐趟 SA 强得多 —— 图上必须标注为
   "joint SA (centralized, full-information)"，不可简写成 "SA"。
4. 实测**可能不是干净指数**。若为多项式则如实画多项式，结论（学习恒定 vs 搜索增长）
   仍成立，只改措辞。

## 联合问题的定义

- **决策向量**：全部 N 列车的逐区间速度模式拼接 → `4N` 个变量，空间 `3^(4N)`。
  （沿用"同一 L2 时刻表、只换 L3 执行层"口径，与 §6.6 一致。）
- **目标**：Σ 各车 terminal reward − FAST penalty，与 `l3_sa.py` 同款目标，保证与
  learned / 逐趟 SA 可比。
- **耦合来源**：共享走廊上的 moving-block headway（经同一个 `L3Simulator` 状态）。
  ⚠️ 只有共享走廊的列车之间才真耦合 —— **耦合强度决定曲线是否指数**，这是本计划
  最大的不确定性，先做可行性探针再决定图表形态。

## TTS 定义

target = **学习式 L3 在同实例上达到的目标值**。
即问："联合搜索要花多久，才能勉强追上反应式策略一次前向就达到的水平？"
→ 直接呼应 §6.5/§6.6，且 target 随实例自适应，不用手调阈值。

记录：达到 target 的**迭代数**、**墙钟**、以及是否达到（未达到记为 censored）。

## 实现

| 文件 | 内容 |
|---|---|
| `src/control/joint_sa.py` | `joint_sa_optimize(...)`：对拼接向量退火；候选评估按发车时刻序重放全部 transit 经共享 `L3Simulator`；返回 (best_modes, best_value, iter_to_target, wall_to_target) |
| `scripts/bench_joint_sa_scaling.py` | 逐 scale（10/15/20/25/30t）：构造实例 → 跑 learned 得 target → 跑联合 SA 测 TTS → 写 CSV |

输出（遵守 CSV-first）：
- `output/figures/joint_sa_scaling.csv` — 列：scale, n_trains, n_vars, n_transits, target, iter_to_target, wall_s_to_target, reached, best_value
- 图另出，从 CSV 画。

## 步骤

1. 实现 `joint_sa.py` 的最小版本。
2. **可行性探针**：只在 10t / 20t 上各跑一次，看曲线形状与耦合是否显著。
   - 若 TTS 明显增长 → 继续 10..30 全扫。
   - 若近似平/线性（耦合太弱）→ 停下来汇报，重新设计（例如只在共享走廊上构造实例，
     或改用"固定预算质量衰减"图 C）。
3. 全扫 + 出图 + 记录 coding_log。

**不做**：RL 训练。本图只需前向仿真，比收敛重训便宜得多。

## 验证

- 联合 SA 的 best_value 不应劣于逐趟 SA 的 Σ（联合空间是逐趟空间的超集）—— 若不成立说明评估器有 bug。
- `n_vars == 4 * n_trains`；`3**n_vars` 与 A 曲线一致。
- TTS 单调性检查：同一 scale 多次运行，iter_to_target 的离散度需记录（SA 随机性）。
