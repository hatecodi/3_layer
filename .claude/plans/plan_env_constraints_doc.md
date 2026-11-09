# Plan: 环境约束清单文档（论文用）

Date: 2026-09-18
Status: done

## What
新增 `docs/environment_constraints.md`（只新增文档，不改任何 .py/.yaml）。内容：当前环境**实际生效**的
硬约束/软约束清单，逐条带 `文件:行` 出处，供论文"实验环境/约束建模"章节直接引用。

覆盖：
1. 场景与拓扑（A–E 五站、站台数 A/B/C/E=4、D=2、走廊里程与时间压缩 /25）
2. L1 时刻表包络（`t_arr_min`/`t_dep_max`/`dwell_min` + 一致性校验）→ 全部走掩码
3. 动作空间与掩码规则（站台候选、通过股道、pass-through 零停站、6 档进入时刻；NO_LEGAL_ACTION→hold）
4. 站台/咽喉资源（`conflict_group` 互斥、`min_gap=5s`、解析器确定性优先级规则、fallback 重验边界、2 次尝试上限）
5. 走廊物理（移动闭塞 500 m + 3 s、咽喉 6 s、走廊下限 5 s、末段 `limit_mps=70`、CRUISE/FAST/SLOW 速度定义、
   校准 `calibrate_corridor_sections`、无越行 FIFO）
6. 终止条件（success / deadlock 15 步 / truncated 1500 s，步长 10 s）
7. 观测/动作维度（L2 obs 31、L3 obs 10、L3 动作 3、local critic `(1+1)×31=62`、global 14）
8. 扰动（成束晚点、visitor、走廊限速 incident）参数
9. 软约束（用奖励而非掩码的项）——显式列出，含 swap_cost、headway/throat 惩罚、FAST 能量惩罚
10. 代码 vs 注释/文档不一致清单（写作时勿引用）
11. 论文可直接用的中文段落

## Why
用户要开始写论文，需要把当前环境的约束整理成可引用文档。现有 `docs/` 里
`L2_design.md`（obs 29 维）、`L3_constraints_and_data.md`（设计期"需要新增参数"）、
`.claude/skills/coding-workflow.md`（74 维 / 444 维）均为陈旧数字；`paper_env_mdp_reward.md`
只讲 MDP 与奖励、不系统列约束。缺一份"以代码为准"的约束清单。

## 已核验的关键数字（本次实测，非引用注释）
- obs_dim = 16(self) + 5(next) + 2(res) + max_plat×2 = **31**（`observation_builder.py:22-47`）
- 训练用 `k_local=1` → local critic **(1+1)×31 = 62**（`run_train_hierarchical.py:49,143`）
- `ONTIME_THRESHOLD_SEC = 15.0`（前后均计）（`reward.py:13`）
- `DEADLOCK_THRESHOLD = 15`（`disturbance.py:17`）；`MAX_EPISODE_TIME = 1500`、步长 10 s（`five_station_env.py:108,127`）
- 移动闭塞 `500 m` / `3 s`（`l3_physics.yaml:76-77`）；咽喉 `6 s`（`l3_sim.py:22`）；
  走廊 headway 下限 `5 s`（`l3_physics.yaml:47`，且 `compute_corridor_headway` 取时刻表实际最小间隔的 max）
- 站台 `min_gap = 5 s`（`five_station_env.py:224` 等多处实例化 `ResourceCalendar`）
- **发现的硬编码**：`five_station_env.py:198` 物理配置写死 `configs/five_station`，各规模（20/25/30t）
  的 `l3_physics.yaml` 实际**不被读取** —— 论文需说明物理层全规模共用一个配置。

## Alignment check
- [ ] No hardcoded IDs: N/A（仅文档）
- [ ] Observation 74-dim: **N/A — 实测为 31 维**（旧 skill 的 74 维已过时，本次在文档中记录）
- [ ] Action mask enforces bounds: OK（文档如实记录掩码为唯一强制路径）
- [ ] Local critic 444-dim: N/A — 实测 62 维（k_local=1）
- [ ] Fallback re-verifies bounds: OK（文档记录 `RebuildFn` 重验）
- [ ] Config-only expansion: OK（文档记录配置驱动；并标注物理配置的单点硬编码例外）

## Pass criteria
- `docs/environment_constraints.md` 每条约束都有 `文件:行` 出处，且数字与当前代码一致。
- 明确区分"掩码/物理强制"与"奖励塑形"两类，并单列陈旧数字清单。
- 不修改任何代码文件；`git status` 只多出该文档与 plan。
