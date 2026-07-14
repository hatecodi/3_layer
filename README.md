# Railway MARL — 三层分层调度框架

三层铁路调度 MARL 框架：Layer 1 硬边界约束 → Layer 2 MAPPO 站台调度 → Layer 3 物理走廊速度控制。

## 快速开始

```bash
# 分层 MAPPO 训练（三阶段：L2 → L3 → joint）
python -m src.experiment.run_train_hierarchical --env five_station

# 带扰动训练
python -m src.experiment.run_train_hierarchical --env five_station --vt --delay

# L2 安全审查（平台资源冲突检测）
python scripts/validate_l3_schedule.py \
  --l2-ckpt output/five_station_hierarchical/{run}/checkpoints/final.pt \
  --l3-ckpt output/five_station_hierarchical/{run}/checkpoints/final.pt \
  --n-episodes 10

# 五站 heuristic baseline
python scripts/five_station/run_heuristic.py
```

## 项目结构

```
├── configs/
│   └── five_station/              # 五站 (A-B-C-D-E) 配置
│       ├── railway_resources.yaml # 站台、喉道、冲突组
│       ├── l3_physics.yaml        # L3 物理参数（速度、距离、headway）
│       └── timetable.yaml         # 11 列车时刻表 + VT 访客配置
│
├── src/
│   ├── common/
│   │   └── base_env.py           # BaseRailwayEnv 抽象接口
│   ├── control/                  # 调度控制层
│   │   ├── dispatch_bounds.py    # Layer 1: 硬边界封装
│   │   ├── action_mask.py        # 动作空间 + mask + derive_schedule
│   │   ├── resource_manager.py   # ResourceCalendar + 冲突 resolver
│   │   ├── l3_sim.py             # L3 物理走廊模拟器（per-class speeds）
│   │   ├── executor.py           # Layer 3: 规则执行器（预留）
│   │   └── execution_target.py   # L2 → L3 接口数据类
│   ├── env/                      # 环境实现
│   │   ├── five_station_env.py   # 五站环境（MDP）— 含 persistent calendar
│   │   ├── two_station_env.py    # 两站环境（遗留）
│   │   ├── disturbance.py        # 初始延误 + VT 注入
│   │   ├── observation_builder.py# config-driven 观测（~31 dims）
│   │   └── reward.py             # 分层 reward（local + global）
│   ├── marl/                     # MAPPO 模块
│   │   ├── actor_critic.py       # L2: FactoredActor + LocalCritic + GlobalCritic
│   │   ├── l3_worker.py          # L3: CorridorActor + CorridorCritic（共享）
│   │   ├── rollout_buffer.py     # RolloutBuffer + 双 critic GAE
│   │   └── mappo.py              # MAPPOTrainer
│   └── experiment/               # 实验入口
│       ├── scenario_loader.py    # 配置加载 + 跨文件校验
│       ├── heuristic_planner.py  # 启发式调度器
│       ├── run_train_hierarchical.py  # 分层训练（Phase A/B/C）
│       ├── run_train_l3_only.py  # 冻结 L2 仅训 L3
│       ├── run_flat_mappo.py     # 无 L3 的平面 MAPPO（对比 baseline）
│       ├── run_heuristic.py      # 启发式 baseline
│       └── run_executor_baseline.py # SUMO 执行器（预留）
│
├── scripts/
│   ├── five_station/
│   ├── validate_l3_schedule.py   # 安全审查：平台重叠 + 喉道冲突检测
│   ├── train_l2_quick.py         # 快速 L2 验证训练
│   └── ...
│
├── docs/                         # L2/L3 设计文档
├── output/                       # 训练输出（按场景/时间戳）
│   ├── five_station/
│   ├── five_station_hierarchical/
│   └── five_station_l3_only/
│
└── tests/
```

## 三层架构

```
Layer 1 (固定规则)          Layer 2 (MAPPO)              Layer 3 (物理执行)
───────────────────        ──────────────────           ───────────────────
DispatchBounds              FactoredActor                L3Simulator
timetable.yaml              + 动作掩码 (L1 硬约束)       per-priority speeds
bounds: arr_min/dep_max     + 站内顺序决策 (优先级)      corridor headway
dwell_min (5s 全部)         + 确定性 resolver            CRUISE/FAST/SLOW
candidate_platforms         + persistent calendar        recovery margin
t_arr_plan floor (防早发)   + VT 扰动                    硬 guard (>4× headway)
```

### Layer 1: 调度边界

- 每列车每站：`arr_min / dep_max / dwell_min / candidate_platforms`
- 中间站 `_get_effective_bounds`：`arr_min = max(actual_arrival, t_arr_plan)` — 防止早发
- 延误时自动扩展 `dep_max`（120s slack）

### Layer 2: MAPPO 调度

**动作空间**：`(platform_idx, entry_idx)` — factored action，总 logits = 4 + 6 = 10

**观测空间**：config-driven，~31 dims（取决于 max_platforms）：
| 段 | 维度 | 内容 |
|----|------|------|
| Self | 16 | 时间、累计延迟、计划/边界、priority、class one-hot、邻居平均延迟、slack |
| Contention | max_plat | 按 conflict group 分摊的平台竞争压力 |
| Next Station | 5 | 可用站台、竞争车数、est arrival、拥堵度、dwell |
| Resource | 2+max_plat | 当前可用站台、corridor delay、per-platform VT 预占标志 |

**Critic**：LocalCritic（(K+1)×obs，K=1）+ GlobalCritic（14 dims）

**冲突 resolver**（确定性）：
1. 按 `(priority, -delay, train_id)` 排序
2. 高优先级先 grant，booking 到 `station_cal`（同 step）+ `persistent_cal`（跨 step）
3. 冲突者：换站台 → 最多 2 次 fallback → held
4. 保证零硬边界违规

**Persistent calendar**：跨 step 追踪站台占用，防止不同 step 的 grant 分配到同一站台重叠窗口。Booking 通过 `w.end > t_now` 自然过期。

### Layer 3: 物理走廊 + 喉道

**L3Simulator**：entry-headway 走廊 transit，per-priority 速度查表 + 喉道 headway（6s）：
| Priority | 列车类型 | cruise (m/s) | max (m/s) |
|----------|---------|-------------|-----------|
| 1 | 复兴号 | 97.22 | 111.11 |
| 2 | 和谐号 | 83.33 | 97.22 |
| 3 | 普速 | 33.33 | 33.33 |

**L3 Worker**：共享 3-way categorical actor（CRUISE/FAST/SLOW），10-dim 观测（headway_gap、recovery_margin、cumulative_delay、distance、arrival_dev、speed_ratio、slack、platforms_free、throat_gap）

**L3 reward**：
```python
late = max(arrival_deviation, 0.0)         # 只罚晚到（早到不罚）
total_delay = headway_delay + throat_delay
effective_recovery = min(recovery_used, total_delay + late)
fuel_penalty = 0.003 * recovery_used        # FAST 微成本（准点时 CRUISE 默认）
return -0.02*headway - 0.02*throat - 0.01*late + 0.01*effective - fuel
```

**硬 guard**：headway delay > 4× corridor_headway → hold for replan

## 训练流程

三阶段交替训练（`run_train_hierarchical.py`）：

| Phase | Episodes | L2 | L3 | 说明 |
|-------|----------|----|----|------|
| A | 3000 | train | frozen/rule | L2 学习站台分配 |
| B | 2000 | frozen | train | L3 学习速度控制 |
| C | 2000 | train | train | 联合微调 |

L3-only 微调（加载已有 L2）：
```bash
python -m src.experiment.run_train_l3_only \
  --ckpt output/five_station_hierarchical/{run}/checkpoints/best.pt --episodes 2000
```

## 扰动

- **Visitor trains (VT)**：随机在 A/B/C/D/E 站预占站台（可配置 prob/p1_bias/K-ratio），观测中提前通知
- **Initial delay**：`disturbance.py` 注入，30% 概率，30-90s

## 站台配置

| 站 | 站台数 | dwell_min | 喉道 |
|----|-------|-----------|------|
| A | 4 | 5s | 每站台独立 |
| B | 4 | 5s | 每站台独立 |
| **C** | | **4** | **5s** | **p1+p2 共享 g1, p3+p4 共享 g2 (up)** |
| D | 2 | 5s | 每站台独立 |
| E | 4 | 5s | 每站台独立 |

C 站喉道共享是唯一存在咽喉资源竞争的场景，L2 通过 intra-step resolver 处理。

## 11 列车

5 上行 (G1380, G1616, G230, G5038, K1185) + 4 下行 (G1347, G1487, G2245, G7629) + 2 K 车 (K1185, K834)。G5038 为 B→C 调机换向场景（C 站 53s dwell）。

## 安全保证

1. **Hard bounds via masking**：L1 边界通过 action mask 强制，非 reward penalty
2. **Persistent platform calendar**：跨 step 防止站台重叠
3. **确定性 resolver**：保证所有 grant 窗口内无资源冲突
4. **L3 hard guard**：headway delay 超限时 hold 重调度

## 最终结果（9000ep，无扰动）

| Phase | dep_dly | ontime | cv |
|-------|---------|--------|-----|
| A (L2) | 2.0s | 91% | 3% |
| B (L3, frozen L2) | 1.3s | 100% | 39%* |
| C (joint) | **1.2s** | **100%** | 38%* |

\* cv 受少量 deadlock_flag 误触发 episode 影响（全部在途时误报），核心指标稳定。

L3 速度分布（Clean）：CRUISE=22, FAST=39, SLOW=11 — 三种模式均被使用。

Checkpoint：`output/five_station_hierarchical/20260714_231614/checkpoints/best.pt`

## 已知限制

- 模拟器在 grant 后立即 `_advance_trains`（不等 `t_dep`），与实际物理有偏差。Persistent calendar 补偿了这一缺陷，保证调度一致性。
- Corridor headway 在 L3 层执行，不在 L2 resolver 中约束。

## 依赖

- Python 3.9+
- PyTorch
- PyYAML, NumPy
- matplotlib（画图）
- SUMO（可选，仅 L3 validation）

## 设计文档

- `docs/L2_design.md` / `docs/L3_constraints_and_data.md` — 设计细节
- `CLAUDE.md` — 项目约束与开发工作流
- `coding_log` — 变更历史
- `BUG_REPORT.md` — 已知 bug 记录
