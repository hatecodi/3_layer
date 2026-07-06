# Railway MARL — Three-Layer Scheduling Framework

三层铁路调度强化学习框架：Layer 1 硬边界约束 → Layer 2 MAPPO 调度 → Layer 3 执行（规划中）。

## 快速开始

```bash
# 五站 heuristic baseline
python scripts/five_station/run_heuristic.py

# 五站 MAPPO 训练
python -m src.experiment.run_train --env five_station --episodes 5000

# 画训练曲线（含启发式参考线）
PYTHONPATH=. python scripts/plot_training_curves.py output/five_station/20260706_201526/training_metrics.csv

# 提取 chunk 级趋势
PYTHONPATH=. python scripts/extract_trend.py output/five_station/20260706_201526/training_metrics.csv

# 跟踪 L2 输出（每列车、每站、站台、时间窗口）
PYTHONPATH=. python scripts/trace_l2_output.py
```

## 项目结构

```
├── configs/
│   ├── two_station/              # 两站 (A-B) 配置（遗留，不再更新）
│   └── five_station/             # 五站 (A-B-C-D-E) 配置 ← 当前主力
│       ├── railway_resources.yaml
│       ├── timetable.yaml
│       └── timetable_annotated.yaml
│
├── src/
│   ├── common/
│   │   └── base_env.py           # BaseRailwayEnv 抽象接口
│   ├── control/                  # 调度控制（通用于所有场景）
│   │   ├── dispatch_bounds.py    # Layer 1: 硬边界封装
│   │   ├── action_mask.py        # 动作空间 + 合法性掩码 + derive_schedule
│   │   ├── resource_manager.py   # ResourceCalendar + 冲突检测 + 确定性 resolver
│   │   ├── executor.py           # Layer 3: 规则执行器（预留，未接入训练回路）
│   │   └── execution_target.py   # Layer 2 → Layer 3 接口数据类
│   ├── env/                      # 环境实现
│   │   ├── five_station_env.py   # 五站环境（MDP，当前主力）
│   │   ├── two_station_env.py    # 两站环境（遗留，bandit-like）
│   │   ├── disturbance.py        # 初始延误注入 + 死锁阈值
│   │   ├── observation_builder.py# 29 维观测构造（config-driven dims）
│   │   └── reward.py             # 分层 reward（local + global curriculum）
│   ├── marl/                     # MAPPO 模块
│   │   ├── actor_critic.py       # FactoredActor + LocalCritic + GlobalCritic
│   │   ├── rollout_buffer.py     # RolloutBuffer + 双 critic GAE
│   │   └── mappo.py              # MAPPOTrainer（dual-critic loss fusion）
│   ├── experiment/               # 实验入口
│   │   ├── scenario_loader.py    # 配置加载 + 5 规则跨文件校验
│   │   ├── heuristic_planner.py  # 启发式调度器 + make_rebuild_fn
│   │   ├── run_train.py          # MAPPO 训练脚本（--env two_station/five_station）
│   │   ├── run_heuristic.py      # 两站启发式 baseline
│   │   └── run_executor_baseline.py # SUMO 执行器 baseline（预留）
│   └── sumo/                     # 预留：SUMO TraCI 连接器
│
├── scripts/
│   ├── two_station/              # 两站脚本（部分已存档）
│   │   └── validate_sumo.py
│   ├── five_station/             # 五站脚本
│   │   ├── run_heuristic.py      # 启发式单 episode trace
│   │   ├── visualize.py          # 网络文本可视化
│   │   ├── plot_timetable.py     # 列车运行图
│   │   ├── plot_network_layout.py# 物理拓扑图
│   │   ├── plot_heuristic_chart.py
│   │   └── generate_sumo_network.py
│   ├── plot_training_curves.py   # 训练曲线（含启发式对比虚线）
│   ├── extract_trend.py          # Chunk 级趋势提取
│   ├── trace_l2_output.py        # L2 输出追踪
│   ├── resume_training.py        # 续训脚本
│   ├── analyze_training.py       # 训练分析
│   ├── difficulty_tiers.py       # 难度分层评估
│   ├── plot_mappo_vs_heuristic.py # MAPPO vs heuristic 对比图
│   └── update_pic_figures.py     # 图片素材更新
│
├── docs/
│   ├── L2_design.md              # L2 层完整设计文档
│   └── L3_constraints_and_data.md # L3 物理约束 + 数据需求设计
│
├── archive/                      # 已存档的遗留/过时文件
│   ├── env_bak/
│   ├── scripts/two_station/
│   └── tests/
│
├── sumo/
│   ├── two_station/
│   └── five_station/             # SUMO 文件（占位，不可用于训练）
│
├── output/                       # 训练输出（按场景/时间戳组织）
│   └── five_station/
│
└── tests/
    └── two_station/
```

## 五站环境 (A-B-C-D-E)

- **10 列车**（5 上行 + 5 下行），依次经过 5 个车站，episode ≈ 30+ 步
- **D 站瓶颈**：仅 2 站台，其余各站 4 站台
- **调度决策跨站传播**：本站延误级联到后续所有站，构成完整 MDP
- **站内顺序决策**：同站列车按优先级排序处理，低优先级车看到高优先级车的选择
- **初始延误注入**：30% 概率，每列车在始发站随机延迟 30-90s

## 三层架构

```
Layer 1 (固定规则)              Layer 2 (MAPPO 学习)                 Layer 3 (规划中)
───────────────────            ────────────────────────              ────────────────
DispatchBounds                  FactoredActor                        Python Sim
timetable.yaml                   (platform_head + entry_head)        事件驱动
bounds: arr_min, dep_max        + 动作掩码 (L1 边界硬约束)            PhysicalCalendar
dwell_min, candidate_platforms  + 站内顺序决策 (优先级排序)            咽喉/走廊/站台物理约束
                                + 确定性 resolver (安全网)            speed profile
                                + VT 扰动 (60% 占正线)               dwell 追回
                                + 初始延误注入 (disturbance)
```

### Layer 1 → Layer 2 接口

YAML 中每列车每站定义：
- `planned.arr / dep`：计划到发时刻
- `bounds.arr_min / dep_max / dwell_min`：硬边界（通过 action mask 强制执行）
- `candidate_platforms`：允许的站台列表（策略只能从中选择）

中间站：`_get_effective_bounds` 将边界平移到实际到达时间，利用 120s slack 消化延误。

### 动作空间（Factored）

```
action = (platform_idx, entry_idx)     # dwell_extend 固定为 0

platform_idx: 0..3    (candidate_platforms 中的索引，≤4)
entry_idx:    0..5    (ENTRY_CHOICES: earliest, plan, plan+1~plan+4)
```

总 logits: 4 + 6 = 10（而非 flat 的 4×6×3=72）。非法动作被 mask 为 `-inf`。

### 观测空间（29 维，config-driven）

| 段 | 维度 | 内容 |
|----|------|------|
| Self | 14 | 时间、累计延迟、计划/边界时间、优先级、列车类型 one-hot、松弛量 |
| Contention | max_plat (4) | 按冲突组分摊的每站台竞争压力 |
| Next Station | 5 | 可用站台、竞争车数、预计到达时间、拥堵度、dwell |
| Resource | 2+max_plat (6) | 当前可用站台、走廊延迟、每站台 VT 预占标志 |

### Reward（curriculum）

```python
local  = -0.10 × |delay| × prio_weight × (1 + |delay|//300)    # 渐进惩罚
         + 2.0 × exp(-|delay|/60) × prio_weight                # 指数衰减 ontime bonus
held   = -3.0

global = -0.10 × mean_delay - 100.0 × deadlock + 10.0 × ontime_rate
total  = local + λ_team × global       (λ: 0.05 → 0.30)
```

ONTIME_THRESHOLD = 15s。同时惩罚早到和晚到。

### Critic（CTDE）

- **LocalCritic**：(K+1) × 29 维输入，K=1 邻居
- **GlobalCritic**：14 维全局状态
- 训练时集中，执行时每个 agent 只看自己的 29 维观测

### 冲突解决（确定性）

1. 请求按 `(priority, -delay, train_id)` 排序
2. 高优先级先 grant，booking 到 `ResourceCalendar`
3. 冲突者：换站台 → 延时 → 最多 2 次 fallback → held
4. 保证零硬边界违规

### 扰动

- **Visitor trains**：随机预占站台，60% 偏向正线 p1，观测中提前通知
- **初始延误**：`disturbance.py` 注入，30% 概率，30-90s 均匀分布

## 当前结果（5000ep，含初始延误 + VT p1_bias）

| 指标 | Heuristic | MAPPO |
|------|-----------|-------|
| dep_delay | 7.5s | **5.8s** (-23%) |
| ontime | 88.2% | **91.1%** (+2.9pt) |
| deadlock | 0 | 0 |

Best episode: 0.1s delay, 100% ontime（epoch 4297）。

训练输出 CSV 在 `output/five_station/{timestamp}/training_metrics.csv`，含 reward、ontime、delay、platform_utilization、fallback 等字段。画图脚本读取 CSV，启发式参考线内嵌于绘图脚本。

## 两站环境（遗留）

两站环境仅一步决策（所有列车在始发站同时调度），是 bandit-like 问题。当前开发重心在五站，两站相关代码保留但不再更新。部分过时的两站脚本已归档至 `archive/scripts/two_station/`。

## 依赖

- Python 3.9+
- PyTorch
- PyYAML
- NumPy
- matplotlib（画图）
- SUMO（可选，仅用于 L3 final validation，当前训练不需要）

## 设计文档

- `docs/L2_design.md` — L2 层完整设计
- `docs/L3_constraints_and_data.md` — L3 物理约束 + 数据需求
- `CLAUDE.md` — 项目约束与开发工作流
- `coding_log` — 变更历史与实验记录
