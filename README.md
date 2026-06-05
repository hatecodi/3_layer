# Railway MARL — Three-Layer Scheduling Framework

三层铁路调度强化学习框架：Layer 1 硬边界约束 → Layer 2 MAPPO 调度 → Layer 3 执行。

## 快速开始

```bash
# 两站 heuristic baseline
python scripts/two_station/run_heuristic.py

# 五站 heuristic baseline
python scripts/five_station/run_heuristic.py

# 五站网络可视化
python scripts/five_station/visualize.py

# MAPPO 训练
python -m src.experiment.run_train --env two_station --episodes 2000
python -m src.experiment.run_train --env five_station --episodes 2000
```

## 项目结构

```
├── configs/
│   ├── two_station/              # 两站 (A-B) 配置
│   │   ├── railway_resources.yaml    # 物理资源：站台、咽喉、走廊
│   │   └── timetable.yaml            # 时刻表 + Layer 1 bounds
│   └── five_station/             # 五站 (A-B-C-D-E) 配置
│       ├── railway_resources.yaml
│       └── timetable.yaml
│
├── src/
│   ├── common/                   # 共享基类
│   │   └── base_env.py           # BaseRailwayEnv 接口
│   ├── control/                  # 调度控制（通用于所有场景）
│   │   ├── dispatch_bounds.py    # Layer 1 边界计算
│   │   ├── resource_manager.py   # 资源日历 + 冲突检测 + 确定性 resolver
│   │   ├── action_mask.py        # 动作空间 (platform, entry, dwell) + 合法性掩码
│   │   ├── executor.py           # Layer 3 执行器
│   │   └── execution_target.py   # 执行目标数据结构
│   ├── env/                      # 环境实现
│   │   ├── two_station_env.py    # 两站环境
│   │   ├── five_station_env.py   # 五站环境（MDP，跨站状态转移）
│   │   ├── disturbance.py        # 扰动注入 + Layer 2 激活逻辑
│   │   ├── observation_builder.py# 74 维观测构造
│   │   └── reward.py             # 分层 reward (local + global)
│   ├── marl/                     # MAPPO 模块
│   │   ├── actor_critic.py       # MaskedActor + LocalCritic + GlobalCritic
│   │   ├── rollout_buffer.py     # RolloutBuffer + 独立 GAE
│   │   └── mappo.py              # MAPPOTrainer (dual-critic loss fusion)
│   ├── experiment/               # 实验入口
│   │   ├── scenario_loader.py    # 配置加载 + 校验
│   │   ├── heuristic_planner.py  # 启发式调度器
│   │   └── run_train.py          # 训练脚本（env-agnostic）
│   └── sumo/                     # 预留：SUMO TraCI 连接器
│
├── scripts/
│   ├── two_station/              # 两站脚本
│   │   ├── run_heuristic.py      # heuristic baseline
│   │   ├── validate_sumo.py      # SUMO 验证
│   │   ├── verify_phase3_env.py  # Phase 3 验收测试
│   │   └── verify_phase4_mappo.py# Phase 4 验收测试
│   └── five_station/             # 五站脚本
│       ├── run_heuristic.py      # heuristic baseline
│       ├── visualize.py          # 网络文本可视化
│       ├── plot_timetable.py     # 列车运行图
│       ├── plot_network_layout.py# 物理拓扑图
│       └── generate_sumo_network.py # SUMO 网络生成
│
├── sumo/
│   ├── two_station/              # 两站 SUMO 文件
│   └── five_station/             # 五站 SUMO 文件
│
├── output/                       # 训练输出（按场景/时间戳组织）
│   ├── two_station/
│   └── five_station/
│
└── tests/
    ├── two_station/
    └── five_station/
```

## 两个环境

### 两站 (A-B)

- 8+1 列车在始发站调度一次，episode = 1 步
- 适合验证调度逻辑和 baseline，不适合 MAPPO（bandit problem）

### 五站 (A-B-C-D-E)

- 9 列车依次经过 5 个车站，episode ≈ 30+ 步
- D 站仅 2 站台，形成自然瓶颈
- 调度决策跨站传播，构成完整 MDP → MAPPO 可学到有意义的策略

## 调度架构

```
Layer 1 (固定规则)        Layer 2 (MAPPO 学习)         Layer 3 (规则执行)
─────────────────        ────────────────────         ────────────────
DispatchBounds           策略选择动作                   Executor
timetable.yaml           (platform, entry, dwell)      TraCI 控制
arr_min / dep_max        动作掩码过滤非法动作           执行 granted window
dwell_min                冲突检测 → resolver           记录实际到发
```

列车默认为 NOMINAL（按时刻表运行），扰动触发 LAYER2_ACTIVE（策略接管）。

## 动作空间

```
a_i = (platform_choice, entry_choice, dwell_extend)
```

| 子动作 | 可选值 |
|--------|--------|
| platform_choice | 候选站台 (P_i) |
| entry_choice | earliest, plan, plan+1, plan+2, plan+4 (分钟) |
| dwell_extend | 0, 60, 120 (秒) |

最大 45 个组合，非法动作被 mask 为 -inf。

## 训练输出

每次训练在 `output/{scenario}/{timestamp}/` 下生成：

- `training_metrics.csv` — episode 级指标（主要数据源，用于画图）
- `training_metrics.json` — 同上，JSON 格式
- `checkpoints/best.pt` — 最优模型
- `checkpoints/final.pt` — 最终模型

CSV 字段：episode, steps, status, total_reward, mean_reward, mean_dep_delay, mean_arr_delay, n_granted, n_held, lam_team。

画图时从 CSV 读取，不直接使用内存数据。

## 配置说明

### railway_resources.yaml

定义物理网络：车站、站台、咽喉路由（含 conflict_group）、走廊。

### timetable.yaml

定义每列车：
- `planned`: 计划到发时刻
- `bounds`: Layer 1 硬边界 (arr_min, dep_max, dwell_min)
- `candidate_platforms`: 每站可用站台
- `route`: 始发站、终到站、方向、走廊

## 依赖

- Python 3.12+
- PyTorch
- PyYAML
- matplotlib (画图)
- SUMO (可选，可视化用)

## 设计文档

详见 `three_layer_railway_marl_fixed_1.md` 和 `Implementation Specification v4.md`。
