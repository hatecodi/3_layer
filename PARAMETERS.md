# 参数中英文对照表

## 正点/晚点定义 (Punctuality)

定义 reward、统计指标所用的正晚点判定规则。

### 出发

| 判定 | 条件 | 含义 |
|------|------|------|
| 正点 | `t_actual_dep <= t_dep_plan` | 实际离开站台时刻不晚于图定出发时刻 |
| 晚点 | `t_actual_dep > t_dep_plan` | 实际离开站台时刻晚于图定出发时刻 |
| 延误时长 | `max(0, t_actual_dep - t_dep_plan)` | 正点 = 0；晚点 = 超出图定的秒数 |

### 到达

| 判定 | 条件 | 含义 |
|------|------|------|
| 正点 | `t_actual_arr <= t_arr_plan` | 实际到达站台时刻不晚于图定到达时刻 |
| 晚点 | `t_actual_arr > t_arr_plan` | 实际到达站台时刻晚于图定到达时刻 |
| 延误时长 | `max(0, t_actual_arr - t_arr_plan)` | 正点 = 0；晚点 = 超出图定的秒数 |

> **注意**：Layer 2 调度层面 `t_actual_dep` 对应于 `t_out_start`（出站咽喉起点）；`t_actual_arr` 对应于 `t_in_start`（进站咽喉起点，即进入受控区的时刻）。

---

## Layer 1 调度边界 (Dispatch Bounds)

| 参数 | 中文 | 含义 |
|------|------|------|
| `dep_plan` / `t_dep_plan` | 计划出发时刻 | 列车按时刻表离开站台的计划时间 (s) |
| `arr_plan` / `t_arr_plan` | 计划到达时刻 | 列车按时刻表到达站台的计划时间 (s) |
| `arr_min` / `t_arr_min` | 最早到达硬限制 | 列车进入受控区的最早允许时刻，由物理或时刻表约束决定 |
| `dep_max` / `t_dep_max` | 最晚出发硬限制 | 列车离开站台的最晚允许时刻，`dep_plan + dep_max_slack_sec` |
| `dwell_min` | 最小停站时间 | 列车在站台必须停够的最短时间 (s) |
| `dep_max_slack_sec` | 出发最大松弛 | 允许比计划出发时刻晚的最大秒数 (=120s) |
| `prio_i` | 优先级 | 1 = G/D/C (最高), 2 = Z/T, 3 = K (最低) |
| `pass_flag` | 通过标志 | True = 通过列车（不停站），dwell 强制为 0 |
| `P_i` | 候选站台集合 | 该列车在此站可使用的站台 ID 列表 |

## Layer 2 动作空间 (Action Space)

| 参数 | 中文 | 含义 |
|------|------|------|
| `platform_choice` | 站台选择 | 从候选站台 P_i 中选择一个 |
| `entry_choice` | 入站时机选择 | {最早, 按计划, 计划+1min, 计划+2min, 计划+4min} |
| `dwell_extend` | 停站延长 | 在 dwell_min 基础上额外延长 {0, +60, +120} 秒；通过列车强制为 0 |

## 时间窗口 (Time Windows)

| 参数 | 中文 | 含义 |
|------|------|------|
| `W_in` | 入站咽喉窗口 | 列车占用进站咽喉道岔的时间区间 `(t_in_start, t_in_end)` |
| `W_plat` | 站台占用窗口 | 列车占用站台的时间区间 `(t_plat_start, t_plat_end)` |
| `W_out` | 出站咽喉窗口 | 列车占用出站咽喉道岔的时间区间 `(t_out_start, t_out_end)` |

## 观测空间 (Observation, 74-dim)

| 参数 | 中文 | 含义 |
|------|------|------|
| `t_now` | 当前仿真时间 | 当前仿真时钟 (s) |
| `delay_i` | 延误 | max(0, actual_dep - dep_plan) |
| `bound_slack` | 边界余量 | 距离 dep_max 的剩余松弛时间 |
| `K_max` | 邻居上限 | 观测中最多包含的冲突邻居数量 (=5) |
| `validity_mask` | 有效性掩码 | 1=真实邻居, 0=填充占位 |

## 资源管理 (Resource Management)

| 参数 | 中文 | 含义 |
|------|------|------|
| `conflict_group` | 冲突组 | 共享同一道岔资源的咽喉路线集合，组内不能同时被占用 |
| `ResourceCalendar` | 资源日历 | 按资源 ID 记录已被授权的占用窗口 |
| `WindowRequest` | 窗口请求 | L2 策略输出转换后的资源占用请求 |
| `GrantResult` | 授权结果 | 冲突解决后实际分配给列车的资源窗口 |
| `fallback` | 降级授权 | 冲突失败后被分配到备选站台或延迟槽位 |

## 列车属性 (Train Attributes)

| 参数 | 中文 | 含义 |
|------|------|------|
| `train_class` | 列车等级 | G(高速), D(动车), Z(直特), K(普快) |
| `direction` | 运行方向 | up = A→B, down = B→A |
| `origin_station` | 始发站 | 受控区入口所在的站 |
| `destination_station` | 目的站 | 受控区出口所在的站 |
| `corridor` | 走廊 | 两站之间的线路段标识 |
| `pass_through` | 通过模式 | 列车在该站是否不停车通过 |

## 训练参数 (Training)

| 参数 | 中文 | 含义 |
|------|------|------|
| `decision_interval_sec` | 决策间隔 | L2 调度决策步长 (=10s) |
| `control_step_sec` | 控制步长 | L3/SUMO 执行步长 (=1s) |
| `TRIGGER_THRESHOLD_SEC` | 触发阈值 | 延误超过此值触发 L2 激活 (=30s) |
| `λ_team` | 团队奖励权重 | 全局效率信号在混合奖励中的权重 |
| `DEADLOCK_THRESHOLD` | 死锁阈值 | 所有 agent 无合法动作持续此步数后终止 episode (=15) |
