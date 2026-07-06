# L3 层约束设计与数据需求

## L3 的职责

L2 产出的是"理想调度计划"（每列车、每站、什么时间、用哪个站台）。L3 在有物理约束的环境中执行这个计划，产出**实际的**到达/出发时间。

```
L2 PlatformStrategy (YAML/内存)
        │
        ▼
L3 Python Sim (纯Python, 无SUMO依赖)
  - 事件驱动时间推进
  - PhysicalCalendar: 站台 + 咽喉 + 走廊 物理占用
  - 每列车状态机: APPROACHING → IN_THROAT → AT_PLATFORM → OUT_THROAT → IN_CORRIDOR → ...
        │
        ▼
L3ExecutionResult
  - per-train per-station 实际时间
  - L2计划 vs L3实际 delta (execution fidelity)
  - headway违反、deadlock报告
```

---

## 一、物理约束分类

### A) 咽喉约束 (Throat)

```
车站物理结构:
         ╔════════╗
  approach║in-throat║══ platform ══ out-throat ══→ corridor
         ╚════════╝
```

| 约束 | 描述 | 参数来源 |
|------|------|---------|
| 咽喉互斥 | 同一 conflict_group 同时只能有一列车占用 | 已有: `railway_resources.yaml` 中的 `conflict_group` |
| 咽喉穿越时间 | 列车通过咽喉需要时间 | 需要: `throat_length` / `throat_speed` |
| 咽喉清空间隔 | 前车完全出清咽喉后，后车才能进入 | 需要: `throat_min_gap` (s) |
| 进-出咽喉联动 | 同一物理道岔区的进出操作互斥 | 需要: 定义哪些 in_group 和 out_group 共享物理道岔 (可选) |

**建模粒度**：使用现有的 `conflict_group` 级别，不需要建模到单个道岔。咽喉抽象为一段独占轨道，长度固定，速度受限。

**论文对应**：道岔区作为"线路端点 (track endpoint)"，是车站容量的关键瓶颈。咽喉容量约束 = "同一冲突组内，连续两列车的咽喉占用时间窗不能重叠 + 必须有安全间隔"。

### B) 走廊约束 (Corridor)

| 约束 | 描述 | 参数来源 |
|------|------|---------|
| 走廊追踪headway | 同向同走廊上，后车与前车必须保持最小时间间隔 | 需要: `corridor_headway_min` (s) |
| 走廊穿越时间 | 站间运行时间 | 已有: `planned[nxt].arr - planned[cur].dep` (时刻表) |
| 越行禁止 | 双线单向走廊，后车不能超过前车 | 隐含: headway + 先入先出 |
| 双向走廊互斥 | 如果某段是双向单线，上下行列车互斥 | 需要: 定义哪些 corridor 是单线 (可选) |

**当前假设**：所有走廊都是双线单向（up/down 用不同的 corridor ID: `A_B_up` vs `B_A_down`），所以上下行之间无冲突。如果未来要建模单线区段，需要合并 corridor ID 或引入跨方向互斥规则。

**论文对应**：走廊追踪间隔 = "闭塞区间 (block section) 通过时间" = 最小 headway。这是铁路运行图压缩的核心参数。

### C) 站台约束 (Platform)

| 约束 | 描述 | 参数来源 |
|------|------|---------|
| 站台互斥 | 同一站台同时只能有一列车 | 已有: `ResourceCalendar` 机制 |
| 站台清空间隔 | 前车离站后，后车进同一站台需要最小间隔 | 需要: `platform_min_gap` (s) |
| 站台长度 | 不同列车长度需要匹配站台长度 | 可选: 当前假设所有站台都够长 |
| 站台折返 | 列车在终点站需要折返时占用站台的额外时间 | 可选: 当前所有列车不折返 |

### D) 速度约束

| 约束 | 描述 |
|------|------|
| 进站接近速度 | 低于走廊速度，列车准备进站时减速 |
| 咽喉限速 | 通过道岔时速度受限 |
| 走廊巡航速度 | 站间稳定运行速度 |
| 出站加速 | 出站后加速到巡航速度 |

**当前简化**：L3 用恒定段速度 (approach_speed, throat_speed, corridor_speed)，不建模加速/减速过程。这样可以把每个物理段建模为固定穿越时间 = length / speed。

---

## 二、关键设计决策

### 2.1 咽喉建模：端点法 vs 轨道段法

**端点法**（推荐）：把道岔区视为车站的两个端点（进站端、出站端），每个端点有 1-2 个冲突组。咽喉互斥 = 同一冲突组的时间窗不重叠。

**轨道段法**：把道岔区建模为带道岔节点的轨道图，每段轨道有独立的占用状态。更精确但更复杂。

**推荐端点法**，原因：
- 与现有 `conflict_group` 配置直接兼容
- 论文中可引用"station throat as bottleneck node"的标准模型
- 实现简单：PhysicalCalendar 检查同一 conflict_group 的时间窗 + min_gap

### 2.2 时间推进：固定步长 vs 事件驱动

**事件驱动（推荐）**：找出下一个会发生状态转换的列车，直接跳到该时间点。效率最高，无"空间隔"。

**固定步长**：每 dt 秒检查所有列车状态。实现简单但效率低。

**推荐事件驱动**，原因：
- 调度场景中事件密度低（大多数时间列车在走廊上运行）
- L3 可能未来集成到训练回路中，需要高效
- 离散事件仿真 (DES) 是铁路仿真的标准方法

### 2.3 与 SUMO 的关系

```
训练阶段: L2 (MAPPO) → L3 Python Sim (快速, 无SUMO依赖)
验证阶段: L2 最优策略 → SUMO TraCI (真实物理, paper结果)
```

Python Sim 用于快速迭代。SUMO 用于最终验证——两个环境应该接受相同的 PlatformStrategy 输入，产出可比较的 L3ExecutionResult。

---

## 三、需要的数据/参数

### 3.1 已有的（从现有配置可读取）

| 参数 | 来源文件 | 字段 |
|------|---------|------|
| 站台列表 | `railway_resources.yaml` | `stations[].platforms[]` |
| 咽喉冲突组 | `railway_resources.yaml` | `stations[].up/down.in_routes[].conflict_group` |
| 走廊方向 | `railway_resources.yaml` | `corridors[].direction` |
| 走廊长度 | `railway_resources.yaml` | `network.corridor_runtime_sec` (以时间表示) |
| 站间计划时间 | `timetable.yaml` | `planned[nxt].arr - planned[cur].dep` |

### 3.2 需要新增的物理参数

创建 `configs/five_station/l3_physics.yaml`:

```yaml
l3_physics:
  # ── 咽喉 ──
  default_throat_length: 80.0       # m, 咽喉区长度 (道岔+连接线)
  default_throat_speed: 10.0        # m/s, 咽喉限速 (约36 km/h, 道岔限速典型值)
  throat_crossing_time: 8.0         # s, throat_length/throat_speed, 也可直接配置
  throat_min_gap: 15.0              # s, 前车出清咽喉后到后车进入的安全间隔

  # ── 进站接近 ──
  approach_length: 200.0            # m, 进站减速段长度
  approach_speed: 15.0              # m/s, 进站接近速度 (约54 km/h)
  approach_time: 13.3               # s, approach_length/approach_speed

  # ── 走廊 ──
  corridor_speed: 30.0              # m/s, 巡航速度 (约108 km/h, 高铁降速后的模拟值)
  corridor_headway_min: 120.0       # s, 最小追踪间隔 (闭塞区间长度/走廊速度)
  # 如果不同走廊有不同的headway:
  corridor_headway:
    A_B: 120
    B_C: 120
    C_D: 150   # 长走廊需要更大headway
    D_E: 120

  # ── 站台 ──
  platform_min_gap: 10.0            # s, 前车离站到后车进同一站台的清空间隔
  platform_stop_buffer: 5.0         # s, 计划停站时间外的额外缓冲 (开门/关门等)

  # ── 可选: 如果要按站独立配置 ──
  per_station:
    D:                             # D站是瓶颈, 咽喉更复杂
      throat_crossing_time: 10.0
      throat_min_gap: 20.0
```

### 3.3 需要确认的数据（如果你有真实数据）

| 数据 | 用途 | 当前值（估计） | 需要真实值？ |
|------|------|-------------|------------|
| 道岔限速 | 咽喉穿越时间 | 36 km/h | 建议用真实值 |
| 闭塞区间长度 | 走廊 headway | 无, headway直接给120s | 如果有闭塞区间数据可以算精确headway |
| 站台长度 | 匹配列车长度 | 假设所有站台够长 | 不是必须 |
| 列车长度 | 占用站台时间 | 不建模 | 不是必须 |
| 咽喉实际长度 | 穿越时间 | 80m | 建议用真实值或估计值 |
| 进站信号机位置 | approach 长度 | 200m | 可以用估计值 |

---

## 四、L3 Python Sim 核心数据结构

### 4.1 输入

```python
@dataclass
class L2StopPlan:          # 从 PlatformStrategy 提取
    train_id: str
    station_id: str
    platform: str           # e.g. "D_p1"
    t_throat_entry: float   # L2计划的进站咽喉入口时间
    t_platform_arrival: float
    t_platform_departure: float
    t_throat_exit: float    # L2计划的出站咽喉出口时间
```

### 4.2 状态机

```
IN_CORRIDOR ──→ APPROACHING ──→ IN_THROAT ──→ AT_PLATFORM ──→ OUT_THROAT ──→ IN_CORRIDOR ──→ ...
    ↑                                                              │
    └──────────────────────────────────────────────────────────────┘
                              (循环 to next station)
DONE: 所有 station 完成
```

### 4.3 输出

```python
@dataclass
class L3StopRecord:        # L3实际执行的记录
    station_id: str
    platform: str
    # 实际时间
    t_throat_entry_actual: float
    t_platform_arrival_actual: float
    t_platform_departure_actual: float
    t_throat_exit_actual: float
    t_corridor_exit_actual: float
    # 延迟归因
    delay_total: float       # actual - L2 plan (正=比计划晚)
    delay_throat_wait: float # 等咽喉空出的时间
    delay_headway: float     # 走廊 headway 造成的延迟
    delay_platform_wait: float # 等站台空出的时间
    was_rerouted: bool       # L3是否因为阻塞改了L2分配的站台 (通常不应发生)

@dataclass
class L3ExecutionResult:
    train_results: Dict[str, L3TrainResult]
    completion_rate: float
    l2_planned_ontime: float    # L2策略声称的正点率
    l3_actual_ontime: float     # L3执行后实际正点率
    execution_fidelity: float   # 1.0 = L3完美执行了L2的计划
    headway_violations: int
    deadlock_flag: bool
```

### 4.4 PhysicalCalendar

```python
class PhysicalCalendar:
    """L3 物理资源日历。与 L2 ResourceCalendar 类似但追踪实际占用。"""
    def occupy(resource_id, start, end, train_id)
    def is_free(resource_id, start, end, gap=0.0) -> bool
    def next_free(resource_id, after, gap=0.0) -> float
    def headway_ok(corridor_id, train_id, entry_time) -> bool
      # 检查: 同corridor上, 前车 entry_time + transit_time + headway_min <= 后车 entry_time
```

资源 ID 格式：
- 站台: `"{station}/{platform}"` (已有)
- 咽喉: `"{station}/{conflict_group}"` (已有)
- 走廊: `"{corridor_id}"` (新增)
- approach: `"{station}/approach"` (新增)

---

## 五、与 L2 的关键差异

| | L2 | L3 |
|------|----|----|
| 时间推进 | 固定 10s 步 | 事件驱动 (跳到下一个状态转换) |
| 咽喉建模 | `THROAT_CLEARANCE_TIME=0` | 有限穿越时间 + 互斥 + 清空间隔 |
| 走廊建模 | 纯时间偏移 | 追踪 headway + 有限穿越时间 |
| 冲突解决 | 确定性 resolver (降级/fallback) | 物理等待 (列车等到资源空闲) |
| 失败处理 | hold → 下个 step 重新决策 | 等待 → 自动推进（除非死锁） |
| 决策能力 | 有 (MAPPO) | 无 (纯规则, 照L2计划执行) |

---

## 六、值得参考的文献方向

如果你要补充相关文献，以下方向最相关：

1. **铁路咽喉容量建模**
   - "Station throat capacity" / "station bottleneck" 相关论文
   - 常见方法: 将咽喉建模为带容量的节点 (node-capacity model) 或排队网络 (queueing network)

2. **离散事件铁路仿真 (DES)**
   - RailSys, OpenTrack 等商业工具的技术白皮书
   - 学术界的轻量级铁路 DES 实现 (如用 SimPy)

3. **三层/分层调度架构**
   - Hierarchical production planning (HPP) in manufacturing
   - 铁路运行图压缩中的分层优化 (strategic → tactical → operational)

4. **MARL + 约束执行**
   - Safe RL / constrained MDP 相关
   - Action masking as hard constraint enforcement

你需要我帮你找具体的论文引用吗？
