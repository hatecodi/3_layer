# L2 层设计文档 (已完成)

## 架构概览

L2 是三层铁路调度架构中唯一的学习层。每 10s 一个决策步（`decision_interval_sec=10`），L2 为当前站所有 pending 列车产出调度决策，通过确定性 resolver 解决冲突，产出 GrantResult，然后列车被推送到下一站。

```
L1 (规则)                    L2 (MAPPO + Resolver)
timetable.yaml               ┌──────────────────────────────────────┐
  bounds ──────────→ action_mask (硬约束)                           │
  candidate_plats ─→ action_mask                                   │
                             │  observation(29) → FactoredActor     │
                             │    → (platform_idx, entry_idx)       │
                             │    → action_to_request()             │
                             │    → WindowRequest                  │
                             │    → resolve_all()                  │
                             │    → GrantResult                    │
                             │    → _advance_trains() (纯时间加法)  │
                             └──────────────────────────────────────┘
```

**无 L3**：走廊运输 = `g.W_out[0] + runtime` 一行加法。无 SUMO，无物理仿真。

---

## 一、网络拓扑

```
A(南昌西) ←90s→ B(鹰潭北) ←90s→ C(上饶) ←180s→ D(杭州东) ←180s→ E(南京南)
 4p             4p             4p             2p (瓶颈)      4p
```

10 列车 (symmetric-5v5)：G1380, G230, G1616, G238, K834 (UP) + G7629, G2245, G1487, G1347, K1185 (DOWN)

---

## 二、L1 → L2 接口

### 数据源：`configs/five_station/timetable.yaml`

每条列车每站有 3 组数据：

```yaml
planned:        # 计划时刻 (raw_timetable / 25, 无dwell_min膨胀)
  arr: 487
  dep: 497
bounds:         # L1 硬边界
  arr_min: 367  # 最早允许到达
  dep_max: 617  # 最晚允许出发 (= planned_dep + slack_120s)
  dwell_min: 5  # 最短停站时间 (2min真实 ≈ 5s模拟)
candidate_platforms: [D_p1, D_p2]
```

### 始发站 vs 中间站

- **始发站**：bounds 直接使用 YAML 中的预计算值
- **中间站**：`_get_effective_bounds()` 动态计算 (`five_station_env.py:554-578`)
  - `arr_min = 列车实际到达时间`
  - `dep_max = max(ref_time + dwell_min + slack, physical_min_dep)`

### L1 bounds 的唯一作用路径：action_mask

```
compute_joint_action_mask() → is_legal_action() → 检查:
  platform ∈ candidate_platforms
  t_in_start >= arr_min
  t_out_start <= dep_max
  dwell >= dwell_min
  through_allowed 检查
→ 非法动作在 softmax 前 mask 为 -inf
```

**核心法则**：硬边界通过 mask 强制执行，不在 reward 中惩罚边界违反。

---

## 三、Observation Space (29 dims)

`compute_obs_dim()`: `14(self) + 5(next) + 2(fixed_res) + max_plat*2(blocked+contention)`

| 段 | 维度 | 内容 | 归一化方式 |
|----|------|------|-----------|
| Self | 10+4 | `t_now/1500`, `cum_delay/600`, `plan_arr/1500`, `plan_dep/1500`, `arr_min/1500`, `dep_max/1500`, `dwell_min/120`, `prio/3`, `pass_flag`, `slack/300` + 4-dim class one-hot | 各特征独立scale |
| Contention | max_plat | 每站台：其他pending列车与"我"dep时间±30s内重叠的比例 | `n_overlap/3`, cap at 1.0 |
| Next | 5 | `avail/plat_count`, `n_heading/n_trains`, `est_arr/1500`, `congestion`, `dwell/120` | 各特征独立scale |
| Resource | 2+max_plat | `avail_now/plat_count`, `corr_delay/180`, per-platform blocked flags (visitor在[t_now, t_now+600]内占用) | binary flags |

**platform_contention 是关键创新**（coding_log 2026-07-01）：将 `n_at_station`(1维标量) 替换为 per-platform contention(4维)，dep_delay 从 14.1s 降到 3.1s (-78%)。

---

## 四、动作空间

### Factored Action

```
action = (platform_idx: 0-3, entry_idx: 0-5, dwell_extend: 0)

platform_idx: candidate_platforms 中的索引 (≤4, padding到4)
entry_idx:    ENTRY_CHOICES 索引
  ["earliest", "plan", "plan+1", "plan+2", "plan+3", "plan+4"]
  共6种, 每种差60s

dwell_extend: 始终为0 (min dwell始终最优, 无物理约束需要额外dwell)
```

总 logits: 4(platform) + 6(entry) = 10（之前 flat 方案为 4×6×3=72）

### 为什么是 Factored 不是 Flat

之前尝试过 Top-K options 方案（每个 option = 完整 schedule），heuristic=0s delay，MAPPO无事可学。回退到 factored action + resolver 后恢复。关键洞察：platform选择和entry选择是条件独立的——站台选择取决于资源可用性，entry选择取决于时间偏好。分解后探索空间从72降到10。

---

## 五、Resolver — 确定性冲突解决

### `resolve_all()` 算法 (`resource_manager.py:238-301`)

```
1. 按优先级排序请求: priority(1>2>3) → -delay(延迟大者受保护) → train_id(字典序)
2. 逐个尝试 grant:
   - 检查 _resource_windows 是否与已 grant 窗口冲突 (_overlap 检测)
   - 无冲突 → 直接 grant, book 到 ResourceCalendar
   - 有冲突 → assign_fallback():
     (a) 优先换其他站台
     (b) 换站台不行 → 在同站台上延迟出发
     (c) 最多 2 次 fallback 尝试, 仍失败 → hold
3. Visitor 预占通过 parked_windows 在 calendar 初始化时注入
```

### `_resource_windows` 内容

```python
{
  "D/D_p1":        W_plat,     # 站台
  "D/D_up_in_g1":  W_in,       # 进站咽喉冲突组
  "D/D_up_out_g1": W_out,      # 出站咽喉冲突组
}
# 走廊不在 _resource_windows 中 — 由未来 L3 负责 headway
```

### Resolver 是确定性规则，MAPPO 学的是什么？

MAPPO 学会"什么时候选哪个站台 + 什么时机进站"以**避免被 resolver 降级为 fallback**。选空闲站台 → 直接grant → 0 delay → 高reward。选拥挤站台 → resolver冲突 → fallback → delay → 低reward。

---

## 六、Network 架构

```
Layer2ActorCritic
├── FactoredActor (MaskedActor)
│   ├── LayerNorm(29)
│   ├── shared: Linear(29→128) → Tanh → Linear(128→128) → Tanh
│   ├── platform_head: Linear(128→4)   ← Categorical, masked
│   └── entry_head: Linear(128→6)      ← Categorical, masked
│
├── LocalCritic: LayerNorm((K+1)*29) → MLP → 1
│   K=1: 输入 58 dims (自己 + 1个冲突邻居的obs)
│
└── GlobalCritic: LayerNorm(14) → MLP → 1
    输入: compact_global_state(14 dims)
```

---

## 七、Reward

```python
local (per-train):
  granted: -0.10 * |dep_delay| * prio_weight * (1 + |delay|//300)   # 渐进惩罚
           + 2.0 * exp(-|dep_delay|/60) * prio_weight               # ontime bonus
  held:    -3.0

global (team):
  -0.10 * avg_dep_delay - 100.0 * deadlock_flag + 10.0 * ontime_rate

total = local + λ_team * global
λ_team: curriculum 0.05→0.30
ONTIME_THRESHOLD: 15s (|departure_actual - departure_plan| <= 15s)
```

关键：`dep_delay = |实际出发 - 计划出发|`，同时惩罚早到和晚到。

---

## 八、Visitor 列车

每 episode 开始时随机生成 visitor，预占 A/B/C/E 站台：

```yaml
visitors:
  enabled: true
  target_stations: [A, B, C, E]  # D 是瓶颈, 不加visitor
  prob_per_slot: 0.3
  slots_per_station: [1, 2]
  k_ratio: 0.6      # 60% K类 (prio=3, 停300-900s)
  g_dwell: [120, 300] # 40% G类 (prio=2, 停120-300s)
```

预报信息在 observation 中可见（`platform_blocked` flags），policy 可以学会避开被占站台。

---

## 九、TrainJourney 状态机

```python
TrainJourney:
  route: ["A", "B", "C", "D", "E"]   # 车站序列
  current_idx: int                     # 当前位置
  arrival_at_current: float            # 到达当前站的时间

状态:
  arrived & pending → L2决策 → granted → advance → 下一站
  is_done → 移除
```

`batch_window_sec=120` 让即将到达的列车提前参与当前站决策，形成真正的多智能体竞争。

---

## 十、训练 HP

```python
HP = {
    "obs_dim": 29,
    "k_local": 1,
    "hidden_dim": 128,
    "lr": 3e-4,
    "gamma": 0.99, "gae_lambda": 0.95,
    "clip_ratio": 0.2, "ppo_epochs": 5,
    "minibatch_size": 256,
    "entropy_coef": 0.01 → 0.003 (前30% episodes线性衰减),
    "value_coef": 0.5,
    "max_grad_norm": 1.0,
    "rollout_steps": 512,
    "batch_window_sec": 120.0,
}
```

---

## 十一、已知局限

1. **D 站瓶颈**：2 站台，3+ 列车同时竞争时无法避免冲突
2. **Visitor 随机性**：某些 seed 的 visitor 分布导致不可恢复的阻塞 → truncation
3. **无 L3 物理**：走廊 = 纯时间偏移。无 headway、无咽喉穿越时间、无速度曲线
4. **dwell 退化**：dwell=0 始终最优（因无物理约束需要 extra dwell），entry 只有 timing 选择有意义
5. **单方向 corridor**：上下行用不同 corridor ID，无跨方向 conflict。现实中双向单线或会让站的情况未建模
