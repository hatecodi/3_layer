# 三层分层 MARL 铁路调度框架 — 架构描述（论文引用用）

> 本文档描述 `three_layer_railway_marl_fixed_1` 代码库中实现的三层分层 MAPPO 铁路调度
> 框架的完整架构：L1→L2→L3 的层次划分、各层动作/观测/网络、安全性机制、奖励函数、
> MAPPO 结构，以及训练流程。写论文时可直接引用其中的术语、公式与超参数。
> 代码位置：`src/control/`、`src/env/`、`src/marl/`、`configs/five_station*/`。
>
> **符号与术语以 `docs/notation.md` 为准**（第三、四章统一的符号对照表：符号 / 含义 /
> 首见 / 代码对应）。本文档描述实现，凡与 `notation.md` 不一致处，以 `notation.md` 为准。
> 另注：$L_2$ 的准确表述是**规则基座 + 学习残差复合而成的决策层**（非单一学习策略），
> 详见 `docs/problem_formulation.md` §5 与 `docs/ch4_algorithm_design_paper.md` §4.2.1。

---

## 1. 总览

框架把铁路调度问题分解为三层，**越下层越接近物理执行，越上层越接近战略决策**：

```
L1  调度边界层  Dispatch Bounds   （规则、固定、无学习）
     │  每列车在每个站的硬时间包络：t_arr_min / t_dep_max / dwell_min /
     │  candidate_platforms / 越行标记 pass_through
     ▼
L2  调度层      MAPPO 调度        （学习：平台分配 + 进站时机）
     │  在 L1 边界内选 (platform, entry)；输出 WindowRequest → 资源日历冲突消解
     │  授权后列车进入走廊
     ▼
L3  走廊速度层  per-corridor Worker（学习：逐区间 CRUISE/FAST/SLOW 速度曲线）
     │  移动闭塞(moving-block)车距为硬边界；实际到站时间反馈给下一站调度
```

- **时间压缩**：仿真秒 = 真实秒 / 25（一个决策步≈实际分钟级）。
- **研究问题**：三层分解（规则边界 → MAPPO 调度 → 学习速度控制）能否在保持低延误、
  零硬约束违反的同时，比 flat MAPPO 更好地扩展到更多列车。


---

## 2. 环境与网络

- **网络**：A–E 五站双线走廊，4 站台/站（D 站 2 站台），每列车沿一条走廊从起点到终点。
- **走廊**：A→B 5600m、B→C 4040m、C→D 13640m、D→E 10240m（压缩后米）。
  每条走廊被分解为 **4 个区间(section)**，末段带进站限速 `limit_mps=70`（≈SLOW 速度，
  250km/h 进站）。区间本质是**速度区段（speed zone）**，不是占用闭塞分区。
- **列车**：按优先级分 3 类——P1 复兴号（cruise 97.22 m/s / max 111.11 m/s）、
  P2 和谐号（83.33/97.22）、P3 普速（33.33/33.33，无恢复裕量）。
- **干扰**：可注入两种扰动——始发"成串延误"(bunching) 与 **走廊暴雨限速 incident**
  （详见 §2.1）。论文扩展性/收敛实验均使用 incident 作为唯一扰动。
- **观测维度**：L2 obs_dim=34（2026-11-06 起：+3 维"后方高优先级来车"让行特征），与规模无关（10t–30t 一致）；L3 obs_dim=10。

### 2.1 走廊限速 incident（暴雨限速）

实验的核心扰动：随机走廊出现**临时限速 = 移动瓶颈**，窗口覆盖的区间变慢，经过的车队被
压缩（platoon → re-dispersion），给 L3 协调与 L2 重调度创造真实压力。

**配置**（`configs/*/timetable.yaml` → `time_generation.corridor_incident`）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `enabled` | false（opt-in） | 需显式 `--incident` 打开，不改既有基线 |
| `prob` | 0.6 | 每集发生概率 |
| `duration_range` | [100, 200] sim-s | 限速窗口时长 |
| `factor_range` | base [0.15,0.4]；15t [0.4,0.6] | **速度因子**（1.0=无限速，越小越慢）：0.15≈大暴雨45km/h，0.4≈暴雨120km/h |
| `safety_factor` | **1.0** | **湿轨制动**：moving_block_safety_m 乘数。原为 1.5（**估算值**），已放宽到 1.0——放大车距会把限速变成硬移动闭塞瓶颈，挤掉恢复空间（配置内注明） |
| `min_trains` | 2 | 窗口至少覆盖 ≥2 次通过 |
| `max_added_delay_sec` | **80** | 校准上限：保持延误可恢复。原为 200，已下调至 80，使其**落在重规划触发阈值之下**，避免扰动经重规划回路雪崩（配置内注明） |

**生成流程**（`disturbance.py: inject_corridor_incident`）：
1. 按概率抽样是否发生；随机选走廊（**按 transit 流量加权**——流量大的走廊更易出现）。
2. 窗口锚定到该走廊的真实 transit 流量，保证覆盖 ≥`min_trains` 次通过。
3. 速度因子随机落在 `factor_range`，然后**校准**到最坏受影响车的附加延误 ≤
   `max_added_delay_sec`（保证 L3 速度控制可恢复）：
   - 完全覆盖（transit ≤ duration）：`factor ≥ 1/(1 + max_added/transit_est)`
   - 部分覆盖：`factor ≥ 1 - max_added/duration`
4. 输出 `{corridor_id: (start, end, speed_factor, safety_factor)}`。

**逐段应用（section-resolved）**：`l3_sim.py: incident_section_factors(start, end, f_inc,
sf_inc, t, run_est)`——限速**只作用于窗口覆盖的区间**（该区间遍历 [t, t+run_est] 与窗口重叠），
其余区间不受影响。原因：整趟限速会把受影响列车变成慢速前车，在连续 moving-block 下堆起
无界队列（雪崩）。应用方式：`speed_factor` 缩放区间运行时间（`v *= speed_factor`），
`safety_factor` 放大移动闭塞安全距离（`moving_block_safety_m × safety_factor`）。

**severity 参数**（`--severity N`，评估时用，不改基础配置）：>1 时加强 incident——
`max_added_delay_sec × N`（允许更大延误）、`factor_range ÷ N`（更慢）、`duration × N`。
目的：把受影响延误推到 `dep_max_slack` 之外，**打开 L2 真正的重调度窗口**。默认 1.0。

**硬守卫交互**：`replan_max_headway_delay_sec=1000`（2026-08-14 上调）——限速导致的
moving-block 车距等待本质是**队列**而非重规划问题，重规划只会在队尾重累延误（雪崩），
因此压制 incident 诱导的 L3 重规划。

**观测信号**：L3 观测最后一位是 `speed_factor`（0=无限速，>0=受限），worker 据此在受限区间
选 FAST 恢复（实证：限速段 FAST 使用率 ~76%，匹配 SA）。

**评估协议**：`eval_protocol --incident --n 60 --blocks 3 --severity 1.0` = 3 个独立种子块 ×
60 集，incident 为唯一扰动（禁用始发 bunching）。论文的扩展性/收敛/延迟数据均在此协议下。

---

## 3. L1 — 调度边界层（规则，固定，无学习）

每个列车在每站的**硬时间包络**由时刻表配置自动展开（`dispatch_bounds.py`）：

```
t_arr_min   最早允许到达（不能早于）
t_dep_max   最晚必须出发（不能晚于）
dwell_min   最小停站时间
prio_i      优先级 1/2/3
pass_flag   该站是否通过不停车（逐站；零计划停站 + dwell_min=0 + 该站预留正线）
candidate_platforms  允许的站台集合（通过车→正线；停站车→其余站台）
```

- L1 不发出逐步指令，只提供约束。
- **L2 的每个动作都必须满足 L1 全部硬边界**；违反边界的动作在采样前被掩码为 `-inf` logits
  （见 §6 安全性），**边界不是奖励项**。

---

## 4. L2 — MAPPO 调度层

### 4.1 动作空间（因式分解，条件离散）

`action_mask.py`：每列车在每个决策步的动作是 **平台 × 进站时机** 的条件离散组合：

```
a_i = (platform_choice, entry_choice)
platform:  ≤4 个候选站台
entry:     6 个时机 —— "earliest" / "plan" / "plan+10" / "plan+20" / "plan+30" / "plan+50"
           （"plan+N" = 计划进站时间 + N 压缩秒，N∈{10,20,30,50}）
dwell:     恒为最小停站（5s），不扩展
```

动作 → 具体时间窗（`derive_schedule`）→ 资源占用窗（`action_to_request` →
`WindowRequest`，含进咽喉/站台/出咽喉三段占用，**走廊不入资源窗**，见 §6）。

### 4.2 观测（31 维）

`observation_builder.py`，配置驱动：

```
SELF(16)      时间、延误、计划、边界、优先级、车型 one-hot、站均延误、剩余裕量
NEXT(5)       下一站平台数、拥挤度、预计到达、停站时长
RESOURCE(2+2*max_plat=10)  资源可用性、走廊延误、每平台 blocked 标志、每平台 contention
```

演员只看"自己该知道的事"；评论家通过 `critic_input()` 看局部冲突图邻居。

### 4.3 网络结构

`actor_critic.py`：

- **FactoredActor**：共享的多智能体演员，双头输出——平台头(4 logits) + 进站头(6 logits)，
  共 10 logits（对比 flat 的 72 logits）。平台头 bias 初始化为偏向 p1（与启发式同先验）。
- **LocalCritic**：输入 `(k_local+1)×obs_dim`（k_local=1 → 62 维），取**局部冲突图邻居**的
  拼接观测，O(1) 扩展。
- **GlobalCritic**：14 维全局状态（活跃数、均延误、死锁标志、完成率、准点率等），共享。
- `Layer2ActorCritic` 组合三者。

### 4.4 冲突消解

`resource_manager.py`：所有 L2 授权写入持久 `ResourceCalendar`（min_gap=5s）。
冲突按确定性规则消解：**优先级（P1 胜 P3）→ 高延误胜 → train_id 字典序**。
败者得到最早可行的备选站台（或同站台延后），**备选必须重新校验 L1 边界**
（`assign_fallback` 用 `RebuildFn` 重建 `WindowRequest`）。跨站台 fallback 带
`swap_cost_sec` 物理税（真实延误进入奖励与到站），使 L2 有梯度避开冲突。

---

### 4.1 车站越行与学习让行（2026-11-06）

- **逐站通过**：若某车在中间站 `planned.arr == planned.dep`、L1 `dwell_min=0`，且该站配置
  预留了正线（`through_allowed` 混合），则该站视为通过站。通过车只能选正线（A/B/C/E 为 P1），
  停站车只能选其余站台；D 两站台均开放以避免容量死锁。逻辑在
  `scenario_loader.pass_stations_for / apply_pass_platform_policy`，其它场景无混合配置时保持旧行为。
- **让行**：不新增动作头，复用 6 档 `entry_choice`（`plan+10/20/30/50`）延迟进站/出发。
  观测新增 3 维：是否存在更高优先级尾随车、优先级差、其到达与自身计划出发的间隔。
- **物理假设**：车站有越行侧线；模型把站台进出与走廊占用分开建模，因此允许停站车在站台
  等待时通过车走正线。越行只在两车同一决策步均 pending（≤120s）时成立——`_advance_trains`
  在授权后立即推进，跨步的站内停留无法表达。
- **注意**：L3 走廊仍是单方向 FIFO，本机制只给 L2 增加重排序自由度，不在走廊内超车。
- **指标**：`overtake.csv` / `per_episode.csv`：overtakes、deliberate_yields、
  priority_overtakes、pass_overtakes、pw_arr_delay；transit 事件带 `front_train`。

---

## 5. L3 — 走廊速度控制层

### 5.1 动作

逐区间选择速度档：`CRUISE / FAST / SLOW`（`SpeedMode` 枚举）。
列车进入走廊后，对每个区间 `k` 选一个档位，形成**逐区间的分段速度曲线**
（不是整趟一个模式）。

### 5.2 观测（10 维）

`build_l3_section_observation`：车距余量(headway gap)、已累计等待、累计延误、
区间序号/总数、区间长度/总长/剩余里程、速度比、下一站空闲/总站台数、咽喉间隔、
限速因子（incident 信号在最后一位）。

### 5.3 网络结构

`l3_worker.py`：共享 `CorridorActor`（10→64→64→3）+ 两个 `CorridorCritic`
（**critic_local 自身准点 + critic_global 团队**，镜像 L2 的 local/global 设计）。
`L3WorkerTrainer` 是轻量 PPO 循环。

### 5.4 走廊物理（移动闭塞）

`l3_sim.py`：

- **移动闭塞(moving-block)**：跟随车只需与前车尾保持**连续安全距离**
  `moving_block_safety_m=500m` + `moving_block_margin_sec=3s`，**不是等整个区间清空**。
  区间仅作速度区段（分段速度曲线），不做占用闭塞分区。
- 每条走廊还有绝对入口车距下限 `corridor_headway_sec`（配置 5.0s，安全下限，防高速下间距过小）。
- 三种速度模式在逐区间路径上物理统一（stepwise ≡ transit，diff=0），杜绝双算。
- 限速 incident 按区间解析：只对窗口覆盖的区间施加速度因子，整趟不受影响。

---

## 6. 安全性机制

安全性靠**结构保证**，不靠奖励惩罚（设计不变量）：

1. **硬边界通过掩码，不是奖励**：`is_legal_action` 校验所有 L1 边界
   （arr_min 下限、dep_max 上限、dwell_min、候选站台、越行必须用正线/通过站台）。
   非法动作 logits 置 `-inf`，策略采样前不可见。**无边界违反奖励项**。
2. **NO_LEGAL_ACTION → hold/replan**：若某车全部动作被掩码，则保持(hold)并在下一步
   重新规划，**绝不从策略采样**（防死锁式非法动作）。
3. **走廊车距是 L3 硬边界**：走廊**不在** L2 资源窗中，解析器只拥有站台资源
   （站台、进咽喉、出咽喉）。最小追踪间隔由 L3 的 moving-block 车距强制执行，
   **从不作为奖励项**。
4. **站台 min_gap**：平台占用在列车离开后仍保持 `min_gap_sec=5s` 的预定间隔，
   防止后车贴前车进入同一站台。
5. **死锁检测**：`deadlock_flag`（全部活跃车被 hold）触发 episode 终止；
   训练/评估均要求死锁 0。
6. **fallback 重验边界**：冲突败者的备选方案通过 `RebuildFn` 重建请求并重查全部 L1 边界。

安全指标（实证）：所有规模、所有模式 **0 硬守卫违反、0 死锁**。

> **复核说明（2026-09-20）**：`scripts/validate_l3_schedule.py` 的检查曾有四处置错
> （站名取成目的站、throat 只按站分组而忽略 conflict_group、用预串行化的 $W^{\text{out}}$
> 当占用时刻、"站台重叠"误用进/出咽喉窗口 $W^{\text{in}}/W^{\text{out}}$ 而非 $W^{\text{plat}}$），
> 已修正。修正后：**throat 串行化零违规**（实测最小间隔恰为 6.00 s）；clean / `--incident`
> 工况站台窗口零重叠。`--vt` 工况存在跨步站台窗口重叠，但 VT 非论文配置（见 `coding_log`
> 2026-09-20）。引用安全结论时须注明所用工况（论文用 `--incident`）。

---

## 7. 奖励函数

### 7.1 L2 奖励（`reward.py: compute_layered_rewards`）

每个活跃智能体的**局部奖励**：

```
local =  -0.02·dep_delay·prio_weight·dep_mult        # 发车延误，渐进惩罚(每300s翻倍)
        + ONTIME_BONUS·exp(-dep_delay/60)·prio_weight # 连续准点指数衰减
        + ONTIME_HARD_BONUS·prio_weight                # dep_delay≤15s 硬准点奖励(=2.0)
        - 0.01·arr_delay·prio_weight                   # 非起点站到站偏差(早期/晚期)
        - 0.02·headway_delay·prio_weight               # 走廊车距延误(给 L2 间隔发车梯度)
        - 0.02·throat_delay·prio_weight                # 出咽喉串行化延误
        - externality                                   # 挡路前车按后车优先级加权付费
被 hold 或未授权时: local = -3.0
```

**外部性定价**（2026-11-06）：后车在走廊的 moving-block 等待
`Σ headway_delay·w_follower` 记到实际前车头上，前车下一步奖励扣除
`min(EXTERN_ALPHA·ledger/w_leader, 20)`，`EXTERN_ALPHA=0.15`。它让"让行"在自身回报上
也划算（实测 G5038@A：α=0.05 时让行自身回报变差；0.15 时盈亏平衡且系统回报改善）。

**全局(团队)奖励**（episode 级，共享）：

```
global = -0.02·mean_dep_delay - 20·deadlock_flag + 2.0·ontime_rate
```

每智能体总奖励：`total = local + λ_team·global`，`λ_team` 从 0.05 线性爬升到 0.30
（`curriculum_lambda`）。优先级权重 `prio_weight = {1:2.0, 2:1.0, 3:0.5}`。

### 7.2 L3 奖励（终端 + 逐区间）

**逐区间**：只存 transition，奖励在走廊末端一次性结算
（`compute_l3_terminal_reward`），车距是硬边界**不是奖励项**：

```
r = ontime_bonus·exp(-(dev/σ)²)                        # 主项：准点高斯包络，每列车都计
    - [late_coef·dev + late_sq_coef·dev²]  (dev>0, 封顶 L3_LATE_CAP=20)  # 次项：尾部超线性
    - early_coef·|dev|                    (dev<0)      # 早到惩罚(始终，防 always-FAST 塌缩)
```

- 参数：`ontime_bonus=3.0, σ=10s, late_coef=0.15, late_sq_coef=0.001, early_coef=0.05`。
- ⚠️ **`global_coef` / `global_delay_sec` 是死参数**（env 从不传入 → 该项恒为 0），
  **不属于有效奖励**。L3 的团队项由独立的 `compute_l3_team_reward`（滑动窗口，
  `2.0·ontime_rate − 0.05·mean_dep`）× `l3_team_lambda` 结算。代码已加注释，
  提交代码前应删除该死参数。
- 优先级缩放：整个局部奖励乘 `PRIORITY_WEIGHT={1:1.4, 2:1.0, 3:0.6}`。
- **FAST 能量代价**：`L3_FAST_PENALTY=0.30`/区间（`reward.py`），防止"always-FAST"塌缩；
  行为分布匹配 SA（CRUISE≈63% / FAST≈37%，限速段 FAST≈76%）。
- **团队项**（episode 级，`compute_l3_team_reward`）：
  `2.0·episode_ontime_rate - 0.05·episode_mean_dep`，经 env 运行统计共享给所有 worker，
  给学习者跨车协调的梯度。

### 7.3 训练时 per-episode 指标

两个训练脚本都记录 `ontime_rate`（**发车**准点）与 `arr_ontime`（**到站**准点，realized L3
到达）。论文收敛对比用 **held-out eval**（每 200 集冻结策略、固定 12 集 incident 种子），
比训练时 per-episode 到站准点干净（flat 25t 训练滚动 0.50 是噪声，held-out 稳定 0.41）。

---

## 8. MAPPO 结构

`mappo.py` + `actor_critic.py` + `l3_worker.py`：

- **共享参数 + 中心化训练、去中心化执行**：所有列车共享一个 `FactoredActor`；训练时用
  局部 + 全局双评论家（中心化），执行时每个智能体只用自身观测采样动作。
- **双评论家（local/global）**：L2 的 `LocalCritic`（局部冲突图邻居拼接，O(1) 扩展）+
  `GlobalCritic`（14 维全局状态）。GAE 对每个评论家**独立**计算。
- **组合优势**：策略损失用 `A_combined = A_local + λ_team·A_global`（优势归一化防 NaN）；
  价值损失 `MSE(V_local, ret_local) + λ_team·MSE(V_global, ret_global)`。
- **PPO**：clip_ratio=0.2，ppo_epochs=5，minibatch=256，entropy_coef 0.01→0.005 退火，
  value_coef=0.5，max_grad_norm=1.0，GAE(γ=0.99, λ=0.95)。
- **动作掩码在策略内**：采样前把非法动作 logits 置 `-inf`（平台/进站双头各自掩码）。
- **L3 镜像结构**：`L3WorkerTrainer` 同样双 critic（critic_local/critic_global），
  组合优势 `adv_l + adv_g`，两评论家各自 MSE。

L2 网络尺寸（15t 配置）：obs 31 → hidden 128 → 平台头 4 / 进站头 6；
LocalCritic 输入 62；GlobalCritic 14 → 1。L3：obs 10 → hidden 128 → 3。

---

## 9. 训练流程

`run_train_hierarchical.py`，两阶段（Phase A+B 足够，Phase C 不需要）：

| Phase | 训练内容 | L2 | L3 | 轮数 | 时长(15t) |
|---|---|---|---|---|---|
| **A** | L2 调度 + SA-L3 提供 L3 经验 | 学习 | 用 SA 基线 | 3000 | ~11 min |
| **B** | 固定 L2，学习 L3 走廊速度 | 冻结 | 从随机学习(双critic) | 3000 | ~20 min |

- **Phase A**：L3 用 SA（`l3_sa=True`）执行，L2 在真实 L3 结果上学习调度——所以分层管线
  "第 0 集即可用"（到站准点 ~0.66 是 SA 的水平）。
- **Phase B**：冻结 Phase A 的 L2，`L3WorkerTrainer` 从随机初始化学习逐区间速度控制，
  到站准点从 ~0.26 爬回 ~0.67-0.71。
- **flat MAPPO 基线**（`run_flat_mappo.py`）：72 动作联合（平台×进站×单档速度），
  单网络，无分层。
- **训练输出**：`output/<scenario>_hierarchical/<timestamp>/` 下
  `training_metrics.csv`（含 `arr_ontime` 列）+ `checkpoints/{best,final}.pt`；
  held-out eval 写 `eval_ontime.csv`。

---

## 10. 关键实证结果（论文引用速查）

- **扩展性 ontime%（10/15/20/25/30t，incident 3x60，15t 训练免重训迁移）**：
  learned 77.4/71.4/64.0/62.2/62.0；sa(oracle) 79.7/73.0/66.7/68.1/66.0；
  sa-online 78.0/71.4/66.3/67.5/65.7；先来先服务+调度员规则 53.7/44.2/41.2/38.7/34.5；
  **flat 48.3/42.9/38.0/38.3/36.5（≤ 调度员规则）**。死锁全 0。
- **收敛（等预算 6000 集，held-out 口径，episodes-to-50% 到站准点）**：
  flat 15t/25t **永不**（最终 0.453/0.412）；hier 15t/25t 首个 eval 点(ep 200)即可用
  （SA 预训练起步），学到的 L3（Phase B 从随机）在累计 4000/5000 集内收敛到 50%+。
- **算力（2026-09-20 用实测替换原估算）**：MAPPO **0.452 ms/区间**（**与规模无关**，log-log
  斜率 −0.01）vs SA 一次重解 **7.33 ms/趟**（随规模上升，斜率 +0.14，$10\!\to\!30$ 车
  $6.61\!\to\!7.74$ ms）。每区间比值 **3.61 → 4.28**（10→30t）。
  数据：`output/9.20/compute_by_scale.csv`、`output/9.20/compute_time_scaling.csv`；
  复现：`scripts/bench_l3_decision_cost.py`。
  优势为**常数倍**，不是渐进优势。
- **行为**：learned 行为分布匹配 SA（CRUISE/FAST 比例），无 always-FAST/never-FAST 塌缩。

数据位置：`output/8_14/scalability/`（10-20t）、`output/8_16/scalability/`（25/30t +
收敛分析）。详见 `HANDOFF.md` 与 `coding_log`。

---

## 附录：关键文件

| 模块 | 文件 |
|---|---|
| L1 边界 | `src/control/dispatch_bounds.py` |
| 动作/掩码/请求 | `src/control/action_mask.py` |
| 冲突消解 | `src/control/resource_manager.py` |
| L3 走廊物理 | `src/control/l3_sim.py`、`configs/*/l3_physics.yaml` |
| L2 环境/观测/奖励 | `src/env/five_station_env.py`、`observation_builder.py`、`reward.py` |
| L2 网络/MAPPO | `src/marl/actor_critic.py`、`mappo.py`、`rollout_buffer.py` |
| L3 网络/训练器 | `src/marl/l3_worker.py` |
| 训练入口 | `src/experiment/run_train_hierarchical.py`、`run_flat_mappo.py` |
| 评估/收敛 | `scripts/eval_protocol.py`、`train_l3_convergence.py`、`plot_conv_scalability.py` |
