# Plan: Fix five-station env bugs + MAPPO critic mismatch

Date: 2026-06-04
Status: done

## MAPPO 策略分析

当前实现：**参数共享 IPPO + 双 critic (local + global) CTDE**

- Actor: 共享 `MaskedActor`，每个 agent 独立从自己观测采样动作
- LocalCritic: 输入 = (K_local+1) × obs_dim = 460-dim（自身+3邻居观测拼接）
- GlobalCritic: 输入 = 14-dim compact global state（系统延迟/冲突/正点率等）
- 策略损失: A_combined = A_local + λ_team × A_global（PPO clip）
- Critic 损失: MSE(V_local) + λ_team × MSE(V_global)
- Curriculum λ: 0.05 → 0.30

**关键发现：MAPPO update 期间两个 critic 的输入都与 collect 时不一致！**

1. Global critic: update 时输入是 `torch.zeros()` (mappo.py:180-182)，collect 时是真实的 14-dim 全局状态 → **GlobalCritic 学到的永远是常数输出**
2. Local critic: update 时输入是 obs 后面补零到 460-dim (`_build_local_critic_input`)，collect 时是真实邻居观测拼接 → **训练/采集分布严重不匹配**
3. `AgentTransition` 不存储 global_state 和 neighbor obs → update 时无法还原真实 critic 输入

这解释了为什么 2000 episode 不收敛：critic 的价值估计不可靠 → GAE advantage 是噪声 → 策略梯度无效。

## Bug 清单和修复方案

### P0-1: reset() info 缺失 active_train_ids + requests

文件: `src/env/five_station_env.py`，reset() 方法

```python
# 当前: info 缺少
# "active_train_ids" → train() 取到 [] → action_dict={} → 第一步用 heuristic
# "requests" → critic input 无冲突图

# 修复: reset 时构建初始 requests 并加入 info
initial_requests = self._build_initial_requests()  # 或直接从 pending 构建
info["active_train_ids"] = self._pending_trains()
info["requests"] = initial_requests  # 空列表或初始请求
```

### P0-2: 删除重复时间跳跃

文件: `src/env/five_station_env.py`，step() 248-255 行

```python
# 当前: 第一次时间跳跃 (248-255) 后，第二次又跳 (293-306)
# 修复: 删除第一个时间跳跃块 (248-255)，只保留 293-306 的版本
# 同时在 293 块中处理 "所有列车在途中" 的情况（第一次跳的功能）
```

### P1-3: 中间站 dep_max 过紧

文件: `src/env/five_station_env.py`，_get_effective_bounds() 502-513 行

```python
# 当前: dep_max = max(actual_arrival + dwell + 120, actual_arrival + 70 + dwell)
# 问题: 如果 t_now > actual_arrival + 50，W_out 必定超过 dep_max
# 修复: dep_max 使用 max(t_now + dwell + 120, actual_arrival + dwell + 120, physical)
# 即给从当前时刻起足够的时间窗
dep_max = max(
    t_now + dwell_min + slack,           # 从当前时刻起
    actual_arrival + dwell_min + slack,  # 从实际到达起
    physical_min_dep,
)
```

需要把 `t_now` 参数传进 `_get_effective_bounds`。

### P1-4: MAPPO update 时存储并使用真实全局状态

文件: `src/marl/rollout_buffer.py` + `src/marl/mappo.py` + `src/experiment/run_train.py`

```python
# AgentTransition 新增字段:
global_state: torch.Tensor  # [14] compact global state at this step
local_group_obs: torch.Tensor  # [critic_input_dim] actual neighbor obs

# collect_step / add_step 时传入真实值
# update 时使用存储的值，而非 zeros
```

这是最重要的修复 — 解决 critic 训练/采集不匹配。

### P2-5: _neighbor_block 优先级硬编码

文件: `src/env/observation_builder.py:170`

```python
# 当前: 3.0 硬编码
# 修复: float(r.priority)
```

### P2-6: 死锁检测移到时间跳跃之后

文件: `src/env/five_station_env.py`，step()

```python
# 当前: 死锁检测在 258 行，使用跳跃前的 granted_ids
# 修复: 把死锁检测移到最终时间设定之后，使用重算的 pending
```

### P3-7: reset 时预计算 metrics 避免第一步 global state 全零

文件: `src/experiment/run_train.py` 或 `src/env/five_station_env.py`

```python
# reset info metrics 当前为 {}
# 修复: 初始 metrics 使用默认值
info["metrics"] = {"n_active": len(all_ids), "n_granted": 0, ...}
```

## Alignment check

- [x] No hardcoded IDs: OK（不改 ID）
- [x] Observation config-driven: OK（不碰 get_obs_dim）
- [x] Action mask enforces bounds: OK（不改 mask 逻辑）
- [x] Local critic input = (K+1)×obs_dim: OK（修复一致性，维度不变）
- [x] Fallback re-verifies bounds: OK
- [x] Config-only expansion: OK

## Pass criteria

1. reset() info 包含 active_train_ids，第一步策略可正常采样
2. 时间跳跃只发生一次，无重复 +10s
3. 中间站列车不会因等待几轮就永久不可调度
4. MAPPO update 时 critic 输入与 collect 时一致（global state + neighbor obs 均存储）
5. P0-1 ~ P1-4 修复后，跑 500 episode 验证 reward 有上升趋势
