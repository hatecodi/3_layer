# Plan — 新增 FCFS 规则基线（保留 cruise）

## 目标

在基线集合中加入 **rule-based FCFS**（`Implementation Specification v4.md:491` 列的基线 (1)），
作为**真正的非学习基线**；`cruise` 模式保留在默认集合里（用户决定），代码不删。

## 关键发现：env 已内置 FCFS 回退，不需要改 env

`src/env/five_station_env.py:837-841`：

```python
action = action_dict.get(train_id)
if action is None:
    return heuristic_action(train, station_id, self._t_now,
                            TRAVEL_TIME_TO_THROAT, self.cfg, bounds_override=bounds)
```

**传空 `action_dict` 即触发 `heuristic_action`**（`action_mask.py:395`：按计划时刻取第一个
合法平台/入场档；`heuristic_planner.schedule_station:80-81` 按 `t_dep_plan` 排序 = 先到先服务）。

因此 FCFS = **空 action_dict + 规则 CRUISE L3**，零 env 改动。

## 与现有 "cruise" 模式的区别（这是本基线的意义）

| mode | L2 | L3 |
|---|---|---|
| `cruise`（现有） | **学习式** MAPPO | 规则 CRUISE |
| `fcfs`（新增） | **规则** `heuristic_action`（FCFS） | 规则 CRUISE |

现有 `cruise` 的 L2 仍在用学习策略，"规则基线"名不副实；`fcfs` 补齐真正的非学习对照。
两者 L3 相同 → 差异**干净归因于 L2 是否学习**。这是论文可用的对比。

## 改动

`scripts/eval_protocol.py`：
1. `run_episode` 增加 `elif mode == "fcfs": ad = {}` 分支（附注释说明回退机制）。
2. 新增 `--fcfs-only`；默认 mode 列表加入 `"fcfs"`（保留 `"cruise"`）。
3. 报表 E1 算力块：`fcfs` 自然落入 `else: method_ms = 0.0`（规则法无前向）——无需改，
   但确认打印正确。

`src/env/five_station_env.py`、`src/control/action_mask.py`：**不改**。

## 诚实边界（必须写进论文/表注）

resolver 的冲突仲裁仍是 **priority → delay → train_id**（`resource_manager.resolve_all`），
不是 FCFS。所以 `fcfs` 是「**FCFS 动作规则 + 共享 priority 仲裁器**」，不是教科书意义上的
纯 FCFS（高优先级车在争用站台时仍可越过先到者）。用户已确认保持 resolver 不变（所有基线
共用同一仲裁器，对比口径一致）。若审稿人追问，这一点须主动说明，不能默认读者按纯 FCFS 理解。

## 验证

- `python scripts/eval_protocol.py --scenario five_station_15t --fcfs-only --n 3` 能跑通。
- `fcfs` 与 `cruise` 的 ontime/延误**应当不同**（L2 不同）；若完全相同说明分支没生效。
- 与 `heuristic_planner.build_heuristic_plan` 的结果量级一致（同一动作规则的两种入口）。
