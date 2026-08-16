# Plan: Resolver swap-cost — 让跨站台 fallback 不再免费

Date: 2026-08-13
Status: done (executed 2026-08-13)

## Outcome

Implemented + validated. swap_cost_sec=12 in five_station_20t.
- resolver: cross-platform fallback starts search at swap_cost_sec (shift=0 only for same-platform delay); default 0.0 = legacy.
- env passes cfg.time_generation.swap_cost_sec to resolve_all; generator has --swap-cost.
- A/B (swap_cost 0 vs 12, 10 seeds): ontime 45.1->42.5, fallback dep_delay p50 187->221s, 10/10 success, 0 guards.
- Sweep 12/30/60 (8 seeds): all feasible; larger cost converts rescues->holds (n_fallback 21->12) but ontime plateaus ~41%.
- Caveat: dense 20t congestion dominates ontime; swap cost is a second-order signal (learned-vs-heuristic headroom ~2-3pp), not a dramatic lever.

## Context

C 站共享咽喉(上行 g1=p1+p2、g2=p3+p4)的争夺被 resolver 免费救场:冲突后换到其它站台 shift=0、W_out 不变、零代价(resource_manager.py:198-209)。L2 的(站台,进站时机)选错咽喉/站台没有任何后果 → 无学习梯度 → L2 学不到咽喉协调(与"咽喉放 L3 没处理"同类:信号无执行器,这里是决策无后果)。且 reward 里 `retried` 信号已计算但未使用(reward.py:171)。

目标:让跨站台换线付出真实延迟(物理税),而非人工奖励扣分。冲突仍被救场(不 deadlock),但代价真实 → 进 dep_delay 惩罚与实际到站时间 → L2 学到"选兼容的站台/咽喉组"。

## What

1. `src/control/resource_manager.py`
   - `assign_fallback(..., swap_cost_sec: float = 0.0)`: `_first_feasible(platform, is_original)` 的初始 `shift = 0.0 if is_original else swap_cost_sec`。换到其它站台(非原站台)从 swap_cost_sec 起步;同站台延后保持 0 起步。默认 0.0 → 全部旧调用点行为不变。
   - `resolve_all(..., swap_cost_sec: float = 0.0)`: 透传给 `assign_fallback`。
2. `src/env/five_station_env.py` `_resolve_requests`(787 行附近): 读 `swap_cost_sec = self.cfg.time_generation.get("swap_cost_sec", 0.0)`,传给 `resolve_all`。
3. `scripts/generate_scaled_timetable.py`: 新增 `--swap-cost` 参数,注入 `variants["time_generation"]["swap_cost_sec"]`,保证重生成不丢。
4. 重生成 20t:`--counts 20 --min-spacing 12 --slack 30 --bounds-slack 60 --swap-cost 12`(布局不变,仅加 swap_cost)。

## Alignment check

- [x] 无硬编码 ID:swap_cost 是通用 resolver 参数,20t 配置注入
- [x] 观测维度:不变(31 维)
- [x] Action mask 强制 bounds:不变
- [x] Fallback 仍 re-verify bounds:rebuild_fn 检查 arr_min/dep_max 不变
- [x] Config-only:新配置项走 timetable.yaml

## Verification

1. `python -m pytest tests/test_phase12.py tests/test_l3_physics.py -q` — 28 测试(默认 0.0 不回归)
2. 20t 加载 OK
3. 启发式+CRUISE 10 seeds:无 deadlock 爆炸、无截断激增、0 hard-guard;确认 swap_cost 生效(跨站台 fallback 产生真实延迟,对比 swap_cost=0 时 retry 无延迟)
4. C 咽喉:测量 swapped C 车次的平均额外延迟(>0 即成本成立)
5. 记录 coding_log

## Files

- `src/control/resource_manager.py`
- `src/env/five_station_env.py`
- `scripts/generate_scaled_timetable.py`
- `configs/five_station_20t/timetable.yaml`(重生成)
- `coding_log`

## Pass criteria

- 28 测试通过;20t 可加载
- 启发式下无 deadlock 爆炸 / 截断不增加 / 0 hard-guard
- swap_cost=12 时,跨站台 fallback 的车次实际到站晚 ≥ swap_cost(成本真实成立)
- C 共享咽喉:被换站台的车次有可测的额外延迟
