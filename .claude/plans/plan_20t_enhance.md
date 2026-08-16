# Plan: 20t 场景增强 — 修重复阴影 + 密集发车放置 + 对齐走廊时间 + 收紧 L2 slack

Date: 2026-08-13
Status: done (executed 2026-08-13 22:49)

## Context

20t 场景是为"L2 协调/重调度"研究的稀缺性环境。审查发现:① 生成器抖动周期与 up 模板周期错位导致完全重复阴影(G1380_s1==G1380_s7、G1616_s3==G1616_s9);② 15→20 准点率只掉 3.5pp,稀缺信号弱;③ corridor_runtime_sec 两处不一致;④ L2 slack 过松。目标:去重、阴影集中于密集发车区、对齐、收紧 slack,让 20t 成为真稀缺环境。

## Changes

- `scripts/generate_scaled_timetable.py` — 密度感知放置:按(始发站,方向)分组,贪心选密度最大且距已有列车 ≥ min_spacing 的槽位;`--min-spacing 12` 保证无完全重复;tie-break 偏好更晚槽位(避免排在所有人之前的伪密集);新增 `--slack/--bounds-slack`。20t 用 `--slack 30 --bounds-slack 60`。
- `src/control/resource_manager.py` — resolve_all():与 PARKED/访客车窗冲突时也尝试 fallback(原仅 `cid in results`,导致首选站台被访客车占用时直接 held、origin 车卡死)。
- `configs/five_station/railway_resources.yaml` + `five_station_15t/railway_resources.yaml` — dead `network.corridor_runtime_sec` 对齐 timetable LIVE 值(B_to_C:90, C_to_D:180, D_to_E:180…)。
- `configs/five_station_20t/` — 重生成:9 阴影无重复,A 上行聚在 96/108/120/132(紧贴 base 31-79),dep_max_slack 60→30,静态 bounds slack 120→60。

## Validation (heuristic+CRUISE, 10 seeds, no bunch)

- 20t 10/10 success,0 hard-guard,0 infeasible;28 tests pass;schedule_cruise 全在等级限速内。
- ontime 20t 44.0% vs 15t 39.4% — **反超**。收紧 origin bounds 强制准点出发,反而减少延误级联。
- **D 站稀缺未成真**:max concurrent pending at D = 1,移动闭塞走廊 headway 提前拉开到达间距,D 的 2 站台从不争用。绑定约束是走廊 headway(L3 域),非 D 站台(L2 域)。

## Pass criteria

- [x] 无重复阴影、场景可加载、测试通过、速度可达
- [x] 0 deadlock / 0 hard-guard / 0 infeasible
- [ ] 20t ontime 与 15t 差距 ≥10pp —— 未达成(方向反了,44.0 vs 39.4)
- [ ] D 平台稀缺 —— 未达成(走廊 headway 预拉间距)

## Follow-up

- L2 与 L3 需在**新 20t 配置上重训**(obs 分布已变)。
- D 稀缺需换杠杆:加密 D 到达(走廊预拉间距会抵消)或显式再调度触发,而非堆始发车。
