# Handoff — 2026-08-08 22:40（明天：实现走廊限速事故场景）

## 项目状态

三层铁路 MARL 框架，L3 已升级为**逐区段速度曲线学习**（顺序决策、移动闭塞 headway 硬边界）。当前在**扩展性研究**阶段：验证分层 MAPPO 相对贪婪 SA 优化器的优势是否随规模/拥堵增长。

## 今天完成

1. **L3 v3**：逐区段序贯调速（worker 每区段选 CRUISE/FAST/SLOW），moving-block headway 硬边界，区段间 done=False/区段末 done=True。塌缩修复：平滑 ontime reward（高斯）+ late_coef=0.05 + entropy schedule。
2. **团队奖励**：episode 级 `compute_l3_team_reward`（λ=0.3），L3 看到系统结果（episode ontime_rate, mean_dep）。仍是 MAPPO。
3. **安全底线**：修了站台 min_gap bug（零时长终点车 booking 立即过期→平台提前释放，`w.end + min_gap > t_now`）。L1/L2 全审计干净。
4. **训练日志采样修复**：单 episode → 滚窗均值。
5. **收敛验证协议**（标准基建）：
   - `scripts/train_l3_convergence.py`（多 seed + 斜率平台化）、`scripts/plot_l3_conv_protocol.py`、`scripts/analyze_l3_behavior.py`、`scripts/eval_protocol.py`（held-out 30 seed 分层）。

## 3 seed 收敛结果（今天跑完，确认收敛）

`scripts/train_l3_convergence.py` 15t，2500ep×3，团队奖励：
| seed | plateau(ontime/dep) | final ontime | final dep |
|---|---|---|---|
| 0 | ✓/✓ | 75.8% | 18.9s |
| 1 | ✓/✓ | 77.2% | 18.2s |
| 2 | ✓/✓ | 77.0% | 18.3s |
- 全部过斜率平台化判据；最终方差小（ontime 76.7±0.7%，dep 18.5±0.4s）→ 训练稳定。
- 收敛图：`output/figures/l3v3_conv_15t_ms.png`；checkpoint：`output/l3v3_five_station_15t_seed{0,1,2}.pt`。
- 注意：这是**当前 origin-bunching 扰动**下的收敛（learned ≈ SA，协调价值小）。

## 关键结论（为什么重做延误场景）

- 15 车下 **learned ≈ SA**（拥堵子集优势极小）。origin-bunching 扰动（60-160s）不可恢复、淹没一切；worker 行为是 delay 驱动（晚点→FAST），**不是拥堵协调**（下一站拥堵仍 FAST 100%），团队奖励也没改变。
- **L3 速度协调在当前扰动下杠杆小 → 重做延误场景**，让协调空间变大（见 plan）。

## 明天计划（part 2，plan 已写）

`plan_corridor_incident_scenario.md`：**走廊临时限速事故**扰动（暴雨），让协调价值显性。

**依据（已查铁路资料，来源在 coding_log 22:01 条目）**：
- 暴雨（1h 降雨 ≥45mm）→ 限速 **120 km/h = 0.4×**（P2 巡航 300）；大暴雨（≥60mm）→ **45 km/h = 0.15×**。硬数据。
- 跟车间隔：正常 CTCS-2/3 ≈ 180s；限速→湿轨制动距离增→间隔增。`safety_factor 1.5`（估计，论文标注）。
- 场景 = 限速 0.4 + `moving_block_safety_m × 1.5`。

**实现**（按 plan）：
1. `disturbance.py` `inject_corridor_incident`（config: prob/min_time/duration/factor/safety_factor）
2. env：`_corridor_incidents`，transit 时算 speed_factor 传 L3 sim + obs（obs 第10槽 = 1.0-speed_factor）
3. `l3_sim.py` `section_run_time`/`section_progress` 加 speed_factor
4. config `time_generation.corridor_incident`
5. 重训 + 协议验证（收敛/行为/held-out，按受限分层）→ 20/25 车扩展性

## 关键文件 / checkpoint

- `src/control/l3_sim.py`（逐区段 + section_progress + moving-block）
- `src/env/five_station_env.py`（stepwise loop + 团队奖励 + min_gap 修复）
- `src/env/reward.py`（smooth ontime + compute_l3_team_reward）
- `src/marl/l3_worker.py`、`src/control/l3_sa.py`（SA 基线）
- `scripts/{train_l3_convergence,plot_l3_conv_protocol,analyze_l3_behavior,eval_protocol}.py`
- `configs/five_station_15t/`（15 车变体，`scripts/generate_scaled_timetable.py` 生成）
- checkpoint：`output/l3v3_five_station_15t_seed{0,1,2}.pt`（3 seed 收敛版）

## 已知坑 / 注意

- **训练 log 采样**：单 episode 会骗人（只采到无束发→假 2.3s/96%）——用滚窗均值/held-out eval。
- **观测里 `next_platforms_free`（obs idx7）就是下游拥堵信号**，但 worker 目前不用它协调（reward 没激励）。加 obs 前先确认是否已有可用信号。
- **reward 量纲**：团队/全局项太弱（0.002）= 无协调；太强（逐次 0.01+）= 震荡。episode 级团队奖励是干净方案（每集一个值）。
- **站台 min_gap**：终点终止车零时长 booking，过滤要 `w.end + min_gap > t_now`（已修）。
- **eval seed 集要固定**：不同 seed 集 learned vs SA 胜负会反转（650000+ learned 赢，800000+ 打平）。用协议固定 held-out。
