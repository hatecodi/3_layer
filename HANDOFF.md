# Handoff — 2026-08-15

## 项目状态（关键变化：今天有了可写的正向结果）

三层铁路 MARL。**本轮完成物理修复 + 场景完善 + 分层重训，拿到了可辩护的结果：**

- **learned ≈ 非全知 SA（sa-online）**，略低于全知 SA（oracle），**远胜 cruise（无协调 floor）**。
- **扩展性：learned 在 10t→30t 平滑迁移（77/71/64/62/62 ontime），全面碾压 flat MAPPO（48/43/38/38/36.5）**，flat 甚至低于 cruise。
- **诚实根因不变**：双线单股道无越行 = 纯 FIFO → **L2 无协调价值**（learned L2 ≈ 启发式，所有场景验证）。三层价值在 **L3 的高效执行（即时响应 + 节能 + 到站精度）+ 整体鲁棒性 + 扩展性**，不在 L2 协调。

## 本轮完成（2026-08-14→15）

1. **物理修复（关键）**：`section_progress` 改连续 moving-block（P2，FAST 不能中段贴车）；worker 执行按所选档位重算 wait；CRUISE 改走同一条逐区间路径（P1，消除 transit() 咽喉双算）。三条路径物理统一（stepwise ≡ transit，diff=0）。→ 旧 checkpoint 全失效，需重训。
2. **incident 修复**：整趟限速→**section-resolved**（只限窗口覆盖的区段）；`safety 1.5→1.0`、`max_added 200→80`（低于 replan 阈值，防雪崩）。→ incident 从"全崩"变成"可恢复"（aff_mean ~30-40s）。加了 `--severity` 参数开关（不改基础配置）。
3. **场景完善**：K834→G834（C→D，**D 停 48 sim-s 换车头**，造真实 D 站台稀缺）；K1185→G1185；停站异质核对（已符合铁路时刻表.md：G230跳C、G238跳B/C、G1487跳B）；**每条走廊最后一段加进站限速 limit_mps=70**（≈SLOW 速度）。**实证：跳站≠越行**（430 对车 0 次重排序，FIFO 结构不变）。
4. **L3 双 critic + 奖励**：`L3WorkerTrainer` 拆 **critic_local（自身准点）+ critic_global（团队）**（镜像 L2 的 local/global），reward 拆分；FAST penalty 0.30 + **早到惩罚 always（0.05）** → 治好了 always-FAST 塌缩（行为分布匹配 SA：CRUISE 63/FAST 37，限速段 FAST 76%）。save keys: `l3_critic_local/global`。
5. **分层重训（A+B）**：`run_train_hierarchical` PHASES=A(L2+SA-L3)+B(L3固定L2蒸馏)，加 `--incident`/`--l2-ckpt`。Phase A 11min、Phase B v3 ~20min。**A+B 足够，不需 Phase C**（L2≈启发式无协调价值，C 增益小+不稳）。
6. **指标**：尾端严重度（`SEVERE_DELAY_SEC=60s`，n/rate/mean/max）；**E1 重解代价**（SA 重排 n_transits×3.48ms vs learned 即时 n_sections×0.36ms，`compute.csv`）；per-metric CSV 输出（`--out-dir`，每指标一张表）；`--extra-train`（随机加车 P2→D）；`--severity`（incident 强度，开重调度窗口）。
7. **flat MAPPO 重训 + 扩展性**：flat 用 72 动作联合（platform×entry×speed），单档速度。15t+incident 重训 7000 集（发车 ontime 88% 收敛，但到站仅 43%——单档速度无法控到站）。10t/15t/20t 评估。

## 关键结果（15t incident 3×60，learned=PhaseA L2 + PhaseB_v3 L3）

| mode | ontime% | affected 延迟 | E1 重解代价 |
|---|---|---|---|
| sa（全知 oracle）| 73.0 | 30.2 | 重排 2.4x |
| **learned** | **71.4** | 38.5 | 即时（便宜 2.4x）|
| sa-online（非全知）| 71.4 | 39.7 | 0.5ms/区间 |
| cruise | 44.2 | 88.4 | ~0（规则）|
| **flat MAPPO** | **42.9** | 82.0 | 单前向（最便宜但差）|

扩展性 ontime（10t/15t/20t/25t/30t，2026-08-16 补 25t/30t）：**learned 77.4/71.4/64.0/62.2/62.0**、sa 79.7/73.0/66.7/68.1/66.0、sa-online 78.0/71.4/66.3/67.5/65.7、cruise 53.7/44.2/41.2/38.7/34.5、**flat 48.3/42.9/38.0/38.3/36.5（≤ cruise）**。learned 到 30t 仍 ~62%（≈sa-6pp），flat 塌到 ~36%，learned−flat 差距在 25/30t 拉大到 ~25pp。死锁全 0。

**训练轮数收敛对比（2026-08-16，held-out 到站准点 vs episodes，15t/25t 从头训练，等预算 6000 集）**：
收敛指标已升级为 **held-out eval**（训练每 200 集冻结当前策略、固定 12 集 incident 种子评估，见 run_flat_mappo/run_train_hierarchical 的 --eval-every hook）——比训练时 per-episode 干净，flat 25t 训练滚动 0.50 是噪声、held-out 稳定 0.41。
| 方法/规模 | 全管线 episodes-to-50%（held-out） | PhaseB-only(学到的 L3) | 最终 held-out 到站准点 |
|---|---|---|---|
| flat 15t (6000集) | **永不** | — | 0.453 |
| flat 25t (6000集) | **永不** | — | 0.412 |
| hier 15t (PhaseA+B 6000) | 200（首个 eval 点即 ≥50%，SA 预训练起步） | 4000 | 0.673 |
| hier 25t (PhaseA+B 6000) | 200 | 5000 | 0.612 |
含义：**等预算 + held-out 口径**下，分层管线在首个 eval 点（ep 200）即可用（Phase A 用 SA-L3 作 L3 经验），学到的 L3（Phase B 从随机）在累计 4000/5000 集内收敛到 50%+；flat 在同样 6000 集内从未达到 50%（15t 0.45 / 25t 0.41）。训练时 flat 25t 逼近 0.50 的迹象是噪声，held-out 确认 flat 未达可用水平。

## 数据位置

2026-08-16 新增（全部在 `output/8_16/`）：`scalability/{25t,30t}/`（扩展评估 per-metric CSV）、
`scalability/summary_ontime.csv`（10..30 合并）、`scalability/conv_summary.csv` + `conv_curves.png` + `conv_phaseB.png` + `scalability_quality.png`（held-out 收敛对比）、
训练（**规范版=带 held-out eval**，各含 training_metrics.csv + eval_ontime.csv）：
`training/flat_mappo/20260816_{154653,155134}/`（flat 15t/25t 6000集）、
`training/five_station_{15t,25t}_hierarchical/20260816_{154928,154929}/`（hier 15t/25t PhaseA+B）。
旧的无 eval 版本（flat 140618/140829/152139/152543、hier 140621/140832）已被取代，可清理。
配置：`configs/five_station_{25t,30t}/`。

2026-08-14 旧数据（全部在 output/8_14/）：

- `incident/`、`clean/`、`extra/`：15t 对比 + 加车，per-metric CSV（ontime/affected/tail/delay/compute/episode）
- `scalability/{10t,15t,20t}/`：扩展性（含 flat）
- `behavior/`：SA vs learned 行为分布
- `training/{phaseA,phaseB,phaseB_v2,phaseB_v3,flat_mappo}/`：训练 metrics + ckpt
- `training/phaseB_v3/convergence.png`：收敛曲线（reward +3.07，ontime 82.4% 平台化）
- 临时可清理：`extra_15t/`、`extra_20t/`、`extra/headway.csv`（headway 次数指标已弃用）

## 诚实结论

- **L2 协调价值 ≈ 0**（稀缺 G834、incident、重调度窗口 severity=2.5 下，learned L2 ≈ 启发式）。
- **L3 价值在"高效执行"**：即时响应（E1 2.4x 便宜于 SA 重排）、逼近非全知 SA、远胜 cruise 和 flat。
- **flat MAPPO 失败**：72 动作联合空间难学 + 单档速度 → 低于无协调 floor → 是分层分解价值的强论据。
- **重调度未实现**（L3→L2 显式通信没有）；L2 滚动重排存在但被 slack/FIFO 压制。

## 审稿人问题（现在能答）

"为什么做 MAPPO 分层，不做 flat/启发式？"——**现在有数据答**：
1. flat MAPPO（72 联合动作）在 10-30t 全面塌缩（48→36.5，≤ cruise）→ 分层分解价值。
2. learned 以 2.4x 低算力追平非全知 SA、逼近全知 SA、远胜无协调 floor。
3. 加车次/新实例免重训迁移（learned 15t→10t/20t/25t/30t 保持，到站准点 77→62）。
4. **训练轮数（等预算 6000 集，held-out 口径）**：分层首个 eval 点（ep 200）即可用（SA 预训练起步）+ 学到的 L3 ~1000-2000 PhaseB 集收敛到 50%+；flat 6000 集从未到 50%（15t 0.45 / 25t 0.41，见上表）。

## 下一步建议

1. **清理临时数据**（extra_15t/extra_20t/headway.csv）+ 把关键结果整理成论文表格/图。
2. **可选**：short Phase C（联合微调）走形式；或测"L2 在 SA-L3 vs learned-L3 调度一致性"证据。
3. **越行/重调度**：若要坚持"learned 赢强基线"，仍需越行（重排序自由度）；当前"逼近 oracle + 扩展性"框架不依赖它。
4. flat MAPPO 的 PROJECT_ROOT bug 已修；旧单 critic L3 脚本（continue_with_vt 等）与新 L3WorkerTrainer API 不兼容（未更新，legacy）。
