# Plan: 扩展性补数据 — 25t/30t 规模点 + 分层 vs flat 收敛速度(训练轮数)

## Context

三层铁路 MARL 论文的扩展性论证当前只有 10t/15t/20t 三个规模点的**评估(transfer)曲线**
(learned 77/71/64 vs flat 48/43/38, 均 incident 3x60)。今天补两块数据:

1. **往大加规模点**: 新增 25t/30t, 把质量扩展性曲线延伸到 5 个点 (10→30)。
2. **训练算力/时间缩放** — 用**训练轮数(episodes-to-converge)**代替墙钟(不依赖硬件):
   展示**分层收敛更快** + **flat 必须重训且越大越差**, 而分层 15t 一次训练可免重训迁移。

关键约束: 训练脚本现在的 `ontime_rate` 是**发车**准点(n_ontime_dep), 而 flat "发车收敛 88% 但到站 43%" —
收敛对比必须用**到站准点**(realized arrival, `n_ontime_arr`/`n_arrivals`, 已在 `reward.compute_metrics` 里, 按决策步计算)。
因此需要给两个训练循环插桩累计到站准点。

用户已确认: 收敛对比的**从头训练只覆盖核心规模 15t + 25t**(两方法), 其余规模用 transfer 曲线。

**执行时更新(用户指示): 所有今日产出保存到 `output/8_16/`**(非 8_15/8_14)。训练脚本加了 `--out-root`
参数(`output/8_16/training/...`); 25t/30t 评估 CSV 已从 8_14 移到 `output/8_16/scalability/`。

## 现状(已验证)

- 生成器: `scripts/generate_scaled_timetable.py --counts N [--slack/--bounds-slack/--min-spacing/--swap-cost]`
  以 base `configs/five_station`(11 车)加 shadow。20t 用 `--slack 60 --bounds-slack 120 --min-spacing 12 --swap-cost 12`(revert 后的最终配置), 25t/30t 沿用同样 flags。
- 训练 CLI:
  - flat: `python -m src.experiment.run_flat_mappo --scenario five_station_15t --incident --episodes N`
  - 分层: `python -m src.experiment.run_train_hierarchical --env five_station_15t --incident --phases A,B`
    (`--env` 直接当 scenario_name; run_dir = `output/{env}_hierarchical/<ts>/`)
- 评估: `python scripts/eval_protocol.py --scenario five_station_25t --l2 <phaseA/best.pt> --l3 <phaseB_v3/best.pt> --flat-ckpt <flat/best.pt> --incident --n 60 --blocks 3 --out-dir ...`
- 已有 ckpt(15t 训练, 用于 transfer): `output/8_14/training/phaseA/best.pt`(L2)、`phaseB_v3/best.pt`(L3)、`flat_mappo/best.pt`。
- 15t 成本基准: phaseA 3000 集 ≈11min、phaseB_v3 3000 集 ≈20min、flat 7000 集 ≈45-60min。
- obs_dim 全规模 = 31(与 scale 无关), 25t/30t 可直接用 15t ckpt。

## 实施步骤

### Part A — 25t/30t 配置 + 扩展评估曲线(质量扩展性 10→30)

1. 生成配置(与 20t 同 flags):
   `python scripts/generate_scaled_timetable.py --counts 25 30 --slack 60 --bounds-slack 120 --min-spacing 12 --swap-cost 12`
2. 校验: `python -m src.experiment.scenario_loader`(需对 25t/30t 都 OK)。
3. 可行性 smoke: `python scripts/eval_protocol.py --scenario five_station_25t --cruise-only --no-delay --n 10`
   (30t 同样) — 检查 0 deadlock / 0 trunc / 0 hard-guard。
4. 扩展评估(transfer, 全部 6 种 mode):
   - `eval_protocol --scenario five_station_25t --l2 phaseA/best.pt --l3 phaseB_v3/best.pt --flat-ckpt flat_mappo/best.pt --incident --n 60 --blocks 3 --out-dir output/8_14/scalability/25t`
   - 30t 同样 → `output/8_14/scalability/30t/`
5. 合并 10/15/20/25/30 的 ontime.csv → `output/8_14/scalability/summary_ontime.csv`
   (列: scale, learned, sa, sa-online, sa-nofc, cruise, flat)。

### Part B — 训练循环插桩到站准点

- `src/experiment/run_flat_mappo.py`:
  - episode 内每步从 `info["metrics"]` 累计 `n_ontime_arr`/`n_arrivals`;
  - 集末算 `arr_ontime = ep_ontime_arr / max(ep_arrivals,1)`, 加入 `episode_metrics` dict(→ CSV);
  - 加滚动窗口 deque(镜像分层训练的 `win_ontime`), 进度打印里加 `arr_ontime`。
- `src/experiment/run_train_hierarchical.py`:
  - 同样在 episode 循环累计到站准点, 加入 `ep_summary`(→ CSV)与滚动窗口。
- 注意: env 每步清 `_corridor_delays`, 每步 `n_ontime_arr` 只计当步 realized arrival — 跨步累计即全集口径。

### Part C — 从头训练收敛运行(15t + 25t, 两方法, 带 per-episode 到站准点)

- flat(预算参数化, 默认 4000 集, 若 4000 时曲线仍明显上升则扩到 7000):
  - `run_flat_mappo --scenario five_station_15t --incident --episodes 4000`
  - `run_flat_mappo --scenario five_station_25t --incident --episodes 4000`
  → `output/flat_mappo/<ts>/`
- 分层(PhaseA+B 各 3000, 累计 6000):
  - `run_train_hierarchical --env five_station_15t --incident --phases A,B`
  - `run_train_hierarchical --env five_station_25t --incident --phases A,B`
  → `output/five_station_{15,25}t_hierarchical/<ts>/`
- 长期任务用后台运行 + 轮询。

### Part D — 分析 + 图(CSV-first)

新增 `scripts/plot_conv_scalability.py`:
- 读各训练 `training_metrics.csv`, 算滚动窗口到站准点 vs 累计 episode。
- 表 `output/8_14/scalability/conv_summary.csv`: (scale, method) × {episodes-to-50% 到站准点(未达到记 inf/never), 预算末滚动到站准点}。
- 图1 `conv_curves.png`: 双 panel(15t, 25t) — 分层 vs flat 到站准点 vs episodes。
- 图2 `scalability_quality.png`: x=N(10/15/20/25/30), y=transfer 到站准点%, 各 mode 折线 + 15t/25t 从头收敛点标注。
- 产出 CSV 全放 `output/8_14/scalability/`。

### Part E — 文档

- 追加 `coding_log`: 今日所有训练 save 路径(`output/flat_mappo/<ts>/`、`output/five_station_{15,25}t_hierarchical/<ts>/`)、
  新评估 CSV(`output/8_14/scalability/{25,30}t/`)、合并表、图、收敛表。
- 更新 `HANDOFF.md`: 扩展性表 10/15/20/25/30 + 收敛对比结论。

## 验证

1. 25t/30t `scenario_loader` OK + cruise smoke 无 deadlock/trunc/hard-guard。
2. 扩展评估产出 `output/8_14/scalability/{25,30}t/ontime.csv`, 合并表 5 个规模齐全。
3. 4 个训练 run 的 `training_metrics.csv` 含 `arr_ontime` 列; 收敛曲线 flat 平在 ~35-45%、
   分层升到 ~60-70%; conv_summary.csv 里 flat 15t/25t 未达 50% 或远晚于分层。
4. `python -m pytest tests/ -q` 不回归(训练脚本改动不破坏测试; 若测试依赖 metrics dict 字段, 只增不删)。
