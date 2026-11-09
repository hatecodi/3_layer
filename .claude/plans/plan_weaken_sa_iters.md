# Plan: 可配置的 SA 迭代预算（本地弱化基线）

## 目标
让 SA 基线更弱：把 `sa_optimize_profile` 的 `n_iter`（默认 300）从调用侧暴露，
供本地实验用更低预算跑 SA。论文数字仍用默认 300（用户确认：只用于本地实验）。

## 改动
1. `src/env/five_station_env.py`
   - `FiveStationEnv.__init__` 新增 `l3_sa_iters: int = 300`，存 `self._l3_sa_iters`。
   - SA 分支调用 `profile_fn(...)` 时显式传 `n_iter=self._l3_sa_iters`（`sa_online_profile`
     接受并忽略该参数，无需分支）。
2. `scripts/eval_protocol.py`
   - `run_episode` 新增 `sa_iters=300` 参数，传给 env。
   - 新增 CLI `--sa-iters`（默认 300），透传到 `run_episode` 调用处。
3. 不改 `l3_sa.py` / `joint_sa.py` 默认值；其它脚本不传即保持 300，行为不变。

## 验证
- `python -m pytest tests/test_l3_physics.py -q`
- 小规模对比：同一批种子下 `--sa-only --sa-iters 300` vs `--sa-iters 30`，
  确认 SA 指标变差、运行更快。
- 在 coding_log 追加记录（含产物路径）。
