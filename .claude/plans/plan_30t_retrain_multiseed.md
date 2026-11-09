# Plan: 30t 从头收敛重训(hier vs flat, 等预算 6000 集, 3 种子)

## 背景
- 扩展性质量图 scalability_quality.png 里 from-scratch 标记目前只有 15/25t;25t 那两条还是在 P3 config 上训的。
- P3 修正后(configs/five_station_30t 已无 P3),需在修正 config 上补 **30t** 从头收敛点,补成 10..30 全曲线 + 30t from-scratch。
- 用户决定: hier 与 flat **等预算**, 每方法 3 种子。等预算口径对齐 8_16 的 15/25t: hier = PhaseA 3000 + PhaseB 3000 = 6000; flat = 6000。

## 训练协议(对齐 15/25t 规范版)
- 干扰: incident(限速 0.6 prob), 关闭 origin bunching / VT / 初始延误。
- held-out eval hook: --eval-every 200 --eval-n 12 --eval-seed-base 900000(每 200 集冻结策略做 12 集 incident eval, 写 eval_ontime.csv)。
- 场景: configs/five_station_30t(修正后, 无 P3)。obs_dim 全规模一致(31), 无 scale 特定改动。
- 种子: {0, 1, 2}(对齐 l3v3 多种子命名)。
- 输出: --out-root output/9_4/training →
    hier: output/9_4/training/five_station_30t_hierarchical/<ts>/
    flat: output/9_4/training/flat_mappo/<ts>/

## 命令
hier (seed s in 0,1,2):
  python -m src.experiment.run_train_hierarchical --env five_station_30t --incident \
      --phases A,B --seed {s} --out-root output/9_4/training \
      --eval-every 200 --eval-n 12 --eval-seed-base 900000
flat (seed s in 0,1,2):
  python -m src.experiment.run_flat_mappo --scenario five_station_30t --incident \
      --episodes 6000 --seed {s} --out-root output/9_4/training \
      --eval-every 200 --eval-n 12 --eval-seed-base 900000

## 代码小改
- run_flat_mappo.py 加 --seed(默认 42) + torch.manual_seed(seed) —— 否则 flat 无显式种子, "3 种子"不可控(env 扰动固定用 episode index seed, 差异来自策略 init/exploration)。

## 验证 / 后续
- 每 run 产出 training_metrics.csv(含 arr_ontime) + eval_ontime.csv + checkpoints/。
- 6 run 结束后: 平均 3 种子的 held-out arr_ontime(30t hier vs flat), 记录进 coding_log;
  plot_conv_scalability 加 30t from-scratch 标记/训练路径(flat/hier 需显式传 ckpt 路径 —— flat_mappo 目录混多 scale)。
- 记录保存位置到 coding_log(CLAUDE.md 要求)。
