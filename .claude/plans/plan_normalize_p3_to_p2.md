# Plan: 去掉 P3 车(K834/K1185 → G 类 P2),重跑迁移 eval

## 背景 / 问题
- base `configs/five_station/`(11 辆)含 2 辆 K 类 P3 车:K834、K1185。
- `five_station_15t/` 早已把这两辆规范成 **G834/G1185(class=G, priority=2)**,15t 无 P3。
- 20t/25t/30t 由**旧 base(带 K)** 经 `generate_scaled_timetable.py` 生成 → 保留 K834/K1185 并复制出
  K834_s5 / K1185_s10 / K834_s11 / K834_s17 等 P3 影子(20t×3, 25t×5, 30t×6)。
- 影响:迁移 eval(15t ckpt)在 20/25/30t 上 learned 的 P3 ontime=0(P3 低优先+慢速档,模型没训过 P3),
  拖累整体准点率并放大 delay 尾部分布。

## 决定(用户确认)
1. 就地规范化 20t/25t/30t 三个 config 的 K 车 → 与 15t 约定一致。
2. 一起规范化 base `configs/five_station`,避免以后再生成又带出 P3。
3. 不重训;修好 config 后用原 **15t ckpt** 重跑 20/25/30t 迁移 eval。

## 改动
- 每个 timetable.yaml 内对 P3(K 类)车:
  - id `K834`→`G834`, `K1185`→`G1185`(含影子 K834_sX/K1185_sX,用 replace_all)
  - `train_class: K`→`train_class: G`
  - `priority: 3`→`priority: 2`
  - 文件头 `priority_map` 的 `'3': 普速(K834,K1185)` 条目删掉,并把 G834/G1185 并入 `'2': 和谐号(...)`
- 涉及文件: `configs/five_station/timetable.yaml`,
  `configs/five_station_20t/timetable.yaml`, `...25t/...`, `...30t/...`
- 只动 id/class/priority 与头部注释,planned/bounds/时刻一律不动(布局保持)。
- priority_map 仅用于 summary 打印(`scenario_loader.py:513`),无逻辑依赖,可安全编辑。

## 验证
- `grep` 确认四个文件无 `priority: 3` / `K834` / `K1185` 残留,train 总数不变。
- `python -m src.experiment.scenario_loader`(及各规模加载)通过。
- 定位原 scalability eval 命令(15t ckpt + args),对 20t/25t/30t 重跑 `scripts/eval_protocol.py`,
  CSVs 覆盖 `output/8_16/scalability/{20t,25t,30t}/`。预期 learned 准点率回升、无 P3 拖累。

## 产物 / 记录
- config 修改 + eval 输出路径记入 `coding_log`。
