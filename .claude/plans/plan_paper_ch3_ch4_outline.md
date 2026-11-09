# Plan: 论文第三、四章详细提纲

Date: 2026-09-18
Status: done

## What
新增 `docs/paper_outline_ch3_ch4.md`：第三章（问题建模）与第四章（本文算法）的**详细提纲**——
到三级标题、每节要点、公式清单、算法框伪代码骨架、素材出处、写作前置决议、数据依赖。

叙事框架（用户指定，2026-09-18）：
- **不写"L2 协调价值为 0"**：L2 定位改为**滚动重调度决策层**（在 L1 包络内选择平台/进入档位并适应扰动）。
- **补写重调度架构**：第三章加"重调度问题刻画"、第四章加"重调度机制"，覆盖代码中真实存在的
  ① 每步滚动重解 ② 动态包络扩张 ③ 冲突救援换台 ④ hold→replan 兜底 ⑤ L3 硬守卫跨层触发 ⑥ 持久化日历一致性。
- **第六章不做消融**（用户 2026-09-18 追加）：全章不含任何组件消融实验。因此：
  - 不得用消融支撑的表述：双 critic / 损失级融合 / λ 课程 / 掩码 vs 惩罚 / L2 learned vs 启发式 ——
    在第四、六章一律改为**设计选择 + 理论依据（引文献）**，不作经验性优越性声明。
  - 方法对比（SA oracle / sa-online / sa-nofc / cruise / **flat MAPPO** / 启发式）**保留**：
    属基线对比，不是消融；其中 flat MAPPO 对比是核心论据。
  - 重调度证据改用**活动统计**（每步 hold/replan、换台重试、硬守卫触发次数）——测量而非消融。

## Why
用户开始写论文，第三/四章素材已齐（`environment_constraints.md` / `paper_env_mdp_reward.md` /
`architecture_three_layer.md`），但需要先对齐三个口径（"三层"措辞、L2 定位、跨层反馈措辞）再落笔，
否则第五、六章要返工。

## 关键事实（本次核验）
- 重调度架构**真实存在**但**实证常被削弱**：`dep_max_slack_sec = 60` 会吸收 incident 延误，
  使 L2 重调度近似"橡皮图章"（`coding_log` 2026-08-13，实测 tightened slack 60→10 仍无 learned-vs-heuristic 差距）。
  → 提纲中把"打开重调度窗口"列为**实验设计前提**（用 `--severity > 1` 把延误推过 slack），不是缺陷掩饰。
- 项目自带的 pivot 备选叙事（`coding_log` 2026-08-13）：三层 = L1 边界 / L2 重调度 / L3 执行反馈触发重调度；
  基线四件套（无重调度 / 规则式重调度 / 学习式三层 / 全知 oracle）——本提纲采纳为第六章备选骨架。
- 移动闭塞**外部性未内部化**（`plan_reward_externality.md` 状态为 draft，代码中无实现）→ 不得在第四章声称已处理。
- L3 网络隐层实测 128（训练用），`l3_worker.py` 默认 64 为过时默认值。
- 优势融合位置：`mappo.py:186-187`，`A = A_loc + λ·A_team`，**仅在策略损失处**融合。

## Alignment check
- [ ] No hardcoded IDs: OK（纯文档）
- [ ] Observation dim: OK（提纲一律用实测 31 维 / L3 10 维）
- [ ] Action mask enforces bounds: OK（作为第三章"约束强制机制"与第四章 4.2.2 的核心）
- [ ] Local critic dim: OK（实测 62 维 = (1+1)×31，不用 444）
- [ ] Fallback re-verifies bounds: OK（写入 4.5 救援/兜底节）
- [ ] Config-only expansion: N/A（文档）

## Pass criteria
- `docs/paper_outline_ch3_ch4.md` 含 3.x / 4.x 三级标题、每节要点、公式清单、≥3 个算法框伪代码骨架。
- 全文不出现"L2 协调价值为 0"或等价否定表述；L2 贡献以"待第六章消融测定"呈现。
- 重调度机制在第三、四章各有对应小节，且标注"实证需 severity>1 打开窗口"。
- 显式列出必须修正的陈旧数字与数据依赖清单。
