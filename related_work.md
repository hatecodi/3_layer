# Related Work — 列车重调度与运行控制：从离线求解到实时反应式策略

> 用途：论文 Related Work / Introduction 素材。按支撑论点分组，含文献链接。
> 核心论点：**三层 MAPPO（L1 界 → L2 调度 → L3 速度）在突发扰动下靠"逐区间滚动决策"实现瞬时调整，而 SA/优化类方法必须重新求解（重排），且其质量依赖对未来（预测）的正确性。**

---

## 1. 列车实时重调度（TTR）：从精确求解到元启发式再到强化学习

列车时刻表重调度（Train Timetable Rescheduling, TTR）本质是一个组合优化问题。传统求解路径经历了三个阶段的演变：

- **精确求解（IP / MIP）**：解质量高，但计算开销大，难以满足实时性。多篇 2024–2025 文献一致指出，MIP 求解需要小时级时间，而铁路实际运营要求分钟级乃至秒级响应。
- **元启发式（SA / Tabu / EA）**：以 Simulated Annealing（SA）为代表的元启发式被广泛用于重调度的搜索环节（例如在两阶段重调度中，用 SA/Tabu 搜索"列车在区段上的运行顺序"）。但搜索空间随问题规模指数增长，"计算时间 vs 解质量"的权衡是公认瓶颈。
- **强化学习（RL / DRL / MADRL）**：离线训练、在线一次性前向推理，天然面向实时响应，近年成为 TTR 的主流新方向。

**支撑文献：**

- IEEE 综述 *Problem Models and Solution Approaches to Rescheduling in Railway Networks*（调查了 SA/Tabu 在重调度中的角色，指出搜索空间指数增长、计算时间与解质量的权衡是关键瓶颈）：
  https://ieeexplore.ieee.org/document/7160720
- Tang et al., *A multi-task deep reinforcement learning approach to real-time railway train rescheduling*, Transportation Research Part E, 2025（MDRL：离线训练、在线 <1 分钟出近优解，而 MIP 需 >1 小时；平均延误比 FIFO 与单任务 DRL 低 11.1–31.3%，对随机扰动鲁棒）：
  https://www.sciencedirect.com/science/article/abs/pii/S1366554524004915
- IEEE 2025, *Unified Scheduling Model for High-Speed Train Timetable Optimization and Rescheduling Based on Deep Reinforcement Learning*（策略网络顺序构造出发顺序，无需重训即可在不同问题规模上胜过启发式与商业求解器，尤其擅长计算时间受限情形）：
  https://xplorestaging.ieee.org/document/10923657

## 2. 关键卖点一：泛化免重训 —— "遇到突发状况瞬时调整"的直接文献支撑

DRL 之所以能满足 TTR 实时性，核心在于**离线训练得到的策略面对未见过的扰动实例无需重训**——这正是"瞬时调整"的论文表述。传统 Q-learning 因 Q 表需要逐场景重训而无法做到；深度策略网络把"在线重训练/重求解"的成本转移到了离线训练阶段。

**支撑文献：**

- Huo et al., *An adaptive multi-agent deep reinforcement learning approach to unify disturbances and disruptions for high-speed railway rescheduling*, Applied Mathematical Modelling, 2025（**MAPPO**：re-timing / re-ordering / 加停 / 取消统一进复合动作空间；"generalization training"使其面对未知实例免重训，满足 TTR 严格实时性）——**方法论与本工作最接近**：
  https://www.sciencedirect.com/science/article/abs/pii/S0307904X25007449
- IEEE 2022, *A Reinforcement Learning Framework for Train Rescheduling* 与 IEEE 2023, *Reinforcement Learning for Combinatorial Optimization of Train Timetable Rescheduling*（DQN/深度 RL 处理多种延迟场景无需重训；在线决策 1–2 秒，胜过 FCFS）：
  https://ieeexplore.ieee.org/document/9911095
  https://ieeexplore.ieee.org/document/10449236
- LECFO, *Timetabling for trains with multiple speed grades in high-speed rail systems: A learning-enhanced conflict-free hierarchical optimization framework*（把无冲突路径搜索嵌进 RL 循环，策略对列车密度/间隔/车型比例/停站模式等跨场景泛化、无需重训）：
  https://www.sciencedirect.com/science/article/abs/pii/S0360835226003840
- *A reinforcement learning approach to solving very-short term train rescheduling problem for a single-track rail corridor*（单线走廊：一个实例上学到的知识可复用到同走廊更难实例，显著减少学习/计算时间）：
  https://eprints.whiterose.ac.uk/id/eprint/217651/

## 3. 关键卖点二：决策延迟 —— "瞬时"的字面数据

多篇工作给出了在线决策的具体计算时间，可作为"学到的策略在线推理代价极低"的量化证据：

| 文献 | 方法 | 在线决策时间 |
|---|---|---|
| IEEE 2023（RL for Combinatorial Optimization of TTR） | DQN | 1–2 秒 |
| Tang et al. 2025（MDRL） | 多任务 DRL | <1 分钟（MIP 需 >1 小时） |
| IOP 2025（地铁冗余时间再分配） | DQN | 平均 0.184 秒 |
| IEEE TNNLS 2025（轨迹+重调度集成） | DRL | 计算时间随问题规模**线性**增长 |

**支撑文献：**

- https://ieeexplore.ieee.org/document/10449236
- https://www.sciencedirect.com/science/article/abs/pii/S1366554524004915
- DQN 冗余时间再分配（Nanning Metro Line 5，平均 0.184s）：
  https://beta.iopscience.iop.org/article/10.1088/2631-8695/ae25bd/pdf
- Dong et al., *Deep Reinforcement Learning for Integration of Train Trajectory Optimization and Timetable Rescheduling Under Disturbances*, IEEE TNNLS, 2025（京沪高铁，计算时间线性扩展）：
  https://pubmed.ncbi.nlm.nih.gov/38373133/

## 4. 关键卖点三：RL 策略 vs 在线重优化（MPC/SA）的范式对比

这是本工作论点（**学到的策略替代在线重求解，把计算从在线转移到离线**）最直接的一般性论证来源：

- **车辆队列控制综述（MDPI 2025）**明确对比两类范式：在线重优化（MPC）每步在线求解、在线计算代价高；学到的策略（RL/NN）**训练代价高但训练后在线推理代价极低**，可用便宜的前向传播替代每步重优化；对扰动的鲁棒性来自训练时覆盖扰动（generalization 依赖数据）：
  https://www.mdpi.com/2673-4052/7/1/35
- 高速列车虚拟联挂控制（DMPC 实时跟踪 + 离线 DP 规划）体现了"离线规划 + 实时控制"的混合架构，其 Stage I 轨迹离线求解、Stage II 用分布式 MPC 在线跟踪——与"离线学习 + 在线反应"的层级思想互补：
  https://www.sciencedirect.com/science/article/abs/pii/S136655452600027X
- 货运列车多目标速度轨迹 RL（DQN-GRU，无模型，无需精确动力学）显示 RL 可用于生成速度曲线，但其结果表明该算法更适合**离线**生成最优轨迹，进一步说明"在线反应式策略"是 RL 在实时控制中的独特价值：
  https://www.sciencedirect.com/science/article/abs/pii/S0967066123001740

## 5. 分层 / 层级架构：决策-执行分离

本工作三层架构（L1 硬界 → L2 调度 → L3 速度执行）与文献中的分层控制思想一致：**高层定决策，低层做鲁棒执行**，并普遍带来训练效率与收敛性提升。

**支撑文献：**

- Xu et al., *Hierarchical speed control for autonomous electric vehicle through deep reinforcement learning and robust control*, IET Control Theory & Applications, 2022（**决策层 DMEPPO 定车速 + 控制层 LMI 鲁棒控制器跟踪**，分层架构显式对抗不确定性/扰动）：
  https://zbmath.org/?q=ai%3Axu.guangfei+ai%3Ahe.xiangkun
- *Deep reinforcement learning for operation strategies optimization in high-speed trains*（HODRL，铁道科学与工程学报 2025：上层先验知识分层优化缩小状态空间/重塑奖励，下层 TD3 连续控制；无效状态空间平均减 79.7%，训练时间仅标准 TD3 的 15.3%）：
  https://opaj.napstic.cn/periodicalArticle/0120250300564604
- *Hierarchical deep reinforcement learning for dynamic logistics dispatching*（Nature Scientific Reports：分层策略分解 + 时序抽象 + 课程学习，在动态扰动环境下比 flat RL 更优，服务 +18.4%）：
  https://www.nature.com/articles/s41598-025-18309-y

## 6. 多智能体重调度在扰动/中断下的协调（与本工作 L2/L3 协作最贴近）

- Ying, Chow et al., *Adaptive rescheduling of rail transit services with short-turnings under disruptions via a multi-agent deep reinforcement learning approach*, Transportation Research Part B, 2024（伦敦地铁双线走廊全阻断，多智能体 PPO，计算速度优于分布式控制算法与元启发式，可扩展到时长不确定的中断）：
  https://trid.trb.org/View/2425517
- *Timetable Rescheduling for High-Speed Trains Based on a Multi-Agent Game Reinforcement Learning Method*, IEEE 2025（Nash-Q，处理区段全阻断下的 TTR）：
  https://ieeexplore.ieee.org/document/10965300
- 地铁改进 Q-learning + SA 动态因子（Shenzhen Metro：利用 SA 改善动作选择策略，快速实时调整，平均总延误比无调整/人工/FIFO 降低 54%）：
  https://go.gale.com/ps/retrieve.do?tabID=T002&resultListType=RESULT_LIST&searchResultsType=SingleTab&retrievalId=d230952f-1afa-4b0d-b206-5a31e9c313dc&hitCount=1

## 7. 与已有文献的差异与本工作贡献

现有文献覆盖了三点：**(a)** DRL 面对未见扰动免重训（Huo 2025 等）；**(b)** 学到的策略在线推理延迟远低于在线重求解（综述与 DQN/MDRL 数字）；**(c)** 分层架构的决策-执行分离。但**没有**一篇直接量化：*在"预测不可靠/突发事件未提前预报"时，反应式策略（逐区间滚动决策）与批求解器（进场一次性求解、结果依赖预测）之间的鲁棒性差距*。

本工作为此设计可控实验，**实测结果（2026-08-14，15t，C_to_D 走廊）**：

1. **E1 决策延迟**：同一走廊上，SA 300 次迭代重排 3.48 ms/趟 vs MAPPO 逐区间前向 1.44 ms/趟（2.4x）；**单次适应事件** SA 全量重解 3.48 ms vs MAPPO 单区间 0.36 ms（**9.7x**，actor-only 18.8x）。结论：学到的策略"瞬时调整"的**计算代价**优势为 ~10x/次适应，且该优势对批量/编译实现仍保守。
2. **E2 未预报 incident**：把 incident 是否"预报"给 SA 作为开关。实测 **SA 无预测只损失 +37s affected 延迟**（424 vs 387），且仍优于 learned（461）与 sa-online（436）。**反应优势未显现**——根因是 incident 是吞吐瓶颈（列车在限制处排队，个体速度选择被 headway 队列中和），且 factor 对整趟均匀施加、无走廊中段变化可供反应。
3. **E3 突发程度扫描**（未实施）：以"未预报比例"为横轴画曲线，需先做分段 incident（限制只压中段 + 中段 onset）让 L3 速度选择真正有杠杆。

诚实结论：**"瞬时调整"的代价优势（E1）已被数据支持，但"反应鲁棒性优势"（E2）在本 sim 的 L3 速度层未能显现**，瓶颈中和了速度选择的价值。要在实验中体现 E2，需重构为分段/中段 onset incident。
