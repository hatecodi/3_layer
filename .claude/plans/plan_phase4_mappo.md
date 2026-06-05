# Plan: Phase 4 MAPPO Integration
Date: 2026-06-03
Status: done

## 来源

- V4 §Phase 4: dual-critic loss-level fusion
- coding_plan Step 4: rollout_buffer.py, mappo.py, actor_critic.py

## 要创建/修改的文件

| 文件 | 状态 | 内容 |
|------|------|------|
| `src/marl/rollout_buffer.py` | 新建 | 存储 (obs, mask, action, logprob, r_local, r_global, V_local, V_global) |
| `src/marl/mappo.py` | 新建 | PPO 更新：独立 GAE → A_local+A_global → 双 critic 损失 |
| `src/experiment/run_train.py` | 新建 | 训练脚本，curriculum λ, checkpoint, tensorboard/metrics |
| `src/marl/actor_critic.py` | 验证 | 已有 LocalCritic(K_local=3→296dim), GlobalCritic(14dim), 确认无误 |

## 关键架构（V4）

```
# GAE — 独立计算，不融合
A_local,  returns_local  = GAE(r_local,  V_local,  gamma, gae_lambda)
A_global, returns_global = GAE(r_global, V_global, gamma, gae_lambda)

# Policy loss
A_combined = A_local + lam_team * A_global
L_policy = -(log_probs * A_combined).mean()

# Critic loss
L_V = MSE(V_local_pred, returns_local) + lam_team * MSE(V_global_pred, returns_global)
```

## 验收标准

| # | 标准 | 来源 |
|---|------|------|
| 1 | action sampling 严格使用 mask，illegal action count = 0 | coding_plan |
| 2 | rollout buffer 记录 legal action logprob | coding_plan |
| 3 | 训练 1 episode 不报错 | coding_plan |
| 4 | GAE 对 local/global 独立计算 | V4 |
| 5 | 优势不融合，只在 loss 中融合 | V4 |
| 6 | Curriculum λ: 0→0.05→0.30 | V4 |
| 7 | 训练 N episodes 后 mean reward ≥ heuristic baseline (3.26) | 新增 |
| 8 | 训练 N episodes 后 mean dep_delay ≤ heuristic baseline (37.2s) | 新增 |
| 9 | 无 NaN/inf 梯度 | 新增 |
