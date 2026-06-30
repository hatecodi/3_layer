"""MAPPO trainer with dual-critic loss-level fusion (V4 Phase 4).

Architecture:
  - Shared MaskedActor for all agents
  - LocalCritic per agent (group-based, 296-dim input)
  - GlobalCritic shared (14-dim global state input)
  - GAE computed independently for each critic
  - A_combined = A_local + lam_team * A_global  (fused in policy loss only)
  - L_V = MSE(V_local, returns_local) + lam_team * MSE(V_global, returns_global)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.optim import Adam

from .actor_critic import Layer2ActorCritic, compact_global_state
from .rollout_buffer import AgentTransition, RolloutBuffer, collate_transitions


def curriculum_lambda(
    episode: int,
    total_episodes: int,
    lam_min: float = 0.05,
    lam_max: float = 0.30,
) -> float:
    """V4 curriculum: lam_team rises 0.05 → 0.30 over training."""
    progress = episode / max(total_episodes, 1)
    if progress < 0.3:
        return lam_min
    elif progress < 0.6:
        return lam_min + (lam_max - lam_min) * (progress - 0.3) / 0.3
    else:
        return lam_max


class MAPPOTrainer:
    def __init__(
        self,
        actor_critic: Layer2ActorCritic,
        lr: float = 3e-4,
        eps: float = 1e-5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        ppo_epochs: int = 5,
        minibatch_size: int = 256,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        total_episodes: int = 1000,
        device: Optional[torch.device] = None,
    ):
        self.ac = actor_critic
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.ac.to(self.device)

        self.optimizer = Adam(self.ac.parameters(), lr=lr, eps=eps)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.total_episodes = total_episodes
        self._episode_count = 0
        self._lam_team = 0.05

    @property
    def lam_team(self) -> float:
        return self._lam_team

    def advance_episode(self) -> None:
        self._episode_count += 1
        self._lam_team = curriculum_lambda(
            self._episode_count, self.total_episodes)

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_step(
        self,
        obs: Dict[str, List[float]],
        masks: Dict[str, List[bool]],
        local_group_obs: Dict[str, List[float]],
        global_state_vec: List[float],
        device: Optional[torch.device] = None,
    ) -> Dict[str, "AgentAction"]:
        """Sample actions for all active agents. Returns agent_id → action info."""
        device = device or self.device
        actions: Dict[str, "AgentAction"] = {}

        global_t = torch.tensor(global_state_vec, dtype=torch.float32,
                                device=device).unsqueeze(0)

        for agent_id in sorted(obs.keys()):
            if agent_id not in masks:
                continue
            mask_tuple = masks[agent_id]
            p_mask, e_mask = mask_tuple
            if not any(p_mask) and not any(e_mask):
                # No legal action — hold, handled by env
                continue

            obs_t = torch.tensor(obs[agent_id], dtype=torch.float32,
                                 device=device).unsqueeze(0)
            # Pad platform mask to 4
            p_mask_t = torch.tensor(list(p_mask) + [False] * (4 - len(p_mask)), dtype=torch.bool, device=device)
            e_mask_t = torch.tensor(list(e_mask), dtype=torch.bool, device=device)
            local_t = torch.tensor(local_group_obs.get(agent_id, obs[agent_id]),
                                   dtype=torch.float32,
                                   device=device).unsqueeze(0)

            with torch.no_grad():
                out = self.ac.act(obs_t, p_mask_t, e_mask_t, local_t, global_t)

            actions[agent_id] = AgentAction(
                platform=out.platform.item(),
                entry=out.entry.item(),
                logprob=out.logprob.item(),
                value_local=out.value_local.item(),
                value_global=out.value_global.item(),
                entropy=out.entropy.item(),
            )

        return actions

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Run PPO epochs over the buffer. Returns loss statistics."""
        transitions = buffer.all_transitions()
        if not transitions:
            return {}

        batch = collate_transitions(transitions, self.device)
        n_samples = batch["obs"].shape[0]

        # Use stored critic inputs from collection (not padded zeros)
        local_group_obs = batch["local_group_obs"]
        global_states = batch["global_states"]

        stats = {"loss_policy": 0.0, "loss_value_local": 0.0,
                 "loss_value_global": 0.0, "loss_total": 0.0,
                 "entropy": 0.0, "approx_kl": 0.0}

        indices = list(range(n_samples))

        for _ in range(self.ppo_epochs):
            # Shuffle for minibatches
            import random
            random.shuffle(indices)

            for start in range(0, n_samples, self.minibatch_size):
                mb_idx = indices[start:start + self.minibatch_size]

                mb_obs = batch["obs"][mb_idx]

                mb_old_logprobs = batch["old_logprobs"][mb_idx]
                mb_pmask = batch["platform_masks"][mb_idx]
                mb_emask = batch["entry_masks"][mb_idx]
                mb_platform = batch["platform"][mb_idx]
                mb_entry = batch["entry"][mb_idx]
                mb_adv_local = batch["advantages_local"][mb_idx]
                mb_adv_global = batch["advantages_global"][mb_idx]
                mb_ret_local = batch["returns_local"][mb_idx]
                mb_ret_global = batch["returns_global"][mb_idx]
                mb_local_obs = local_group_obs[mb_idx]
                mb_global = global_states[mb_idx]

                # Evaluate actions under current policy
                out = self.ac.evaluate_actions(
                    mb_obs, mb_pmask, mb_emask, mb_platform, mb_entry,
                    mb_local_obs, mb_global)

                # ---- Policy loss (V4: A_combined = A_local + lam_team * A_global) ----
                advantages = mb_adv_local + self._lam_team * mb_adv_global
                # Normalize advantages (guard against zero-variance, which produces NaN)
                adv_std = advantages.std()
                if adv_std > 1e-8:
                    advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

                ratio = torch.exp(out.logprob - mb_old_logprobs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio,
                                    1.0 + self.clip_ratio) * advantages
                loss_policy = -torch.min(surr1, surr2).mean()

                # ---- Critic loss (V4: independent heads, fused in loss) ----
                loss_v_local = F.mse_loss(out.value_local, mb_ret_local)
                loss_v_global = F.mse_loss(out.value_global, mb_ret_global)
                loss_value = loss_v_local + self._lam_team * loss_v_global

                # ---- Entropy bonus ----
                entropy = out.entropy.mean()

                # ---- Total loss ----
                loss = (
                    loss_policy
                    + self.value_coef * loss_value
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track stats
                stats["loss_policy"] += loss_policy.item()
                stats["loss_value_local"] += loss_v_local.item()
                stats["loss_value_global"] += loss_v_global.item()
                stats["loss_total"] += loss.item()
                stats["entropy"] += entropy.item()
                with torch.no_grad():
                    stats["approx_kl"] += (
                        (mb_old_logprobs - out.logprob).mean().item())

        # Average over all updates
        n_updates = self.ppo_epochs * max(
            1, math.ceil(n_samples / self.minibatch_size))
        for k in stats:
            stats[k] /= max(n_updates, 1)

        return stats


@dataclass
class AgentAction:
    platform: int
    entry: int
    logprob: float
    value_local: float
    value_global: float
    entropy: float


def _build_local_critic_input(
    transitions: List[AgentTransition],
    obs_dim: int,
    critic_input_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Pad each agent's obs to critic_input_dim for LocalCritic.

    In production, this should use observation_builder.critic_input().
    For simplicity in PPO updates, we pad with zeros to reach the target dim.
    """
    n = len(transitions)
    padded = torch.zeros(n, critic_input_dim, device=device)
    for i, t in enumerate(transitions):
        vec = t.obs.to(device)
        padded[i, :len(vec)] = vec  # pad with zeros beyond obs_dim
    return padded
