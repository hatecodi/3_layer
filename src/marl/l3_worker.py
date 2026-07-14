"""Layer 3 Worker — shared corridor speed controller.

Architecture:
  - CorridorActor: shared across all trains, 3-way categorical (cruise/fast/slow)
  - CorridorCritic: per-train value estimation for corridor state
  - L3WorkerTrainer: lightweight PPO update loop

The Worker consumes an ExecutionTarget from the L2 Manager and selects a
corridor speed mode.  It is trained separately from the Manager (alternating
phases), with its own reward that reflects corridor-level objectives:
headway delay, arrival punctuality, and recovery margin consumption.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

L3_OBS_DIM = 10
L3_ACTION_DIM = 3  # cruise / fast / slow


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


class CorridorActor(nn.Module):
    """Shared 3-way categorical actor for corridor speed control."""

    def __init__(
        self,
        obs_dim: int = L3_OBS_DIM,
        action_dim: int = L3_ACTION_DIM,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def dist(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.logits(obs))


class CorridorCritic(nn.Module):
    """Per-train value estimator for corridor state."""

    def __init__(
        self,
        obs_dim: int = L3_OBS_DIM,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------------


@dataclass
class L3Transition:
    obs: torch.Tensor
    action: int
    logprob: float
    reward: float
    value: float
    done: bool


class L3RolloutBuffer:
    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self._transitions: List[L3Transition] = []

    def add(self, t: L3Transition) -> None:
        self._transitions.append(t)

    def clear(self) -> None:
        self._transitions.clear()

    def __len__(self) -> int:
        return len(self._transitions)

    def compute_gae(self, final_value: float) -> tuple:
        """Return (advantages, returns) as tensors."""
        T = len(self._transitions)
        advantages = torch.zeros(T)
        returns = torch.zeros(T)
        gae = 0.0
        next_value = final_value
        for t in reversed(range(T)):
            tr = self._transitions[t]
            delta = tr.reward + self.gamma * next_value * (1 - int(tr.done)) - tr.value
            gae = delta + self.gamma * self.gae_lambda * (1 - int(tr.done)) * gae
            advantages[t] = gae
            returns[t] = gae + tr.value
            next_value = tr.value
        return advantages, returns


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class L3WorkerTrainer:
    def __init__(
        self,
        actor: CorridorActor,
        critic: CorridorCritic,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        ppo_epochs: int = 5,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.actor = actor
        self.critic = critic
        self.device = device or torch.device("cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.optimizer = Adam(
            list(actor.parameters()) + list(critic.parameters()),
            lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self._buffer = L3RolloutBuffer(gamma, gae_lambda)
        self._last_stats: Dict[str, float] = {}

    @property
    def buffer(self) -> L3RolloutBuffer:
        return self._buffer

    @torch.no_grad()
    def select_action(self, obs: torch.Tensor) -> tuple:
        """Return (action_index, logprob, value)."""
        obs = obs.unsqueeze(0).to(self.device)
        dist = self.actor.dist(obs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        value = self.critic(obs)
        return action.item(), logprob.item(), value.item()

    def update(self) -> Dict[str, float]:
        """Run PPO epochs over the buffer. Returns loss stats."""
        buffer = self._buffer
        if len(buffer) == 0:
            return {}
        advantages, returns = buffer.compute_gae(0.0)

        obs = torch.stack([t.obs for t in buffer._transitions]).to(self.device)
        old_logprobs = torch.tensor([t.logprob for t in buffer._transitions],
                                    device=self.device)
        old_actions = torch.tensor([t.action for t in buffer._transitions],
                                   device=self.device)

        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        # Normalize advantages
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / adv_std

        stats = {}

        for _ in range(self.ppo_epochs):
            dist = self.actor.dist(obs)
            new_logprobs = dist.log_prob(old_actions)
            entropy = dist.entropy().mean()
            values = self.critic(obs)

            # Policy loss
            ratio = torch.exp(new_logprobs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio,
                                1.0 + self.clip_ratio) * advantages
            loss_policy = -torch.min(surr1, surr2).mean()

            # Value loss
            loss_value = F.mse_loss(values, returns)

            loss = loss_policy + self.value_coef * loss_value - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.max_grad_norm)
            self.optimizer.step()

            stats = {
                "l3_loss_policy": loss_policy.item(),
                "l3_loss_value": loss_value.item(),
                "l3_entropy": entropy.item(),
            }

        buffer.clear()
        self._last_stats = stats
        return stats

    def get_last_stats(self) -> Dict[str, float]:
        return self._last_stats
