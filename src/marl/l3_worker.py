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
    """One per-section transition with DECOMPOSED rewards/values.

    reward_local / value_local  -> the train's OWN corridor outcome (punctuality,
                                   FAST energy).
    reward_global / value_global -> the episode-level TEAM term (system ontime /
                                   delay), credited at the corridor end.
    Two critics (local + global) separate these so credit assignment doesn't mix
    own-punctuality gradients with the shared team term (mirrors L2's local/global).
    """
    obs: torch.Tensor
    action: int
    logprob: float
    reward_local: float
    reward_global: float
    value_local: float
    value_global: float
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

    def compute_gae(self, final_local: float, final_global: float) -> tuple:
        """Return ((adv_local, ret_local), (adv_global, ret_global)) tensors —
        one GAE per reward component (local critic, global critic)."""
        T = len(self._transitions)
        adv_l = torch.zeros(T); ret_l = torch.zeros(T)
        adv_g = torch.zeros(T); ret_g = torch.zeros(T)
        gae_l = gae_g = 0.0
        nv_l, nv_g = final_local, final_global
        for t in reversed(range(T)):
            tr = self._transitions[t]
            not_done = 1 - int(tr.done)
            d_l = tr.reward_local + self.gamma * nv_l * not_done - tr.value_local
            d_g = tr.reward_global + self.gamma * nv_g * not_done - tr.value_global
            gae_l = d_l + self.gamma * self.gae_lambda * not_done * gae_l
            gae_g = d_g + self.gamma * self.gae_lambda * not_done * gae_g
            adv_l[t] = gae_l; ret_l[t] = gae_l + tr.value_local
            adv_g[t] = gae_g; ret_g[t] = gae_g + tr.value_global
            nv_l, nv_g = tr.value_local, tr.value_global
        return (adv_l, ret_l), (adv_g, ret_g)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class L3WorkerTrainer:
    def __init__(
        self,
        actor: CorridorActor,
        critic_local: CorridorCritic,
        critic_global: CorridorCritic,
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
        self.critic_local = critic_local
        self.critic_global = critic_global
        self.device = device or torch.device("cpu")
        self.actor.to(self.device)
        self.critic_local.to(self.device)
        self.critic_global.to(self.device)
        self.optimizer = Adam(
            list(actor.parameters()) + list(critic_local.parameters())
            + list(critic_global.parameters()),
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
        """Return (action_index, logprob, value_local, value_global)."""
        obs = obs.unsqueeze(0).to(self.device)
        dist = self.actor.dist(obs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        v_local = self.critic_local(obs)
        v_global = self.critic_global(obs)
        return action.item(), logprob.item(), v_local.item(), v_global.item()

    def update(self) -> Dict[str, float]:
        """Run PPO epochs over the buffer with TWO critics (local + global).
        The policy is driven by the combined advantage (local + global); each
        critic regresses its own return.  Returns loss stats."""
        buffer = self._buffer
        if len(buffer) == 0:
            return {}
        (adv_l, ret_l), (adv_g, ret_g) = buffer.compute_gae(0.0, 0.0)

        obs = torch.stack([t.obs for t in buffer._transitions]).to(self.device)
        old_logprobs = torch.tensor([t.logprob for t in buffer._transitions],
                                    device=self.device)
        old_actions = torch.tensor([t.action for t in buffer._transitions],
                                   device=self.device)

        adv_l = adv_l.to(self.device); adv_g = adv_g.to(self.device)
        ret_l = ret_l.to(self.device); ret_g = ret_g.to(self.device)

        # combined advantage for the policy (own punctuality + team term)
        advantages = adv_l + adv_g
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / adv_std

        stats = {}

        for _ in range(self.ppo_epochs):
            dist = self.actor.dist(obs)
            new_logprobs = dist.log_prob(old_actions)
            entropy = dist.entropy().mean()
            vals_l = self.critic_local(obs)
            vals_g = self.critic_global(obs)

            # Policy loss
            ratio = torch.exp(new_logprobs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio,
                                1.0 + self.clip_ratio) * advantages
            loss_policy = -torch.min(surr1, surr2).mean()

            # Value loss: two critics, each on its own return
            loss_value = F.mse_loss(vals_l, ret_l) + F.mse_loss(vals_g, ret_g)

            loss = loss_policy + self.value_coef * loss_value - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic_local.parameters())
                + list(self.critic_global.parameters()),
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
