"""Actor and critic networks for Layer 2 MAPPO with factored action space.

FactoredActor: separate platform head (4 logits) and entry head (6 logits).
Total logits: 10 (was 72). Dwell always 0 (min dwell, optimal).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn
from torch.distributions import Categorical


DEFAULT_OBS_DIM = 26
PLATFORM_DIM = 4  # max platforms per station
ENTRY_DIM = 6      # 6 entry choices
GLOBAL_STATE_DIM = 14


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


class FactoredActor(nn.Module):
    """Two-headed actor: platform choice + entry timing.
    
    4 platform logits + 6 entry logits = 10 total (was 72).
    Dwell is always min_dwell (0 extra).
    """
    def __init__(
        self,
        obs_dim: int = DEFAULT_OBS_DIM,
        platform_dim: int = PLATFORM_DIM,
        entry_dim: int = ENTRY_DIM,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.platform_dim = platform_dim
        self.entry_dim = entry_dim
        self.action_dim = platform_dim + entry_dim  # 10 total
        self.norm = nn.LayerNorm(obs_dim)
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.platform_head = nn.Linear(hidden_dim, platform_dim)
        self.entry_head = nn.Linear(hidden_dim, entry_dim)

    def logits(
        self,
        obs: torch.Tensor,
        platform_mask: torch.Tensor | None = None,
        entry_mask: torch.Tensor | None = None,
    ):
        h = self.norm(obs)
        h = self.shared(h)
        p_logits = self.platform_head(h)
        e_logits = self.entry_head(h)
        if platform_mask is not None:
            platform_mask = platform_mask.to(dtype=torch.bool, device=p_logits.device)
            p_logits = p_logits.masked_fill(~platform_mask, float("-inf"))
        if entry_mask is not None:
            entry_mask = entry_mask.to(dtype=torch.bool, device=e_logits.device)
            e_logits = e_logits.masked_fill(~entry_mask, float("-inf"))
        return p_logits, e_logits

    def dist(self, obs, platform_mask=None, entry_mask=None):
        p_logits, e_logits = self.logits(obs, platform_mask, entry_mask)
        return Categorical(logits=p_logits), Categorical(logits=e_logits)


class LocalCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int = DEFAULT_OBS_DIM,
        k_local: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = (k_local + 1) * obs_dim
        self.norm = nn.LayerNorm(self.input_dim)
        self.net = _mlp(self.input_dim, hidden_dim, 1)

    def forward(self, local_group_obs: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(local_group_obs)).squeeze(-1)


class GlobalCritic(nn.Module):
    def __init__(
        self,
        global_state_dim: int = GLOBAL_STATE_DIM,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(global_state_dim)
        self.net = _mlp(global_state_dim, hidden_dim, 1)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(global_state)).squeeze(-1)


@dataclass
class ActorCriticOutput:
    platform: torch.Tensor
    entry: torch.Tensor
    logprob: torch.Tensor
    entropy: torch.Tensor
    value_local: torch.Tensor
    value_global: torch.Tensor


class Layer2ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int = DEFAULT_OBS_DIM,
        platform_dim: int = PLATFORM_DIM,
        entry_dim: int = ENTRY_DIM,
        k_local: int = 3,
        global_state_dim: int = GLOBAL_STATE_DIM,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.actor = FactoredActor(obs_dim, platform_dim, entry_dim, hidden_dim)
        self.local_critic = LocalCritic(obs_dim, k_local, hidden_dim)
        self.global_critic = GlobalCritic(global_state_dim, hidden_dim)

    def act(
        self,
        obs: torch.Tensor,
        platform_mask: torch.Tensor,
        entry_mask: torch.Tensor,
        local_group_obs: torch.Tensor,
        global_state: torch.Tensor,
    ) -> ActorCriticOutput:
        p_dist, e_dist = self.actor.dist(obs, platform_mask, entry_mask)
        platform = p_dist.sample()
        entry = e_dist.sample()
        logprob = p_dist.log_prob(platform) + e_dist.log_prob(entry)
        entropy = p_dist.entropy() + e_dist.entropy()
        return ActorCriticOutput(
            platform=platform,
            entry=entry,
            logprob=logprob,
            entropy=entropy,
            value_local=self.local_critic(local_group_obs),
            value_global=self.global_critic(global_state),
        )

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        platform_mask: torch.Tensor,
        entry_mask: torch.Tensor,
        platform: torch.Tensor,
        entry: torch.Tensor,
        local_group_obs: torch.Tensor,
        global_state: torch.Tensor,
    ) -> ActorCriticOutput:
        p_dist, e_dist = self.actor.dist(obs, platform_mask, entry_mask)
        logprob = p_dist.log_prob(platform) + e_dist.log_prob(entry)
        entropy = p_dist.entropy() + e_dist.entropy()
        return ActorCriticOutput(
            platform=platform,
            entry=entry,
            logprob=logprob,
            entropy=entropy,
            value_local=self.local_critic(local_group_obs),
            value_global=self.global_critic(global_state),
        )


def compact_global_state(metrics: dict) -> List[float]:
    """Build the 14-dim global critic input from metrics."""
    return [
        float(metrics.get("n_active", 0)),
        float(metrics.get("mean_dep_delay", 0.0)),
        float(metrics.get("mean_arr_delay", 0.0)),
        float(metrics.get("unresolved_conflicts", 0)),
        float(metrics.get("n_granted", 0)),
        float(metrics.get("n_held", 0)),
        float(metrics.get("total_dep_delay", 0.0)),
        float(metrics.get("deadlock_flag", 0)),
        float(metrics.get("completion_rate", 0.0)),
        float(metrics.get("hard_bound_infeasible_count", 0)),
        float(metrics.get("ontime_rate", 0.0)),
        float(metrics.get("n_ontime_dep", 0)),
        0.0,
        0.0,
    ]
