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


DEFAULT_OBS_DIM = 34
OPTION_K = 8       # top-K schedule options
GLOBAL_STATE_DIM = 14


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


class OptionActor(nn.Module):
    """Single-headed actor: pick from K concrete schedule options.

    Each option is a (platform, entry_time, departure_time) tuple.
    Mask is a single boolean tensor of length K.
    """
    def __init__(
        self,
        obs_dim: int = DEFAULT_OBS_DIM,
        K: int = OPTION_K,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.K = K
        self.action_dim = K
        self.norm = nn.LayerNorm(obs_dim)
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.head = nn.Linear(hidden_dim, K)

    def logits(self, obs: torch.Tensor, mask: torch.Tensor | None = None):
        h = self.norm(obs)
        h = self.shared(h)
        logits = self.head(h)
        if mask is not None:
            mask = mask.to(dtype=torch.bool, device=logits.device)
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits

    def dist(self, obs, mask=None):
        logits = self.logits(obs, mask)
        return Categorical(logits=logits)


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
    option: torch.Tensor          # index into top-K schedule options
    logprob: torch.Tensor
    entropy: torch.Tensor
    value_local: torch.Tensor
    value_global: torch.Tensor


class Layer2ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int = DEFAULT_OBS_DIM,
        K: int = OPTION_K,
        k_local: int = 3,
        global_state_dim: int = GLOBAL_STATE_DIM,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.actor = OptionActor(obs_dim, K, hidden_dim)
        self.local_critic = LocalCritic(obs_dim, k_local, hidden_dim)
        self.global_critic = GlobalCritic(global_state_dim, hidden_dim)

    def act(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        local_group_obs: torch.Tensor,
        global_state: torch.Tensor,
    ) -> ActorCriticOutput:
        dist = self.actor.dist(obs, mask)
        option = dist.sample()
        return ActorCriticOutput(
            option=option,
            logprob=dist.log_prob(option),
            entropy=dist.entropy(),
            value_local=self.local_critic(local_group_obs),
            value_global=self.global_critic(global_state),
        )

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        option: torch.Tensor,
        local_group_obs: torch.Tensor,
        global_state: torch.Tensor,
    ) -> ActorCriticOutput:
        dist = self.actor.dist(obs, mask)
        return ActorCriticOutput(
            option=option,
            logprob=dist.log_prob(option),
            entropy=dist.entropy(),
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
