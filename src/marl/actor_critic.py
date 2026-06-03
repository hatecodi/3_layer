"""Actor and critic networks for Layer 2 MAPPO."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
from torch.distributions import Categorical


OBS_DIM = 74
DEFAULT_ACTION_DIM = 45
GLOBAL_STATE_DIM = 14


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


class MaskedActor(nn.Module):
    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = DEFAULT_ACTION_DIM,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.net = _mlp(obs_dim, hidden_dim, action_dim)

    def logits(self, obs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.net(obs)
        if mask is None:
            return logits
        mask = mask.to(dtype=torch.bool, device=logits.device)
        return logits.masked_fill(~mask, -1e9)

    def dist(self, obs: torch.Tensor, mask: torch.Tensor | None = None) -> Categorical:
        return Categorical(logits=self.logits(obs, mask))


class LocalCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        k_local: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = (k_local + 1) * obs_dim
        self.net = _mlp(self.input_dim, hidden_dim, 1)

    def forward(self, local_group_obs: torch.Tensor) -> torch.Tensor:
        return self.net(local_group_obs).squeeze(-1)


class GlobalCritic(nn.Module):
    def __init__(
        self,
        global_state_dim: int = GLOBAL_STATE_DIM,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.net = _mlp(global_state_dim, hidden_dim, 1)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state).squeeze(-1)


@dataclass
class ActorCriticOutput:
    action: torch.Tensor
    logprob: torch.Tensor
    entropy: torch.Tensor
    value_local: torch.Tensor
    value_global: torch.Tensor


class Layer2ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = DEFAULT_ACTION_DIM,
        k_local: int = 3,
        global_state_dim: int = GLOBAL_STATE_DIM,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.actor = MaskedActor(obs_dim, action_dim, hidden_dim)
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
        action = dist.sample()
        return ActorCriticOutput(
            action=action,
            logprob=dist.log_prob(action),
            entropy=dist.entropy(),
            value_local=self.local_critic(local_group_obs),
            value_global=self.global_critic(global_state),
        )

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        actions: torch.Tensor,
        local_group_obs: torch.Tensor,
        global_state: torch.Tensor,
    ) -> ActorCriticOutput:
        dist = self.actor.dist(obs, mask)
        return ActorCriticOutput(
            action=actions,
            logprob=dist.log_prob(actions),
            entropy=dist.entropy(),
            value_local=self.local_critic(local_group_obs),
            value_global=self.global_critic(global_state),
        )


def compact_global_state(metrics: dict) -> List[float]:
    """Build the 14-dim global critic input from metrics."""
    return [
        float(metrics.get("n_active", 0)),
        float(metrics.get("mean_delay", 0.0)),
        float(metrics.get("max_delay", 0.0)),
        float(metrics.get("unresolved_conflicts", 0)),
        float(metrics.get("n_granted", 0)),
        float(metrics.get("n_held", 0)),
        float(metrics.get("total_delay", 0.0)),
        float(metrics.get("deadlock_flag", 0)),
        float(metrics.get("completion_rate", 0.0)),
        float(metrics.get("hard_bound_infeasible_count", 0)),
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def pad_mask(mask: Sequence[bool], action_dim: int = DEFAULT_ACTION_DIM) -> List[bool]:
    values = list(mask[:action_dim])
    values.extend([False] * (action_dim - len(values)))
    return values
