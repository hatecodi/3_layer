"""Rollout buffer for MAPPO — stores per-step data for all agents.

Each step records a flat list of transitions (one per active agent), plus
the global state needed for the GlobalCritic.  This keeps the buffer agnostic
to the number of agents per step.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch


@dataclass
class AgentTransition:
    agent_id: str
    obs: torch.Tensor           # [obs_dim] from config
    action: int
    logprob: float
    mask: torch.Tensor          # [action_dim] boolean
    value_local: float
    value_global: float
    reward_local: float
    reward_global: float
    done: bool
    global_state: torch.Tensor       # [global_state_dim] compact system state
    local_group_obs: torch.Tensor    # [critic_input_dim] neighbour observations

    def to(self, device: torch.device) -> "AgentTransition":
        self.obs = self.obs.to(device)
        self.mask = self.mask.to(device)
        return self


@dataclass
class RolloutBuffer:
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # internal storage
    _transitions: List[List[AgentTransition]] = field(default_factory=list)
    _global_states: List[torch.Tensor] = field(default_factory=list)

    def add_step(
        self,
        transitions: List[AgentTransition],
        global_state: torch.Tensor,
    ) -> None:
        self._transitions.append(transitions)
        self._global_states.append(global_state)

    def clear(self) -> None:
        self._transitions.clear()
        self._global_states.clear()

    @property
    def num_steps(self) -> int:
        return len(self._transitions)

    def all_transitions(self) -> List[AgentTransition]:
        """Flatten all steps into a single list (for PPO epoch batching)."""
        flat: List[AgentTransition] = []
        for step in self._transitions:
            flat.extend(step)
        return flat

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def compute_gae(
        self,
        next_value_local: Dict[str, float],
        next_value_global: float,
    ) -> None:
        """Compute GAE advantages and returns in-place for local and global.

        Stores results as attributes on each AgentTransition:
          .advantage_local, .return_local
          .advantage_global, .return_global
        """
        # Work backwards through steps
        gae_local: Dict[str, float] = defaultdict(float)
        gae_global: float = 0.0

        for step_idx in range(self.num_steps - 1, -1, -1):
            step = self._transitions[step_idx]
            next_step = (
                self._transitions[step_idx + 1]
                if step_idx + 1 < self.num_steps else None
            )

            for t in step:
                # Terminal / next value
                if t.done:
                    next_v_local = 0.0
                    next_v_global = 0.0
                elif next_step is not None:
                    # Find same agent in next step
                    next_t = next((nt for nt in next_step if nt.agent_id == t.agent_id), None)
                    next_v_local = next_t.value_local if next_t else next_value_local.get(t.agent_id, 0.0)
                    next_v_global = next_t.value_global if next_t else next_value_global
                else:
                    next_v_local = next_value_local.get(t.agent_id, 0.0)
                    next_v_global = next_value_global

                # Local GAE
                delta_local = t.reward_local + self.gamma * next_v_local - t.value_local
                gae_local[t.agent_id] = (
                    delta_local + self.gamma * self.gae_lambda * gae_local[t.agent_id]
                )
                t.advantage_local = gae_local[t.agent_id]
                t.return_local = t.advantage_local + t.value_local

                # Global GAE
                delta_global = t.reward_global + self.gamma * next_v_global - t.value_global
                gae_global = delta_global + self.gamma * self.gae_lambda * gae_global
                t.advantage_global = gae_global
                t.return_global = gae_global + t.value_global


def collate_transitions(
    transitions: List[AgentTransition],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Batch a list of AgentTransitions into tensors for a PPO update."""
    obs = torch.stack([t.obs for t in transitions]).to(device)
    actions = torch.tensor([t.action for t in transitions], dtype=torch.long, device=device)
    logprobs = torch.tensor([t.logprob for t in transitions], device=device)
    masks = torch.stack([t.mask for t in transitions]).to(device)
    adv_local = torch.tensor([t.advantage_local for t in transitions], device=device)
    adv_global = torch.tensor([t.advantage_global for t in transitions], device=device)
    ret_local = torch.tensor([t.return_local for t in transitions], device=device)
    ret_global = torch.tensor([t.return_global for t in transitions], device=device)
    global_states = torch.stack([t.global_state.to(device) for t in transitions])
    local_group_obs = torch.stack([t.local_group_obs.to(device) for t in transitions])

    return {
        "obs": obs,
        "actions": actions,
        "old_logprobs": logprobs,
        "masks": masks,
        "advantages_local": adv_local,
        "advantages_global": adv_global,
        "returns_local": ret_local,
        "returns_global": ret_global,
        "global_states": global_states,
        "local_group_obs": local_group_obs,
    }
