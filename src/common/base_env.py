"""Base class for railway scheduling environments.

TwoStationEnv and FiveStationEnv implement this interface so training
and evaluation code can work with either without import-time coupling.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class BaseRailwayEnv(ABC):
    """Gym-style interface shared by all railway scheduling environments.

    Subclasses must implement: reset, step, pending_trains, last_requests.
    """

    @abstractmethod
    def reset(self) -> Tuple[Dict[str, List[float]], Dict]:
        """Reset episode.  Returns (obs, info)."""

    @abstractmethod
    def step(
        self, action_dict: Dict[str, int] | None = None
    ) -> Tuple[
        Dict[str, List[float]],    # observations
        Dict[str, float],          # rewards (train_id -> scalar)
        bool,                      # terminated
        bool,                      # truncated
        Dict,                      # info
    ]:
        """Advance one decision step."""

    @property
    @abstractmethod
    def pending_trains(self) -> List[str]:
        """Train IDs that need scheduling at the current decision step."""

    @property
    @abstractmethod
    def last_requests(self) -> List:
        """WindowRequests from the most recent decision step (for critic)."""
