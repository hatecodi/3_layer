"""Disturbance injection and Layer 2 activation logic."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Optional, Set

from ..control.dispatch_bounds import TrainBounds


TRIGGER_THRESHOLD_SEC = 30.0


class ControlStatus(Enum):
    NOMINAL = "nominal"
    LAYER2_ACTIVE = "layer2_active"
    LAYER2_MONITORING = "layer2_monitoring"


@dataclass
class TrainRuntimeState:
    train_id: str
    delay_sec: float = 0.0
    pending_layer2_target: bool = False
    held_steps: int = 0


@dataclass
class DisturbanceInjector:
    prob_per_step: float = 0.03
    delay_range: tuple[float, float] = (30.0, 180.0)
    rng: random.Random = field(default_factory=random.Random)

    def step(
        self,
        train_ids: Iterable[str],
        states: Dict[str, TrainRuntimeState],
    ) -> Dict[str, float]:
        injected: Dict[str, float] = {}
        for train_id in train_ids:
            if self.rng.random() >= self.prob_per_step:
                continue
            delta = self.rng.uniform(*self.delay_range)
            states.setdefault(
                train_id, TrainRuntimeState(train_id=train_id)
            ).delay_sec += delta
            injected[train_id] = delta
        return injected


def layer2_control_status(
    state: TrainRuntimeState,
    bounds: TrainBounds,
    conflict_neighbors: Optional[Set[str]] = None,
    trigger_threshold_sec: float = TRIGGER_THRESHOLD_SEC,
) -> ControlStatus:
    if abs(state.delay_sec) > trigger_threshold_sec:
        return ControlStatus.LAYER2_ACTIVE
    if conflict_neighbors:
        return ControlStatus.LAYER2_ACTIVE
    if state.pending_layer2_target:
        return ControlStatus.LAYER2_MONITORING
    return ControlStatus.NOMINAL
