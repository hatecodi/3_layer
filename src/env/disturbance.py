"""Disturbance injection and Layer 2 activation logic."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Optional, Set

from ..control.dispatch_bounds import TrainBounds


TRIGGER_THRESHOLD_SEC = 30.0
DEADLOCK_THRESHOLD = 25           # V4: consecutive no-legal-action steps before deadlock


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


# ---------------------------------------------------------------------------
# Corridor speed restriction — simulates track failure / maintenance
# ---------------------------------------------------------------------------

@dataclass
class CorridorDisturbance:
    """Speed restriction on a corridor segment during a time window.

    During [start_time, end_time], trains traversing *corridor_id* experience
    reduced speed, increasing their transit time and causing downstream delays.
    """
    corridor_id: str            # e.g. "A_B_up", "B_A_down"
    start_time: float           # restriction window start (sim seconds)
    end_time: float             # restriction window end
    speed_ratio: float = 0.3    # fraction of normal speed (0.3 = 30% → 3.3× transit)

    def extra_transit_time(self, corridor_runtime_sec: float) -> float:
        """Additional seconds added to corridor transit."""
        return corridor_runtime_sec * (1.0 / max(self.speed_ratio, 0.01) - 1.0)


def compute_corridor_delay(
    train_id: str,
    corridor_id: str,
    t_corridor_entry: float,
    corridor_runtime_sec: float,
    disturbances: list[CorridorDisturbance],
) -> float:
    """Return extra delay (seconds) caused by corridor restrictions.

    If the train's corridor transit window [entry, entry+runtime] overlaps
    any restriction on this corridor, return the extra transit time.
    """
    t_exit = t_corridor_entry + corridor_runtime_sec
    for d in disturbances:
        if d.corridor_id != corridor_id:
            continue
        if t_corridor_entry < d.end_time and t_exit > d.start_time:
            return d.extra_transit_time(corridor_runtime_sec)
    return 0.0


# ---------------------------------------------------------------------------
# Layer 2 activation
# ---------------------------------------------------------------------------

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
