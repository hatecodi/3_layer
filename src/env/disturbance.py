"""Disturbance module — delay injection and deadlock threshold.

Layer 2 injects disturbances at initialisation time.  Trains that are delayed
at their origin station propagate the delay to all downstream stations via the
existing _get_effective_bounds shift mechanism.

Conflicts arise from the combination of timetable overlaps AND random initial
delays — this gives MAPPO a harder coordination problem than the deterministic
timetable alone.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

DEADLOCK_THRESHOLD = 15  # global NO_LEGAL_ACTION consecutive steps


# ---------------------------------------------------------------------------
# initial departure delay
# ---------------------------------------------------------------------------

INITIAL_DELAY_PROB = 0.3         # per-train probability
INITIAL_DELAY_RANGE = (30, 90)   # uniform [min, max] seconds


def inject_initial_delays(
    journeys: Dict,
    trains: Dict,
    rng: random.Random,
    prob: float = INITIAL_DELAY_PROB,
    delay_range: Tuple[float, float] = INITIAL_DELAY_RANGE,
) -> None:
    """Randomly delay a subset of trains at their origin station.

    Modifies ``journey.arrival_at_current`` in-place.  Delayed trains will
    arrive at their first station later, and the delay cascades to all
    subsequent stations via the existing _get_effective_bounds shift.
    """
    for tid, j in journeys.items():
        if j.is_done:
            continue
        if rng.random() < prob:
            delay = rng.uniform(*delay_range)
            j.arrival_at_current += delay
