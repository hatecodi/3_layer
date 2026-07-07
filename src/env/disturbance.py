"""Disturbance module — delay injection and deadlock threshold.

Two disturbance modes:
  - per_train: independent random delays per train (legacy)
  - bunched: infrastructure failure — all trains in a time window at one
    origin station are held until the window clears, creating a bunch of
    trains that must navigate the downstream corridor together.

Layer 2 injects disturbances at initialisation time.  Delayed trains
propagate delay to downstream stations via _get_effective_bounds shift.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

DEADLOCK_THRESHOLD = 15  # global NO_LEGAL_ACTION consecutive steps


# ---------------------------------------------------------------------------
# per-train random delay (legacy)
# ---------------------------------------------------------------------------

INITIAL_DELAY_PROB = 0.5
INITIAL_DELAY_RANGE = (30, 180)


def inject_initial_delays(
    journeys: Dict,
    trains: Dict,
    rng: random.Random,
    prob: float = INITIAL_DELAY_PROB,
    delay_range: Tuple[float, float] = INITIAL_DELAY_RANGE,
) -> None:
    """Randomly delay a subset of trains at their origin station."""
    for tid, j in journeys.items():
        if j.is_done:
            continue
        if rng.random() < prob:
            delay = rng.uniform(*delay_range)
            j.arrival_at_current += delay


# ---------------------------------------------------------------------------
# bunched infrastructure delay
# ---------------------------------------------------------------------------

BUNCH_WINDOW_SPAN = 40   # seconds — width of the disruption window
BUNCH_DELAY_MIN = 60     # minimum forced delay after window clears
BUNCH_DELAY_MAX = 120


def inject_bunched_delay(
    journeys: Dict,
    trains: Dict,
    rng: random.Random,
    origin_stations: List[str] | None = None,
    window_span: float = BUNCH_WINDOW_SPAN,
    delay_min: float = BUNCH_DELAY_MIN,
    delay_max: float = BUNCH_DELAY_MAX,
) -> None:
    """Infrastructure failure at a random origin station.

    1. Picks a random origin station from *origin_stations*
    2. Collects all trains departing from that station
    3. Randomly selects a time window [t0, t0 + window_span] covering ≥ 2 trains
    4. All trains in the window are delayed to depart at
       max(planned_dep, window_end) + uniform(delay_min, delay_max)

    This creates genuine corridor bunching — multiple trains released
    close together must navigate headway constraints downstream.
    """
    if origin_stations is None:
        origin_stations = ["A", "E"]

    # Collect trains by origin station with their planned departure
    by_origin: Dict[str, List[Tuple[str, float]]] = {}
    for tid, train in trains.items():
        origin = train.origin_station
        if origin not in origin_stations:
            continue
        j = journeys.get(tid)
        if j is None or j.is_done:
            continue
        planned_dep = train.planned.get(origin)
        if planned_dep is None:
            continue
        by_origin.setdefault(origin, []).append((tid, planned_dep.dep))

    if not by_origin:
        return

    # Pick a random origin station
    chosen = rng.choice(list(by_origin.keys()))
    deps = sorted(by_origin[chosen], key=lambda x: x[1])

    if len(deps) < 2:
        return

    # Pick a random start time that covers at least 2 trains
    t_min = deps[0][1]
    t_max = deps[-1][1]
    if t_max - t_min < 5:
        return

    t0 = rng.uniform(t_min, max(t_min, t_max - window_span * 0.5))
    t1 = t0 + window_span
    bunched = [(tid, d) for tid, d in deps if t0 <= d <= t1]

    if len(bunched) < 2:
        return

    # Delay all bunched trains
    delay = rng.uniform(delay_min, delay_max)
    new_dep = t1 + delay
    for tid, planned_d in bunched:
        j = journeys[tid]
        # Shift arrival_at_current from planned_dep to the new departure time
        extra = new_dep - planned_d
        j.arrival_at_current = max(j.arrival_at_current, new_dep)

