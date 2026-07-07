"""Layer 3 — Python event-driven physical simulator.

Decomposes platform-to-platform corridor transit into physical segments
(throat → corridor → approach → throat) and enforces corridor headway.
No SUMO dependency.

Keeps per-corridor "last exit time" to enforce headway: a train cannot enter
a corridor until the previous train's exit + headway.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml


class SpeedMode(IntEnum):
    CRUISE = 0  # normal operating speed
    FAST = 1    # max speed — consume recovery margin
    SLOW = 2    # below cruise — create headway buffer



@dataclass
class L3Physics:
    throat_in_sec: float = 3.0
    throat_out_sec: float = 3.0
    approach_sec: float = 3.0
    cruise_speed_mps: float = 25.0
    max_speed_mps: float = 30.0
    corridor_headway_sec: float = 15.0
    platform_min_gap_sec: float = 5.0

    @property
    def station_overhead_sec(self) -> float:
        """Physical time consumed inside a station (not corridor)."""
        return self.throat_in_sec + self.throat_out_sec

    @property
    def corridor_overhead_sec(self) -> float:
        """Physical time added at corridor entry/exit."""
        return self.throat_out_sec + self.approach_sec + self.throat_in_sec


@dataclass
class L3CorridorState:
    """Tracks last entry time per corridor for headway enforcement.

    Headway = minimum time between successive trains entering the same corridor
    (moving-block / following-interval semantics).  NOT absolute block.
    """
    _last_entry: Dict[str, float] = field(default_factory=dict)

    def earliest_entry(self, corridor_id: str, headway: float) -> float:
        """Earliest time a train can enter *corridor_id*."""
        return self._last_entry.get(corridor_id, -float("inf")) + headway

    def record_entry(self, corridor_id: str, entry_time: float) -> None:
        self._last_entry[corridor_id] = entry_time


@dataclass
class L3TransitResult:
    """Physical transit from one station platform to the next."""
    train_id: str
    from_station: str
    to_station: str
    corridor_id: str

    # L2 planned platform-to-platform transit (from timetable)
    planned_total_sec: float

    # Physical segments
    distance_m: float                     # corridor length (derived)
    corridor_run_planned_sec: float       # cruise-speed corridor time
    corridor_run_min_sec: float           # max-speed corridor time (recovery floor)
    recovery_margin_sec: float            # max time that can be recovered via speed

    # Actual execution
    headway_delay_sec: float = 0.0        # delay waiting for headway
    recovery_used_sec: float = 0.0        # time saved by going FAST (vs cruise)
    actual_transit_sec: float = 0.0       # total platform→platform actual time
    actual_arrival: float = 0.0           # arrival time at next station


class L3Simulator:
    """Event-driven L3 simulator — per-train corridor transit."""

    def __init__(self, physics: L3Physics, corridor_headways: Dict[str, float] | None = None,
                 corridor_distances: Dict[str, float] | None = None):
        self.p = physics
        self._corridors = L3CorridorState()
        self._headways = corridor_headways or {}
        self._distances = corridor_distances or {}

    def transit(
        self,
        train_id: str,
        from_station: str,
        to_station: str,
        corridor_id: str,
        planned_total_sec: float,
        actual_departure: float,
        speed_choice: SpeedMode = SpeedMode.CRUISE,
    ) -> L3TransitResult:
        """Compute actual corridor transit with physical decomposition.

        Decomposes platform-to-platform transit into:
          throat_out → approach → corridor_run → throat_in

        *speed_choice* selects the corridor running speed:
          CRUISE — normal speed, no recovery consumed
          FAST   — max speed, consumes recovery margin to save time
          SLOW   — below cruise, creates headway buffer for following trains

        Headway is checked at corridor entry (after throat_out).
        """
        p = self.p
        distance = self._distances.get(corridor_id, 0.0)

        if distance > 0:
            corridor_run_cruise = distance / p.cruise_speed_mps
            corridor_run_fast = distance / p.max_speed_mps
            corridor_run_slow = corridor_run_cruise * 1.2
            recovery_margin = max(corridor_run_cruise - corridor_run_fast, 0.0)
        else:
            corridor_run_cruise = planned_total_sec
            corridor_run_fast = planned_total_sec
            corridor_run_slow = planned_total_sec * 1.2
            recovery_margin = 0.0

        result = L3TransitResult(
            train_id=train_id,
            from_station=from_station,
            to_station=to_station,
            corridor_id=corridor_id,
            planned_total_sec=planned_total_sec,
            distance_m=distance,
            corridor_run_planned_sec=corridor_run_cruise,
            corridor_run_min_sec=corridor_run_fast,
            recovery_margin_sec=recovery_margin,
        )

        # Select corridor run time based on speed choice
        if speed_choice == SpeedMode.FAST:
            corridor_run = corridor_run_fast
            result.recovery_used_sec = recovery_margin
        elif speed_choice == SpeedMode.SLOW:
            corridor_run = corridor_run_slow
        else:
            corridor_run = corridor_run_cruise

        # Headway check — per-corridor entry-time-based
        corridor_enter = actual_departure + p.throat_out_sec
        hw = self._headways.get(corridor_id, p.corridor_headway_sec)
        earliest_enter = self._corridors.earliest_entry(corridor_id, hw)
        if corridor_enter < earliest_enter:
            result.headway_delay_sec = earliest_enter - corridor_enter
            corridor_enter = earliest_enter

        result.actual_transit_sec = (
            p.throat_out_sec + p.approach_sec + result.headway_delay_sec
            + corridor_run + p.throat_in_sec
        )
        result.actual_arrival = actual_departure + result.actual_transit_sec

        # Record corridor entry for next train's headway
        self._corridors.record_entry(corridor_id, corridor_enter)

        return result

    def reset(self) -> None:
        self._corridors = L3CorridorState()


def load_l3_physics(config_dir: Path) -> L3Physics:
    """Load L3 physics from l3_physics.yaml, with sensible defaults."""
    path = config_dir / "l3_physics.yaml"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f).get("l3_physics", {})
    else:
        raw = {}
    return L3Physics(**{k: v for k, v in raw.items() if hasattr(L3Physics, k)})


def compute_corridor_headway(trains: dict, physics: L3Physics) -> dict:
    """Derive per-corridor headway from timetable minimum departure spacing.

    For each corridor, collect planned departure times of trains entering it,
    sort ascending, and take the minimum gap.  Falls back to physics default
    if fewer than 2 trains use the corridor.
    """
    corridor_deps: Dict[str, list] = {}
    for tid, train in trains.items():
        planned = train.planned
        for i, (st_a, _plan_a) in enumerate(planned.items()):
            route_stations = list(planned.keys())
            if i + 1 >= len(route_stations):
                continue
            st_b = route_stations[i + 1]
            corridor_key = f"{st_a}_to_{st_b}"
            dep = planned[st_a].dep
            corridor_deps.setdefault(corridor_key, []).append(dep)

    headways = {}
    for cid, deps in corridor_deps.items():
        deps = sorted(deps)
        if len(deps) >= 2:
            gaps = [deps[j] - deps[j - 1] for j in range(1, len(deps))]
            headways[cid] = max(min(gaps), physics.corridor_headway_sec)
        else:
            headways[cid] = physics.corridor_headway_sec

    return headways


def compute_corridor_distances(
    trains: dict, physics: L3Physics,
) -> Dict[str, float]:
    """Derive per-corridor distance from minimum planned transit time.

    For each corridor, finds the shortest planned platform-to-platform transit
    across all trains and derives pure corridor distance by subtracting
    the station overhead (throat_out + approach + throat_in).

    The timetable overhead is already baked into planned transit times —
    subtracting the config overhead yields the pure cruise-distance component.
    Distance = (min_transit - overhead) * cruise_speed.
    """
    overhead = physics.throat_out_sec + physics.approach_sec + physics.throat_in_sec
    min_transit: Dict[str, float] = {}
    for tid, train in trains.items():
        planned = train.planned
        route_stations = list(planned.keys())
        for i, st_a in enumerate(route_stations):
            if i + 1 >= len(route_stations):
                continue
            st_b = route_stations[i + 1]
            corridor_key = f"{st_a}_to_{st_b}"
            transit = planned[st_b].arr - planned[st_a].dep
            if transit > 0:
                prev = min_transit.get(corridor_key, float("inf"))
                min_transit[corridor_key] = min(prev, transit)

    distances: Dict[str, float] = {}
    for cid, mt in min_transit.items():
        pure_run = max(mt - overhead, 1.0)
        distances[cid] = pure_run * physics.cruise_speed_mps

    return distances
