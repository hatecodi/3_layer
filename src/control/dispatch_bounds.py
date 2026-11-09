"""
Layer 1 — Dispatch Centre (rule-based, fixed).

Provides per-train hard envelopes.  Does NOT issue step-level commands.

Reads from the loaded ScenarioConfig.  In Phase 1 the bounds are already
pre-computed in timetable_layer1.yaml.  Later, when using compact timetable
specs, this module will auto-expand them via time_generation parameters.

API:
    layer1 = DispatchBounds(cfg)
    bound = layer1.origin_bounds(next(iter(cfg.trains)))
    bound.t_arr_min   # earliest allowed arrival at current station
    bound.t_dep_max   # latest allowed departure
    bound.dwell_min   # minimum dwell seconds
    bound.prio_i      # priority class (1, 2, 3)
    bound.pass_flag   # True for pass-through trains
    bound.candidate_platforms  # list of platform IDs
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..experiment.scenario_loader import (
    ScenarioConfig,
    TrainTimetable,
    is_station_pass,
)


@dataclass
class TrainBounds:
    """Layer 1 output for a single train at a specific station."""
    train_id: str
    station_id: str
    t_arr_plan: float
    t_dep_plan: float
    t_arr_min: float
    t_dep_max: float
    dwell_min: float
    prio_i: int
    pass_flag: bool
    candidate_platforms: List[str]


class DispatchBounds:
    """Layer 1 dispatch centre — provides hard envelopes per train per station."""

    def __init__(self, cfg: ScenarioConfig):
        self._cfg = cfg
        self._trains = cfg.trains
        self._stations = cfg.stations

    # -- per-train / per-station query ---------------------------------

    def for_train(self, train_id: str,
                  station_id: str) -> TrainBounds:
        """Return Layer 1 bounds for *train_id* at *station_id*."""
        train = self._trains[train_id]
        planned = train.planned.get(station_id)
        bounds = train.bounds.get(station_id)

        return TrainBounds(
            train_id=train_id,
            station_id=station_id,
            t_arr_plan=planned.arr if planned else 0.0,
            t_dep_plan=planned.dep if planned else 0.0,
            t_arr_min=bounds.arr_min if bounds else 0.0,
            t_dep_max=bounds.dep_max if bounds else float("inf"),
            dwell_min=bounds.dwell_min if bounds else 0.0,
            prio_i=train.priority,
            pass_flag=is_station_pass(train, station_id),
            candidate_platforms=train.candidate_platforms.get(station_id, []),
        )

    def origin_bounds(self, train_id: str) -> TrainBounds:
        """Layer 1 bounds at the train's origin station."""
        train = self._trains[train_id]
        return self.for_train(train_id, train.origin_station)

    # -- bulk access ---------------------------------------------------

    @property
    def train_ids(self) -> List[str]:
        return sorted(self._trains.keys())

    @property
    def trains(self) -> Dict[str, TrainTimetable]:
        return self._trains

    def active_stations(self, train_id: str) -> List[str]:
        """Return stations this train visits (origin + destination)."""
        train = self._trains[train_id]
        return [train.origin_station, train.destination_station]

    # -- validation ----------------------------------------------------

    def validate_bounds(self) -> List[str]:
        """Return list of bound-consistency errors (empty = valid)."""
        errors: List[str] = []
        for tid, train in self._trains.items():
            for sid in self.active_stations(tid):
                b = train.bounds.get(sid)
                if b is None:
                    continue
                # dwell floor
                if b.dwell_min < 0:
                    errors.append(f"{tid}@{sid}: dwell_min < 0 ({b.dwell_min})")
                # time-consistency
                if b.dep_max < b.arr_min + b.dwell_min - 1e-6:
                    errors.append(
                        f"{tid}@{sid}: dep_max ({b.dep_max}) < "
                        f"arr_min ({b.arr_min}) + dwell_min ({b.dwell_min})")
        return errors
