"""Layer 3 - entry-headway corridor transit with per-class speeds."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, Optional
import yaml

class SpeedMode(IntEnum):
    CRUISE = 0; FAST = 1; SLOW = 2

@dataclass
class TrainClassSpeed:
    cruise_mps: float; max_mps: float

@dataclass
class L3Physics:
    throat_in_sec: float = 3.0
    throat_out_sec: float = 3.0
    approach_sec: float = 3.0
    corridor_headway_sec: float = 15.0
    throat_headway_sec: float = 6.0
    replan_threshold_headway_mult: float = 4.0
    platform_min_gap_sec: float = 5.0
    train_speeds: Dict[str, TrainClassSpeed] = field(default_factory=dict)
    default_cruise_mps: float = 25.0
    default_max_mps: float = 30.0
    corridor_distances_m: Dict[str, float] = field(default_factory=dict)

    def get_speeds(self, priority: int) -> TrainClassSpeed:
        """Look up speeds by train priority (1=Fuxing, 2=Hexie, 3=K)."""
        key = str(priority)
        if key in self.train_speeds:
            return self.train_speeds[key]
        return TrainClassSpeed(self.default_cruise_mps, self.default_max_mps)

    @property
    def station_overhead_sec(self) -> float:
        return self.throat_in_sec + self.throat_out_sec

    @property
    def corridor_overhead_sec(self) -> float:
        return self.throat_out_sec + self.approach_sec + self.throat_in_sec

@dataclass
class L3CorridorState:
    _last_entry: Dict[str, float] = field(default_factory=dict)
    def earliest_entry(self, corridor_id: str, headway: float) -> float:
        return self._last_entry.get(corridor_id, -float("inf")) + headway
    def record_entry(self, corridor_id: str, entry_time: float) -> None:
        self._last_entry[corridor_id] = entry_time

@dataclass
class L3ThroatState:
    """Track throat occupancy per station+conflict_group."""
    _last_exit: Dict[str, float] = field(default_factory=dict)
    def earliest_exit(self, throat_id: str, headway: float) -> float:
        return self._last_exit.get(throat_id, -float("inf")) + headway
    def record_exit(self, throat_id: str, exit_time: float) -> None:
        self._last_exit[throat_id] = exit_time

@dataclass
class L3TransitResult:
    train_id: str; from_station: str; to_station: str; corridor_id: str
    planned_total_sec: float; distance_m: float
    corridor_run_cruise_sec: float; corridor_run_fast_sec: float
    recovery_margin_sec: float
    headway_delay_sec: float = 0.0; recovery_used_sec: float = 0.0
    throat_delay_sec: float = 0.0
    actual_transit_sec: float = 0.0; actual_arrival: float = 0.0

class L3Simulator:
    def __init__(self, physics: L3Physics, corridor_headways=None,
                 corridor_distances=None, throat_headways=None):
        self.p = physics
        self._corridors = L3CorridorState()
        self._throats = L3ThroatState()
        self._headways = corridor_headways or {}
        self._throat_hw = throat_headways or {}
        self._distances = corridor_distances or dict(physics.corridor_distances_m)

    def throat_gap(self, throat_id: str, departure: float) -> float:
        """Seconds until throat is free after *departure*."""
        earliest = self._throats.earliest_exit(throat_id, self.p.throat_headway_sec)
        return max(0.0, earliest - departure)

    def transit(self, train_id, from_station, to_station, corridor_id,
                planned_total_sec, actual_departure, speed_choice=SpeedMode.CRUISE,
                throat_conflict_group=None, priority=2):
        p = self.p
        speeds = p.get_speeds(priority)
        distance = self._distances.get(corridor_id, 0.0)
        if distance > 0:
            cruise = distance / speeds.cruise_mps
            fast = distance / speeds.max_mps
            slow = cruise * 1.2
            recovery = max(cruise - fast, 0.0)
        else:
            cruise = fast = planned_total_sec; slow = planned_total_sec * 1.2
            recovery = 0.0
        result = L3TransitResult(train_id=train_id, from_station=from_station,
            to_station=to_station, corridor_id=corridor_id,
            planned_total_sec=planned_total_sec, distance_m=distance,
            corridor_run_cruise_sec=cruise, corridor_run_fast_sec=fast,
            recovery_margin_sec=recovery)
        if speed_choice == SpeedMode.FAST:
            run = fast; result.recovery_used_sec = recovery
        elif speed_choice == SpeedMode.SLOW:
            run = slow
        else:
            run = cruise

        # Throat headway check (before corridor entry)
        throat_exit = actual_departure + p.throat_out_sec
        if throat_conflict_group:
            throat_id = f"{from_station}/{throat_conflict_group}"
            earliest_throat = self._throats.earliest_exit(throat_id, self.p.throat_headway_sec)
            if throat_exit < earliest_throat:
                result.throat_delay_sec = earliest_throat - throat_exit
                throat_exit = earliest_throat

        corridor_enter = throat_exit  # throat-out feeds into corridor entry
        hw = self._headways.get(corridor_id, p.corridor_headway_sec)
        earliest = self._corridors.earliest_entry(corridor_id, hw)
        if corridor_enter < earliest:
            result.headway_delay_sec = earliest - corridor_enter
            corridor_enter = earliest
        result.actual_transit_sec = (p.throat_out_sec + result.throat_delay_sec +
            p.approach_sec + result.headway_delay_sec + run + p.throat_in_sec)
        result.actual_arrival = actual_departure + result.actual_transit_sec

        # Record throat exit and corridor entry
        if throat_conflict_group:
            self._throats.record_exit(throat_id, throat_exit)
        self._corridors.record_entry(corridor_id, corridor_enter)
        return result

    def reset(self):
        self._corridors = L3CorridorState()
        self._throats = L3ThroatState()

def _parse_train_speeds(raw):
    result = {}
    for cls, vals in (raw or {}).items():
        if isinstance(vals, dict):
            result[cls] = TrainClassSpeed(float(vals.get('cruise_mps',25.0)),
                                          float(vals.get('max_mps',30.0)))
    return result

def load_l3_physics(config_dir):
    path = Path(config_dir) if isinstance(config_dir, str) else config_dir
    path = path / "l3_physics.yaml"
    if not path.exists(): return L3Physics()
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f).get("l3_physics", {})
    kw = {}
    for k, v in raw.items():
        if k == "train_speeds": kw["train_speeds"] = _parse_train_speeds(v)
        elif k == "corridor_distances_m": kw["corridor_distances_m"] = dict(v) if v else {}
        elif hasattr(L3Physics, k): kw[k] = v
    return L3Physics(**kw)

def compute_corridor_headway(trains, physics):
    deps = {}
    for tid, t in trains.items():
        route = list(t.planned.keys())
        for i, st_a in enumerate(route):
            if i+1 >= len(route): continue
            st_b = route[i+1]
            cid = f"{st_a}_to_{st_b}"
            deps.setdefault(cid, []).append(t.planned[st_a].dep)
    hw = {}
    for cid, ds in deps.items():
        ds = sorted(ds)
        hw[cid] = max(min(ds[j]-ds[j-1] for j in range(1,len(ds))), physics.corridor_headway_sec) if len(ds)>=2 else physics.corridor_headway_sec
    return hw

def compute_corridor_distances(trains, physics):
    speeds = physics.get_speeds(2)  # priority 2 (Hexie) as reference
    overhead = physics.corridor_overhead_sec
    mt = {}
    for tid, t in trains.items():
        route = list(t.planned.keys())
        for i, st_a in enumerate(route):
            if i+1 >= len(route): continue
            st_b = route[i+1]
            cid = f"{st_a}_to_{st_b}"
            transit = t.planned[st_b].arr - t.planned[st_a].dep
            if transit > 0:
                mt[cid] = min(mt.get(cid, float("inf")), transit)
    return {cid: max(t - overhead, 1.0) * speeds.cruise_mps for cid, t in mt.items()}
