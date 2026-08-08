"""Layer 3 - entry-headway corridor transit with per-class speeds."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
    replan_max_headway_delay_sec: float = 120.0
    platform_min_gap_sec: float = 5.0
    train_speeds: Dict[str, TrainClassSpeed] = field(default_factory=dict)
    default_cruise_mps: float = 25.0
    default_max_mps: float = 30.0
    corridor_distances_m: Dict[str, float] = field(default_factory=dict)
    # Moving-block signalling: continuous safety distance behind the front
    # train's tail (not whole-section clearance).  corridor_sections remain as
    # SPEED ZONES (piecewise speed curve), not occupancy blocks.
    moving_block_safety_m: float = 500.0
    moving_block_margin_sec: float = 3.0
    corridor_sections: Dict[str, list] = field(default_factory=dict)

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
class L3Section:
    """One block section of a corridor. length_m is the calibrated length."""
    length_m: float
    limit_mps: Optional[float] = None  # per-section speed restriction (None = train speed)


@dataclass
class L3SectionOccupancy:
    """Per-section entry/exit times of one train on one corridor."""
    train_id: str
    entry_times: List[float]
    exit_times: List[float]


@dataclass
class L3CorridorState:
    _last_entry: Dict[str, float] = field(default_factory=dict)
    _occupants: Dict[str, List[L3SectionOccupancy]] = field(default_factory=dict)

    def earliest_entry(self, corridor_id: str, headway: float) -> float:
        return self._last_entry.get(corridor_id, -float("inf")) + headway

    def front_time_at(self, corridor_id: str, x: float,
                      sec_bounds: List[float]) -> float:
        """Absolute time the front occupant reaches position x (linear interp).
        +inf when the corridor is empty (no constraint)."""
        occs = self._occupants.get(corridor_id, [])
        if not occs or not sec_bounds:
            return float("inf")
        front = occs[-1]
        total = sec_bounds[-1]
        x = min(max(x, 0.0), total)
        for k in range(len(sec_bounds) - 1):
            a, b = sec_bounds[k], sec_bounds[k + 1]
            if a <= x <= b:
                span = b - a
                if span <= 1e-9:
                    return front.entry_times[k]
                frac = (x - a) / span
                return front.entry_times[k] + frac * (
                    front.exit_times[k] - front.entry_times[k])
        return front.exit_times[-1]

    def earliest_moving_entry(self, corridor_id: str, run_times: List[float],
                              sec_bounds: List[float], safety_m: float,
                              margin_sec: float) -> float:
        """Moving-block earliest corridor entry.

        A following train may be at position x only after the front occupant is
        at least *safety_m* ahead of x (continuous safety distance — braking +
        reaction, NOT whole-section clearance).  sec_bounds are the shared
        cumulative section boundaries (position in m).  Returns -inf when the
        corridor is empty.
        """
        occs = self._occupants.get(corridor_id, [])
        if not occs or not sec_bounds:
            return -float("inf")
        total = sec_bounds[-1]

        def our_time_to(pos: float) -> float:
            """Our run time (from corridor entry) to reach position pos."""
            pos = min(max(pos, 0.0), total)
            cum = 0.0
            for k, rt in enumerate(run_times):
                a = sec_bounds[k] if k < len(sec_bounds) else total
                b = sec_bounds[k + 1] if k + 1 < len(sec_bounds) else total
                if a <= pos <= b:
                    span = b - a
                    if span <= 1e-9:
                        return cum
                    return cum + (pos - a) / span * rt
                cum += rt
            return cum

        # Constraint (continuous over position): entry >= front_time_at(pos+safety)
        # - our_time_to(pos) for every pos we occupy.  The bound is piecewise-linear,
        # so it suffices to check our section boundaries and front boundaries shifted
        # back by the safety distance.
        positions = set(sec_bounds)
        for k in range(len(sec_bounds)):
            positions.add(sec_bounds[k] - safety_m)
        earliest = -float("inf")
        for pos in positions:
            pos = min(max(pos, 0.0), total)
            earliest = max(earliest,
                           self.front_time_at(corridor_id, pos + safety_m, sec_bounds)
                           - our_time_to(pos))
        return earliest + margin_sec

    def record_occupancy(self, corridor_id: str, train_id: str,
                         entry: float, run_times: List[float]) -> None:
        entries: List[float] = []
        exits: List[float] = []
        cum = entry
        for rt in run_times:
            entries.append(cum)
            cum += rt
            exits.append(cum)
        self._occupants.setdefault(corridor_id, []).append(
            L3SectionOccupancy(train_id, entries, exits))

    def record_occupancy_abs(self, corridor_id: str, train_id: str,
                             entry_times: List[float],
                             exit_times: List[float]) -> None:
        """Record per-section occupancy with explicit absolute entry/exit times
        (used by the stepwise path where headway waits shift each section)."""
        self._occupants.setdefault(corridor_id, []).append(
            L3SectionOccupancy(train_id, list(entry_times), list(exit_times)))

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
                 corridor_distances=None, throat_headways=None,
                 sections=None):
        self.p = physics
        self._corridors = L3CorridorState()
        self._throats = L3ThroatState()
        self._headways = corridor_headways or {}
        self._throat_hw = throat_headways or {}
        self._distances = corridor_distances or dict(physics.corridor_distances_m)
        # corridor_id -> List[L3Section] (calibrated); when present, transit uses
        # moving-block physics with a per-section speed profile.
        self._sections = sections or {}
        # cumulative section boundaries (position in m) per corridor
        self._sec_bounds: Dict[str, List[float]] = {}
        for cid, secs in self._sections.items():
            bounds = [0.0]
            for s in secs:
                bounds.append(bounds[-1] + s.length_m)
            self._sec_bounds[cid] = bounds

    def throat_gap(self, throat_id: str, departure: float) -> float:
        """Seconds until throat is free after *departure*."""
        earliest = self._throats.earliest_exit(throat_id, self.p.throat_headway_sec)
        return max(0.0, earliest - departure)

    def transit(self, train_id, from_station, to_station, corridor_id,
                planned_total_sec, actual_departure, speed_choice=SpeedMode.CRUISE,
                throat_conflict_group=None, priority=2):
        p = self.p
        speeds = p.get_speeds(priority)
        sections = self._sections.get(corridor_id)
        run_list: List[float] = []
        if sections:
            cruise_t = self._section_run_times(sections, speeds.cruise_mps)
            fast_t = self._section_run_times(sections, speeds.max_mps)
            slow_t = self._section_run_times(sections, speeds.cruise_mps / 1.2)
            cruise = sum(cruise_t); fast = sum(fast_t); slow = sum(slow_t)
            recovery = max(cruise - fast, 0.0)
            distance = sum(s.length_m for s in sections)
        else:
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
            if sections: run_list = fast_t
        elif speed_choice == SpeedMode.SLOW:
            run = slow
            if sections: run_list = slow_t
        else:
            run = cruise
            if sections: run_list = cruise_t

        # Throat headway check (before corridor entry)
        throat_exit = actual_departure + p.throat_out_sec
        if throat_conflict_group:
            throat_id = f"{from_station}/{throat_conflict_group}"
            earliest_throat = self._throats.earliest_exit(throat_id, self.p.throat_headway_sec)
            if throat_exit < earliest_throat:
                result.throat_delay_sec = earliest_throat - throat_exit
                throat_exit = earliest_throat

        corridor_enter = throat_exit  # throat-out feeds into corridor entry
        if sections:
            # Moving-block: continuous safety distance (safety_m + margin) behind
            # the front train's tail; corridor_headway_sec is an absolute floor.
            earliest = max(
                self._corridors.earliest_entry(corridor_id, p.corridor_headway_sec),
                self._corridors.earliest_moving_entry(
                    corridor_id, run_list, self._sec_bounds.get(corridor_id, []),
                    p.moving_block_safety_m, p.moving_block_margin_sec))
        else:
            hw = self._headways.get(corridor_id, p.corridor_headway_sec)
            earliest = self._corridors.earliest_entry(corridor_id, hw)
        if corridor_enter < earliest:
            result.headway_delay_sec = earliest - corridor_enter
            corridor_enter = earliest
        result.actual_transit_sec = (p.throat_out_sec + result.throat_delay_sec +
            p.approach_sec + result.headway_delay_sec + run + p.throat_in_sec)
        result.actual_arrival = actual_departure + result.actual_transit_sec

        # Record throat exit, corridor entry, and section occupancy
        if throat_conflict_group:
            self._throats.record_exit(throat_id, throat_exit)
        self._corridors.record_entry(corridor_id, corridor_enter)
        if sections and run_list:
            self._corridors.record_occupancy(
                corridor_id, train_id, corridor_enter, run_list)
        return result

    # ------------------------------------------------------------------
    # block-section helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _section_run_times(sections: List[L3Section], v: float) -> List[float]:
        """Per-section traverse times at effective speed v (min with section limit)."""
        times = []
        for s in sections:
            lim = s.limit_mps
            eff = min(v, lim) if lim else v
            times.append(s.length_m / max(eff, 1e-6))
        return times

    def section_run_time(self, section: L3Section, speed_mode: SpeedMode,
                         priority: int) -> float:
        """Traverse time for one section at a speed mode (min with section limit)."""
        speeds = self.p.get_speeds(priority)
        if speed_mode == SpeedMode.FAST:
            v = speeds.max_mps
        elif speed_mode == SpeedMode.SLOW:
            v = speeds.cruise_mps / 1.2
        else:
            v = speeds.cruise_mps
        lim = section.limit_mps
        eff = min(v, lim) if lim else v
        return section.length_m / max(eff, 1e-6)

    def speed_info(self, corridor_id: str, priority: int) -> Tuple[float, float]:
        """Return (distance_m, recovery_margin_sec) for observations."""
        speeds = self.p.get_speeds(priority)
        sections = self._sections.get(corridor_id)
        if sections:
            dist = sum(s.length_m for s in sections)
            cruise = sum(self._section_run_times(sections, speeds.cruise_mps))
            fast = sum(self._section_run_times(sections, speeds.max_mps))
            return dist, max(cruise - fast, 0.0)
        dist = self._distances.get(corridor_id, 0.0)
        if dist > 0:
            return dist, max(dist / speeds.cruise_mps - dist / speeds.max_mps, 0.0)
        return 0.0, 0.0

    def corridor_length(self, corridor_id: str) -> float:
        sections = self._sections.get(corridor_id)
        if sections:
            return sum(s.length_m for s in sections)
        return self._distances.get(corridor_id, 0.0)

    def max_corridor_length(self, default: float = 5000.0) -> float:
        lens = list(self._distances.values())
        for secs in self._sections.values():
            lens.append(sum(s.length_m for s in secs))
        return max(lens) if lens else default

    def next_entry_earliest(self, corridor_id: str,
                            run_times: Optional[List[float]] = None) -> float:
        """Earliest allowed corridor entry: moving-block safety distance when
        sections exist, else the moving-block floor."""
        p = self.p
        earliest = self._corridors.earliest_entry(corridor_id, p.corridor_headway_sec)
        if self._sections.get(corridor_id) and run_times:
            earliest = max(
                earliest,
                self._corridors.earliest_moving_entry(
                    corridor_id, run_times, self._sec_bounds.get(corridor_id, []),
                    p.moving_block_safety_m, p.moving_block_margin_sec))
        return earliest

    def estimate_headway_gap(self, corridor_id: str, departure: float,
                             priority: int) -> float:
        """Predicted corridor wait (s) for a CRUISE departure at *priority*,
        block-aware.  Used for the L3 observation before the speed is chosen."""
        p = self.p
        sections = self._sections.get(corridor_id)
        run_times = None
        if sections:
            speeds = p.get_speeds(priority)
            run_times = self._section_run_times(sections, speeds.cruise_mps)
        earliest = self.next_entry_earliest(corridor_id, run_times)
        return max(0.0, earliest - (departure + p.throat_out_sec))

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
        # default_factory fields are not class attributes, so hasattr() misses them
        elif k == "corridor_sections": kw["corridor_sections"] = dict(v) if v else {}
        elif hasattr(L3Physics, k): kw[k] = v
    return L3Physics(**kw)

def compute_corridor_headway(trains, physics):
    deps = {}
    for tid, t in trains.items():
        route = _train_route(t)
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
        route = _train_route(t)
        for i, st_a in enumerate(route):
            if i+1 >= len(route): continue
            st_b = route[i+1]
            cid = f"{st_a}_to_{st_b}"
            transit = t.planned[st_b].arr - t.planned[st_a].dep
            if transit > 0:
                mt[cid] = min(mt.get(cid, float("inf")), transit)
    return {cid: max(t - overhead, 1.0) * speeds.cruise_mps for cid, t in mt.items()}


def _sec_len(raw) -> float:
    return float(raw.get("length_m", 0.0)) if isinstance(raw, dict) else float(raw)


def _sec_limit(raw) -> Optional[float]:
    if isinstance(raw, dict):
        v = raw.get("limit_mps")
        return float(v) if v is not None else None
    return None


_STATION_ORDER_UP = ["A", "B", "C", "D", "E"]
_STATION_ORDER_DOWN = ["E", "D", "C", "B", "A"]


def _train_route(t) -> List[str]:
    """Actual origin->destination station list for a train.

    Some configs carry placeholder planned entries for stations outside the
    train's route (e.g. G1487 has E/D zeros before its origin C); walking
    ``planned.keys()`` would invent fake legs.  Derive the real route from
    direction + origin/destination instead.
    """
    order = _STATION_ORDER_UP if t.direction == "up" else _STATION_ORDER_DOWN
    try:
        i = order.index(t.origin_station)
        j = order.index(t.destination_station)
    except ValueError:
        return list(t.planned.keys())
    if j >= i:
        return order[i:j + 1]
    return order[j:i + 1][::-1]


def _reference_transit(corridor_id: str, trains,
                       priority: Optional[int]) -> Optional[float]:
    """Min planned transit on a corridor, filtered to a priority class."""
    vals: List[float] = []
    for t in trains.values():
        route = _train_route(t)
        for i, a in enumerate(route[:-1]):
            b = route[i + 1]
            if f"{a}_to_{b}" == corridor_id:
                tr = t.planned[b].arr - t.planned[a].dep
                if tr > 0 and (priority is None or t.priority == priority):
                    vals.append(tr)
    if not vals and priority is not None:
        # fall back to any train on this corridor
        for t in trains.values():
            route = _train_route(t)
            for i, a in enumerate(route[:-1]):
                b = route[i + 1]
                if f"{a}_to_{b}" == corridor_id:
                    tr = t.planned[b].arr - t.planned[a].dep
                    if tr > 0:
                        vals.append(tr)
    return min(vals) if vals else None


def calibrate_corridor_sections(trains, physics: L3Physics,
                                reference_priority: int = 2) -> Dict[str, List[L3Section]]:
    """Scale each corridor's section lengths so that CRUISE at the reference
    priority reproduces the timetable reference transit (min planned transit of
    that priority on the corridor).  On-time departure + CRUISE -> on-time
    arrival, fixing the systematic early-arrival that masked L3 recovery value.
    Sections on corridors without a reference transit keep raw lengths.
    """
    raw = physics.corridor_sections or {}
    ref_speeds = physics.get_speeds(reference_priority)
    overhead = physics.corridor_overhead_sec
    out: Dict[str, List[L3Section]] = {}
    for cid, secs in raw.items():
        if not secs:
            continue
        ref_transit = _reference_transit(cid, trains, reference_priority)
        raw_run = sum(
            _sec_len(s) / max(
                min(ref_speeds.cruise_mps, _sec_limit(s) or ref_speeds.cruise_mps), 1e-6)
            for s in secs)
        if ref_transit is not None and raw_run > 0:
            scale = max(ref_transit - overhead, 1.0) / raw_run
        else:
            scale = 1.0
        out[cid] = [L3Section(_sec_len(s) * scale, _sec_limit(s)) for s in secs]
    return out
