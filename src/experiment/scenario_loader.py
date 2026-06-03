"""
Phase 1: Scenario configuration loader and cross-file consistency validator.

Loads and validates:
  - configs/railway_resources.yaml   → physical infrastructure
  - configs/timetable_layer1.yaml    → train schedule + Layer1 bounds
  - sumo/routes/base.rou.xml         → SUMO vehicle definitions

Validation rules (per coding_plan Phase 1):
  1. Every timetable train has a matching <vehicle> in rou.xml.
  2. Every platform referenced in timetable exists in railway_resources.
  3. Every corridor referenced in timetable exists in railway_resources.
  4. dep_max >= arr_min + dwell_min for every train at every station.
  5. Physical platform edges referenced in resources exist in rou.xml routes.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml


# ---------------------------------------------------------------------------
# data-classes
# ---------------------------------------------------------------------------

@dataclass
class ThroatRoute:
    id: str
    edges: List[str]
    platform: str
    conflict_group: str


@dataclass
class DirectionConfig:
    approach_edge: Optional[str]
    depart_edge: Optional[str]
    in_routes: List[ThroatRoute]
    out_routes: List[ThroatRoute]


@dataclass
class PlatformConfig:
    id: str
    up_edge: str
    down_edge: str
    through_allowed: bool = True


@dataclass
class StationConfig:
    id: str
    platforms: List[PlatformConfig]
    up: DirectionConfig
    down: DirectionConfig
    dwell_min_sec_stop: int = 60


@dataclass
class CorridorConfig:
    id: str
    direction: str
    from_station: str
    to_station: str
    edges: List[str]
    min_headway_sec: float


@dataclass
class TrainBound:
    arr_min: float
    dep_max: float
    dwell_min: float


@dataclass
class TrainPlan:
    arr: float
    dep: float


@dataclass
class TrainTimetable:
    id: str
    direction: str              # "up" or "down"
    train_class: str            # "G", "D", "Z", "K"
    priority: int               # 1 (highest) .. 3
    pass_through: bool
    origin_station: str
    destination_station: str
    corridor: str
    planned: Dict[str, TrainPlan]     # station_id -> TrainPlan
    bounds: Dict[str, TrainBound]     # station_id -> TrainBound
    candidate_platforms: Dict[str, List[str]]  # station_id -> [platform_ids]


@dataclass
class SUMOVehicle:
    id: str
    vtype: str
    route_id: str
    depart: float


@dataclass
class SUMORoute:
    id: str
    edges: List[str]


@dataclass
class SUMOVType:
    id: str
    length: float
    maxSpeed: float
    accel: float
    decel: float


@dataclass
class ScenarioConfig:
    """Immutable snapshot of all validated config data."""
    project_root: Path
    network: Dict[str, Any]
    stations: Dict[str, StationConfig]
    corridors: Dict[str, CorridorConfig]
    trains: Dict[str, TrainTimetable]
    sumo_routes: Dict[str, SUMORoute]
    sumo_vehicles: Dict[str, SUMOVehicle]
    sumo_vtypes: Dict[str, SUMOVType]
    priority_map: Dict[str, int]
    time_generation: Dict[str, Any]

    def sumo_net_path(self) -> Path:
        if "sumo_net" not in self.network:
            raise ValidationError("network.sumo_net is required in railway_resources.yaml")
        return self.project_root / self.network["sumo_net"]

    def platform_edge(self, station_id: str, platform_id: str,
                      direction: str) -> str:
        station = self.stations[station_id]
        platform = next(p for p in station.platforms if p.id == platform_id)
        return platform.up_edge if direction == "up" else platform.down_edge

    def route_for_platform(self, train: "TrainTimetable",
                           origin_platform: str) -> Tuple[str, Optional[str]]:
        """Find a SUMO route and destination platform edge from config data.

        The lookup is intentionally structural: it matches configured platform
        and corridor edges against SUMO route edges.  This keeps Python free of
        assumptions about route names, station names, or platform suffixes.
        """
        origin_edge = self.platform_edge(
            train.origin_station, origin_platform, train.direction)
        corridor_edges = set(self.corridors[train.corridor].edges) \
            if train.corridor in self.corridors else set()

        destination_edges = {
            self.platform_edge(train.destination_station, p.id, train.direction)
            for p in self.stations[train.destination_station].platforms
        }

        for route in self.sumo_routes.values():
            edge_set = set(route.edges)
            if origin_edge not in edge_set:
                continue
            if corridor_edges and not corridor_edges.issubset(edge_set):
                continue
            dest = next((e for e in route.edges if e in destination_edges), None)
            return route.id, dest

        raise ValidationError(
            f"No SUMO route contains platform edge '{origin_edge}' "
            f"and corridor '{train.corridor}' for train '{train.id}'")


class ValidationError(Exception):
    """Raised when cross-file consistency checks fail."""


# ---------------------------------------------------------------------------
# YAML loaders
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_resources(path: Path) -> Tuple[
    Dict[str, Any],
    Dict[str, StationConfig],
    Dict[str, CorridorConfig],
]:
    raw = _load_yaml(path)
    network = raw.get("network", {})

    stations: Dict[str, StationConfig] = {}
    for s in raw.get("stations", []):
        platforms = [
            PlatformConfig(
                id=p["id"],
                up_edge=p["up_edge"],
                down_edge=p["down_edge"],
                through_allowed=p.get("through_allowed", True),
            )
            for p in s["platforms"]
        ]
        dwell_stop = s.get("dwell_min_sec", {}).get("stop", 60)

        def _dir(d: Dict) -> DirectionConfig:
            return DirectionConfig(
                approach_edge=d.get("approach_edge"),
                depart_edge=d.get("depart_edge"),
                in_routes=[ThroatRoute(**r) for r in d.get("in_routes", [])],
                out_routes=[ThroatRoute(**r) for r in d.get("out_routes", [])],
            )

        stations[s["id"]] = StationConfig(
            id=s["id"],
            platforms=platforms,
            up=_dir(s.get("up", {})),
            down=_dir(s.get("down", {})),
            dwell_min_sec_stop=dwell_stop,
        )

    corridors: Dict[str, CorridorConfig] = {}
    for c in raw.get("corridors", []):
        corridors[c["id"]] = CorridorConfig(
            id=c["id"],
            direction=c["direction"],
            from_station=c["from_station"],
            to_station=c["to_station"],
            edges=c.get("edges", []),
            min_headway_sec=float(c.get("min_headway_sec", 120)),
        )

    return network, stations, corridors


def _parse_timetable(path: Path) -> Tuple[
    Dict[str, TrainTimetable],
    Dict[str, int],
    Dict[str, Any],
]:
    raw = _load_yaml(path)

    priority_map: Dict[str, int] = raw.get("priority_map", {})
    time_gen: Dict[str, Any] = raw.get("time_generation", {})

    trains: Dict[str, TrainTimetable] = {}
    for t in raw.get("trains", []):
        planned = {
            sid: TrainPlan(arr=v["arr"], dep=v["dep"])
            for sid, v in t.get("planned", {}).items()
        }
        bounds = {
            sid: TrainBound(
                arr_min=v["arr_min"],
                dep_max=v["dep_max"],
                dwell_min=v["dwell_min"],
            )
            for sid, v in t.get("bounds", {}).items()
        }
        route = t.get("route", {})
        trains[t["id"]] = TrainTimetable(
            id=t["id"],
            direction=t["direction"],
            train_class=t["train_class"],
            priority=t.get("priority", 99),
            pass_through=t.get("pass_through", False),
            origin_station=route.get("origin_station", ""),
            destination_station=route.get("destination_station", ""),
            corridor=route.get("corridor", ""),
            planned=planned,
            bounds=bounds,
            candidate_platforms=t.get("candidate_platforms", {}),
        )

    return trains, priority_map, time_gen


def _parse_sumo_routes(path: Path) -> Tuple[
    Dict[str, SUMORoute],
    Dict[str, SUMOVehicle],
    Dict[str, SUMOVType],
]:
    tree = ET.parse(path)
    root = tree.getroot()

    vtypes: Dict[str, SUMOVType] = {}
    for v in root.findall("vType"):
        vtypes[v.get("id")] = SUMOVType(
            id=v.get("id"),
            length=float(v.get("length", 200)),
            maxSpeed=float(v.get("maxSpeed", 50)),
            accel=float(v.get("accel", 1.0)),
            decel=float(v.get("decel", 1.5)),
        )

    routes: Dict[str, SUMORoute] = {}
    for r in root.findall("route"):
        edges = (r.get("edges") or "").split()
        routes[r.get("id")] = SUMORoute(id=r.get("id"), edges=edges)

    vehicles: Dict[str, SUMOVehicle] = {}
    for v in root.findall("vehicle"):
        vehicles[v.get("id")] = SUMOVehicle(
            id=v.get("id"),
            vtype=v.get("type"),
            route_id=v.get("route"),
            depart=float(v.get("depart", 0)),
        )

    return routes, vehicles, vtypes


# ---------------------------------------------------------------------------
# validators
# ---------------------------------------------------------------------------

def _collect_platform_ids(stations: Dict[str, StationConfig]) -> Set[str]:
    ids: Set[str] = set()
    for s in stations.values():
        for p in s.platforms:
            ids.add(p.id)
    return ids


def _collect_all_edges(routes: Dict[str, SUMORoute]) -> Set[str]:
    edges: Set[str] = set()
    for r in routes.values():
        edges.update(r.edges)
    return edges


def _collect_resource_edges(stations: Dict[str, StationConfig],
                            corridors: Dict[str, CorridorConfig]) -> Set[str]:
    edges: Set[str] = set()
    for s in stations.values():
        for d in (s.up, s.down):
            for r in d.in_routes + d.out_routes:
                edges.update(r.edges)
            if d.approach_edge:
                edges.add(d.approach_edge)
            if d.depart_edge:
                edges.add(d.depart_edge)
    for c in corridors.values():
        edges.update(c.edges)
    return edges


def _validate(stations: Dict[str, StationConfig],
              corridors: Dict[str, CorridorConfig],
              trains: Dict[str, TrainTimetable],
              sumo_routes: Dict[str, SUMORoute],
              sumo_vehicles: Dict[str, SUMOVehicle]) -> List[str]:
    errors: List[str] = []

    platform_ids = _collect_platform_ids(stations)
    sumo_edges = _collect_all_edges(sumo_routes)
    resource_edges = _collect_resource_edges(stations, corridors)
    timetable_train_ids = set(trains.keys())
    sumo_vehicle_ids = set(sumo_vehicles.keys())

    # P1-1: train id match
    if timetable_train_ids != sumo_vehicle_ids:
        missing = timetable_train_ids - sumo_vehicle_ids
        extra = sumo_vehicle_ids - timetable_train_ids
        if missing:
            errors.append(
                f"Timetable trains not in rou.xml: {sorted(missing)}")
        if extra:
            errors.append(
                f"rou.xml vehicles not in timetable: {sorted(extra)}")

    # P1-2: platform id match
    for t in trains.values():
        for sid, plats in t.candidate_platforms.items():
            for pid in plats:
                if pid not in platform_ids:
                    errors.append(
                        f"Train '{t.id}': platform '{pid}' at station "
                        f"'{sid}' not found in railway_resources.yaml")

    # P1-3: corridor id match
    for t in trains.values():
        if t.corridor and t.corridor not in corridors:
            errors.append(
                f"Train '{t.id}': corridor '{t.corridor}' "
                f"not found in railway_resources.yaml")

    # P1-4: dep_max >= arr_min + dwell_min
    for t in trains.values():
        for sid, b in t.bounds.items():
            min_required = b.arr_min + b.dwell_min
            if b.dep_max < min_required:
                errors.append(
                    f"Train '{t.id}' at '{sid}': "
                    f"dep_max ({b.dep_max}) < arr_min ({b.arr_min}) "
                    f"+ dwell_min ({b.dwell_min}) = {min_required}")

    # P1-5: resource edges exist in SUMO routes
    missing_edges = resource_edges - sumo_edges
    if missing_edges:
        errors.append(
            f"Resource edges not found in any SUMO route: "
            f"{sorted(missing_edges)}")

    return errors


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def load_scenario(project_root: Optional[Path] = None) -> ScenarioConfig:
    """Load and validate all config files.  Raises ValidationError on failure.

    Args:
        project_root: root dir containing configs/ and sumo/.  Defaults to
                      two levels above this file (src/experiment -> project).
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent

    resources_path = project_root / "configs" / "railway_resources.yaml"
    timetable_path = project_root / "configs" / "timetable_layer1.yaml"
    routes_path = project_root / "sumo" / "routes" / "base.rou.xml"

    for p in (resources_path, timetable_path, routes_path):
        if not p.exists():
            raise ValidationError(f"Config file not found: {p}")

    network, stations, corridors = _parse_resources(resources_path)
    trains, priority_map, time_gen = _parse_timetable(timetable_path)
    sumo_routes, sumo_vehicles, sumo_vtypes = _parse_sumo_routes(routes_path)

    errors = _validate(stations, corridors, trains, sumo_routes, sumo_vehicles)
    if errors:
        raise ValidationError(
            "Scenario validation failed:\n  - " + "\n  - ".join(errors))

    return ScenarioConfig(
        project_root=project_root,
        network=network,
        stations=stations,
        corridors=corridors,
        trains=trains,
        sumo_routes=sumo_routes,
        sumo_vehicles=sumo_vehicles,
        sumo_vtypes=sumo_vtypes,
        priority_map=priority_map,
        time_generation=time_gen,
    )


# ---------------------------------------------------------------------------
# Phase 1 diagnostic
# ---------------------------------------------------------------------------

def print_summary(cfg: ScenarioConfig) -> None:
    """Print a human-readable summary of the loaded scenario."""
    print(f"\n{'='*60}")
    print(f"  Scenario Summary")
    print(f"{'='*60}")

    print(f"\n  Stations ({len(cfg.stations)}):")
    for s in cfg.stations.values():
        plat_ids = [p.id for p in s.platforms]
        print(f"    {s.id}: platforms={plat_ids}, "
              f"dwell_min_stop={s.dwell_min_sec_stop}s")

    print(f"\n  Corridors ({len(cfg.corridors)}):")
    for c in cfg.corridors.values():
        print(f"    {c.id}: {c.from_station}→{c.to_station} "
              f"({c.direction}), edges={c.edges}, "
              f"min_headway={c.min_headway_sec}s")

    print(f"\n  Trains ({len(cfg.trains)}):")
    for t in sorted(cfg.trains.values(), key=lambda x: x.planned[
        x.origin_station].dep if x.origin_station in x.planned else 0):
        p = t.planned
        origin_dep = p[t.origin_station].dep if t.origin_station in p else "?"
        dest_arr = p[t.destination_station].arr if t.destination_station in p else "?"
        flags = []
        if t.pass_through:
            flags.append("pass")
        if t.train_class == "K" and t.bounds.get(t.origin_station) and \
           t.bounds[t.origin_station].dwell_min >= 120:
            flags.append("long-dwell")
        print(f"    {t.id:>6}  {t.origin_station}→{t.destination_station}  "
              f"class={t.train_class}  prio={t.priority}  "
              f"dep={origin_dep}→arr={dest_arr}  "
              f"{'|'.join(flags) if flags else ''}")

    print(f"\n  Priority map: {cfg.priority_map}")
    print(f"  Validated: OK\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        cfg = load_scenario()
        print_summary(cfg)
    except ValidationError as e:
        print(f"\n  ❌ {e}")
        raise SystemExit(1)
