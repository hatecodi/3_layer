"""Rule-based SUMO execution layer for granted station windows.

The first implementation is intentionally conservative: it converts granted
Layer 2 windows into a generated SUMO route file with platform stops, then uses
TraCI only to record actual platform arrival/departure events. Resource choices
still come from Layer 2 grants, not from SUMO.
"""
from __future__ import annotations

import csv
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol

from .execution_target import ExecutionTarget
from .resource_manager import GrantResult
from ..experiment.scenario_loader import ScenarioConfig, TrainTimetable


APPROACH_TRAVEL_SEC = 30.0


@dataclass
class ExecutionResult:
    completed_vehicle_ids: List[str]
    expected_vehicle_ids: List[str]
    event_csv: Path
    route_file: Path
    tripinfo_file: Path

    @property
    def completion_rate(self) -> float:
        if not self.expected_vehicle_ids:
            return 1.0
        return len(self.completed_vehicle_ids) / len(self.expected_vehicle_ids)


class ExecutionController(Protocol):
    """Stable Layer 3 interface.

    A future learned Worker should implement this protocol and consume the same
    ExecutionTarget mapping that the current rule-based executor consumes.
    """

    def build_targets(self, grants: Iterable[GrantResult]) -> Dict[str, ExecutionTarget]:
        ...

    def run(
        self,
        targets: Dict[str, ExecutionTarget],
        end_time: float = 1200.0,
    ) -> ExecutionResult:
        ...


class RuleBasedExecutor:
    def __init__(self, cfg: ScenarioConfig, project_root: Optional[Path] = None):
        self.cfg = cfg
        self.project_root = project_root or Path(__file__).resolve().parents[2]
        self.sumo_dir = self.project_root / "sumo"
        self.output_dir = self.sumo_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_targets(self, grants: Iterable[GrantResult]) -> Dict[str, ExecutionTarget]:
        targets: Dict[str, ExecutionTarget] = {}
        for grant in grants:
            train = self.cfg.trains[grant.train_id]
            origin = train.origin_station
            platform_edge = self._platform_edge(origin, grant.platform, train.direction)
            route_id, dest_edge = self.cfg.route_for_platform(train, grant.platform)
            dest_bound = train.bounds.get(train.destination_station)

            targets[grant.train_id] = ExecutionTarget(
                train_id=grant.train_id,
                station_id=origin,
                allocated_platform=grant.platform,
                route_id=route_id,
                platform_edge=platform_edge,
                t_throat_entry=grant.W_in[0],
                t_platform_arrival=grant.W_plat[0],
                t_platform_departure=grant.W_out[0],
                t_throat_exit=grant.W_out[1],
                pass_flag=train.pass_through,
                dwell_min=max(0.0, grant.W_out[0] - grant.W_plat[0]),
                destination_platform_edge=dest_edge,
                destination_dwell_min=0.0 if train.pass_through
                else (dest_bound.dwell_min if dest_bound else 0.0),
            )
        return targets

    def write_route_file(
        self,
        targets: Dict[str, ExecutionTarget],
        filename: str = "generated_exec.rou.xml",
    ) -> Path:
        base_path = self.sumo_dir / "routes" / "base.rou.xml"
        base = ET.parse(base_path).getroot()
        root = ET.Element("routes")

        for child in base:
            if child.tag in {"vType", "route"}:
                root.append(child)

        vehicles = []
        for tid, target in targets.items():
            train = self.cfg.trains[tid]
            depart = max(0.0, target.t_throat_entry - APPROACH_TRAVEL_SEC)
            vehicles.append((depart, tid, train, target))

        for depart, tid, train, target in sorted(vehicles):
            veh = ET.SubElement(root, "vehicle", {
                "id": tid,
                "type": train.train_class,
                "route": target.route_id,
                "depart": f"{depart:.1f}",
                "departLane": "best",
                "departSpeed": "max",
            })
            self._add_stop(veh, target.platform_edge, target.dwell_min,
                           enabled=not target.pass_flag)
            if target.destination_platform_edge:
                self._add_stop(
                    veh,
                    target.destination_platform_edge,
                    target.destination_dwell_min,
                    enabled=not target.pass_flag and target.destination_dwell_min > 0,
                )

        route_file = self.sumo_dir / "routes" / filename
        ET.indent(ET.ElementTree(root), space="    ")
        ET.ElementTree(root).write(route_file, encoding="UTF-8", xml_declaration=True)
        return route_file

    def run(
        self,
        targets: Dict[str, ExecutionTarget],
        end_time: float = 1200.0,
    ) -> ExecutionResult:
        route_file = self.write_route_file(targets)
        tripinfo_file = self.output_dir / "executor_tripinfo.xml"
        event_csv = self.output_dir / "actual_events.csv"

        traci = _import_traci()
        cmd = [
            "sumo",
            "-n", str(self.cfg.sumo_net_path()),
            "-r", str(route_file),
            "--begin", "0",
            "--end", str(end_time),
            "--no-step-log", "true",
            "--time-to-teleport", "300",
            "--tripinfo-output", str(tripinfo_file),
        ]

        events = _init_events(targets, self.cfg.trains)
        completed: List[str] = []
        traci.start(cmd)
        try:
            while traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                sim_t = traci.simulation.getTime()
                active_ids = set(traci.vehicle.getIDList())
                for tid in active_ids:
                    if tid not in targets:
                        continue
                    edge = traci.vehicle.getRoadID(tid)
                    self._record_edge_events(events[tid], edge, sim_t)
                for tid in traci.simulation.getArrivedIDList():
                    if tid in targets and tid not in completed:
                        completed.append(tid)
        finally:
            traci.close(False)

        self._write_events(event_csv, events)
        return ExecutionResult(
            completed_vehicle_ids=sorted(completed),
            expected_vehicle_ids=sorted(targets.keys()),
            event_csv=event_csv,
            route_file=route_file,
            tripinfo_file=tripinfo_file,
        )

    def _record_edge_events(self, event: Dict[str, object], edge: str, sim_t: float) -> None:
        for prefix, key in [
            ("origin", "origin_edge"),
            ("destination", "destination_edge"),
        ]:
            target_edge = event.get(key)
            if not target_edge:
                continue
            arr_key = f"{prefix}_arr_actual"
            dep_key = f"{prefix}_dep_actual"
            seen_key = f"_{prefix}_seen"
            if edge == target_edge and event[arr_key] is None:
                event[arr_key] = sim_t
                event[seen_key] = True
            elif event.get(seen_key) and event[dep_key] is None and edge != target_edge:
                event[dep_key] = sim_t

    def _write_events(self, path: Path, events: Dict[str, Dict[str, object]]) -> None:
        fields = [
            "train_id", "origin_station", "destination_station",
            "origin_platform", "origin_arr_actual", "origin_dep_actual",
            "origin_dwell_actual", "destination_arr_actual",
            "destination_dep_actual", "destination_dwell_actual",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for e in events.values():
                row = {k: e.get(k) for k in fields}
                row["origin_dwell_actual"] = _duration(
                    row["origin_arr_actual"], row["origin_dep_actual"])
                row["destination_dwell_actual"] = _duration(
                    row["destination_arr_actual"], row["destination_dep_actual"])
                writer.writerow(row)

    def _platform_edge(self, station_id: str, platform: str, direction: str) -> str:
        return self.cfg.platform_edge(station_id, platform, direction)

    def _add_stop(self, vehicle: ET.Element, edge_id: str, duration: float,
                  enabled: bool) -> None:
        if not enabled:
            return
        ET.SubElement(vehicle, "stop", {
            "lane": f"{edge_id}_0",
            "duration": f"{max(0.0, duration):.1f}",
            "startPos": "20",
            "endPos": "230",
        })


def _init_events(
    targets: Dict[str, ExecutionTarget],
    trains: Dict[str, TrainTimetable],
) -> Dict[str, Dict[str, object]]:
    events: Dict[str, Dict[str, object]] = {}
    for tid, target in targets.items():
        train = trains[tid]
        events[tid] = {
            "train_id": tid,
            "origin_station": train.origin_station,
            "destination_station": train.destination_station,
            "origin_platform": target.allocated_platform,
            "origin_edge": target.platform_edge,
            "destination_edge": target.destination_platform_edge,
            "origin_arr_actual": None,
            "origin_dep_actual": None,
            "destination_arr_actual": None,
            "destination_dep_actual": None,
            "_origin_seen": False,
            "_destination_seen": False,
        }
    return events


def _duration(start: object, end: object) -> Optional[float]:
    if start is None or end is None:
        return None
    return float(end) - float(start)


def _import_traci():
    try:
        import traci  # type: ignore
        return traci
    except ModuleNotFoundError:
        sumo_home = os.environ.get("SUMO_HOME")
        if sumo_home:
            tools = Path(sumo_home) / "tools"
            if tools.exists():
                sys.path.append(str(tools))
        import traci  # type: ignore
        return traci
