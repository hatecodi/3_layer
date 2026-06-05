#!/usr/bin/env python3
"""Heuristic baseline for multi-station scheduling."""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT))

from src.env.five_station_env import FiveStationEnv
from src.env.disturbance import CorridorDisturbance

import yaml

def load_multistation_cfg():
    """Load 5-station config. Reuses scenario_loader for structure but
    overrides with multistation YAML files."""
    # For now, handle requires scenario_loader adaptation.
    # Direct YAML loading with ScenarioConfig construction.
    from src.experiment.scenario_loader import (
        ScenarioConfig, StationConfig, PlatformConfig, CorridorConfig,
        TrainTimetable, TrainPlan, TrainBound, DirectionConfig, ThroatRoute,
        ValidationError,
    )

    project_root = Path(__file__).resolve().parent.parent.parent
    config_dir = project_root / "configs" / "five_station"
    with open(config_dir / "railway_resources.yaml", encoding="utf-8") as f:
        rsrc = yaml.safe_load(f)
    with open(config_dir / "timetable.yaml", encoding="utf-8") as f:
        tt = yaml.safe_load(f)

    # Parse stations
    stations = {}
    for s in rsrc["stations"]:
        platforms = [
            PlatformConfig(
                id=p["id"], up_edge=p["up_edge"], down_edge=p["down_edge"],
                through_allowed=p.get("through_allowed", True))
            for p in s["platforms"]
        ]
        dwell_stop = s.get("dwell_min_sec", {}).get("stop", 60)

        def _dir(d):
            return DirectionConfig(
                approach_edge=d.get("approach_edge"),
                depart_edge=d.get("depart_edge"),
                in_routes=[ThroatRoute(**r) for r in d.get("in_routes", [])],
                out_routes=[ThroatRoute(**r) for r in d.get("out_routes", [])],
            )

        stations[s["id"]] = StationConfig(
            id=s["id"], platforms=platforms,
            up=_dir(s.get("up", {})), down=_dir(s.get("down", {})),
            dwell_min_sec_stop=dwell_stop,
        )

    # Parse corridors
    corridors = {}
    for c in rsrc["corridors"]:
        corridors[c["id"]] = CorridorConfig(
            id=c["id"], direction=c["direction"],
            from_station=c["from_station"], to_station=c["to_station"],
            edges=c.get("edges", []), min_headway_sec=c.get("min_headway_sec", 120),
        )

    # Parse trains
    trains = {}
    for t in tt["trains"]:
        planned = {sid: TrainPlan(arr=v["arr"], dep=v["dep"])
                   for sid, v in t.get("planned", {}).items()}
        bounds = {sid: TrainBound(arr_min=v["arr_min"], dep_max=v["dep_max"],
                                 dwell_min=v["dwell_min"])
                  for sid, v in t.get("bounds", {}).items()}
        route = t.get("route", {})
        trains[t["id"]] = TrainTimetable(
            id=t["id"], direction=t["direction"],
            train_class=t["train_class"], priority=t.get("priority", 99),
            pass_through=t.get("pass_through", False),
            origin_station=route.get("origin_station", ""),
            destination_station=route.get("destination_station", ""),
            corridor=route.get("corridor", ""),
            planned=planned, bounds=bounds,
            candidate_platforms=t.get("candidate_platforms", {}),
        )

    return ScenarioConfig(
        project_root=project_root,
        network=rsrc.get("network", {}),
        stations=stations, corridors=corridors, trains=trains,
        sumo_routes={}, sumo_vehicles={}, sumo_vtypes={},
        priority_map=tt.get("priority_map", {}),
        time_generation=tt.get("time_generation", {}),
    )


def main():
    cfg = load_multistation_cfg()
    print(f"Stations: {list(cfg.stations.keys())}")
    print(f"Trains: {list(cfg.trains.keys())}")

    # Corridor disturbance on B_C_up (the longest corridor)
    corridor_d = CorridorDisturbance(
        corridor_id="A_B_up", start_time=200, end_time=600, speed_ratio=0.5)
    env = FiveStationEnv(
        cfg, disturbance_prob=0.0,
        corridor_disturbances=[corridor_d],
        initial_train_delays={"K999": 60.0},
    )

    obs, info = env.reset()
    print(f"Pending: {len(info['pending'])} trains")
    print(f"Stations: {[(tid, env._journeys[tid].current_station) for tid in sorted(info['pending'])]}")

    step = 0
    while True:
        obs, reward, terminated, truncated, info = env.step({})
        step += 1
        grants = info.get("granted_windows", [])
        held = info.get("held_train_ids", [])
        print(f"\nStep {step} (t={info['t_now']:.0f}): {len(grants)} grants, {len(held)} held")
        print(f"  Status: {info['episode_status']}")
        for g in grants:
            j = env._journeys[g.train_id]
            st = j.current_station
            nxt = j.next_station
            print(f"  {g.train_id} @ {st}: {g.platform} winner={g.winner} "
                  f"W_in={g.W_in[0]:.0f} W_out={g.W_out[0]:.0f} "
                  f"→ next: {nxt or 'DONE'}")
        if held:
            print(f"  HELD: {held}")
        if terminated or truncated:
            break

    m = info.get("metrics", {})
    print(f"\nEpisode done: {info['episode_status']}")
    print(f"Completed: {info['completed']}")
    print(f"Steps: {step}")
    print(f"mean_dep_delay={m.get('mean_dep_delay', 0):.1f}s")
    print(f"mean_arr_delay={m.get('mean_arr_delay', 0):.1f}s")


if __name__ == "__main__":
    main()
