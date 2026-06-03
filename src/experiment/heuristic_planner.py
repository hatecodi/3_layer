"""Reusable heuristic planner for Phase 2+ baselines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..control.action_mask import action_to_request, heuristic_action, is_legal_action
from ..control.dispatch_bounds import DispatchBounds
from ..control.resource_manager import GrantResult, RebuildFn, WindowRequest, resolve_all
from .scenario_loader import ScenarioConfig


@dataclass
class HeuristicPlan:
    grants: List[GrantResult]
    held_train_ids: List[str]
    requests: List[WindowRequest]
    violations: Dict[str, int]


def make_rebuild_fn(
    layer1: DispatchBounds,
    station_id: str,
) -> RebuildFn:
    """Build shifted requests with fresh resource IDs for a new platform."""

    def rebuild(train_id: str, platform: str, shift: float,
                base_req: WindowRequest) -> Optional[WindowRequest]:
        train = layer1.trains[train_id]
        bounds = layer1.for_train(train_id, station_id)
        w_in = (base_req.W_in[0] + shift, base_req.W_in[1] + shift)
        w_plat = (base_req.W_plat[0] + shift, base_req.W_plat[1] + shift)
        w_out = (base_req.W_out[0] + shift, base_req.W_out[1] + shift)

        if w_in[0] < bounds.t_arr_min - 1e-6:
            return None
        if w_out[0] > bounds.t_dep_max + 1e-6:
            return None

        station = layer1._cfg.stations[station_id]
        dir_cfg = station.up if train.direction == "up" else station.down
        in_route = next((r for r in dir_cfg.in_routes
                         if r.platform == platform), None)
        out_route = next((r for r in dir_cfg.out_routes
                          if r.platform == platform), None)

        req = WindowRequest(
            train_id=train_id,
            platform=platform,
            W_in=w_in,
            W_plat=w_plat,
            W_out=w_out,
            W_corridor=(base_req.W_corridor[0] + shift,
                        base_req.W_corridor[1] + shift)
            if base_req.W_corridor else None,
            priority=base_req.priority,
            delay=base_req.delay + shift,
        )
        req._resource_windows = {f"{station_id}/{platform}": w_plat}
        if in_route:
            req._resource_windows[f"{station_id}/{in_route.conflict_group}"] = w_in
        if out_route:
            req._resource_windows[f"{station_id}/{out_route.conflict_group}"] = w_out
        return req

    return rebuild


def schedule_station(
    layer1: DispatchBounds,
    station_id: str,
    train_ids: List[str],
    t_now: float,
    travel_time_to_throat: float,
    cfg: ScenarioConfig,
) -> Tuple[List[WindowRequest], Dict[str, int]]:
    requests: List[WindowRequest] = []
    violations: Dict[str, int] = {}

    for tid in sorted(train_ids,
                      key=lambda t: layer1.for_train(t, station_id).t_dep_plan):
        train = layer1.trains[tid]
        bounds = layer1.for_train(tid, station_id)
        try:
            action = heuristic_action(
                train, station_id, t_now, travel_time_to_throat, cfg)
        except ValueError:
            violations[tid] = 1
            continue

        if not is_legal_action(
            train, station_id, action, t_now, travel_time_to_throat, cfg):
            violations[tid] = 2
            continue

        req = action_to_request(
            train, station_id, action, t_now, travel_time_to_throat,
            current_delay=max(0.0, t_now - bounds.t_arr_plan),
            cfg=cfg,
        )
        requests.append(req)
        violations[tid] = 0

    return requests, violations


def build_heuristic_plan(
    cfg: ScenarioConfig,
    t_now: float = 0.0,
    travel_time_to_throat: float = 30.0,
) -> HeuristicPlan:
    layer1 = DispatchBounds(cfg)

    all_requests: List[WindowRequest] = []
    all_violations: Dict[str, int] = {}

    train_ids_by_station: Dict[str, List[str]] = {}
    for tid in layer1.train_ids:
        station_id = layer1.trains[tid].origin_station
        train_ids_by_station.setdefault(station_id, []).append(tid)

    requests_by_station: Dict[str, List[WindowRequest]] = {}
    for station_id, train_ids in train_ids_by_station.items():
        station_reqs, station_violations = schedule_station(
            layer1, station_id, train_ids, t_now, travel_time_to_throat, cfg)
        requests_by_station[station_id] = station_reqs
        all_requests.extend(station_reqs)
        all_violations.update(station_violations)

    candidate_platforms: Dict[str, List[str]] = {
        tid: layer1.for_train(
            tid, layer1.trains[tid].origin_station).candidate_platforms
        for tid in layer1.train_ids
    }
    grants = []
    for station_id, station_reqs in requests_by_station.items():
        grants.extend(resolve_all(
            station_reqs, candidate_platforms, make_rebuild_fn(layer1, station_id)))

    granted_ids = {g.train_id for g in grants}
    held = [r.train_id for r in all_requests if r.train_id not in granted_ids]
    return HeuristicPlan(grants, held, all_requests, all_violations)
