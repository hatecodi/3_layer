"""Fixed-size observation builder for the two-station baseline."""
from __future__ import annotations

from typing import Dict, List, Set

from ..control.dispatch_bounds import DispatchBounds
from ..control.resource_manager import (
    GrantResult,
    WindowRequest,
    build_conflict_graph,
)
from ..experiment.scenario_loader import ScenarioConfig


OBS_DIM = 74
K_MAX = 5


def build_observations(
    cfg: ScenarioConfig,
    requests: List[WindowRequest],
    grants: List[GrantResult],
    t_now: float = 0.0,
) -> Dict[str, List[float]]:
    layer1 = DispatchBounds(cfg)
    grants_by_train = {g.train_id: g for g in grants}
    request_by_train = {r.train_id: r for r in requests}
    conflict_graph = build_conflict_graph(requests)
    observations: Dict[str, List[float]] = {}

    for train_id, train in cfg.trains.items():
        station_id = train.origin_station
        bounds = layer1.for_train(train_id, station_id)
        req = request_by_train.get(train_id)
        grant = grants_by_train.get(train_id)
        delay = 0.0
        if grant:
            delay = max(0.0, grant.W_out[0] - bounds.t_dep_plan)
        remaining_slack = bounds.t_dep_max - (grant.W_out[0] if grant else t_now)

        self_state = [
            t_now,
            0.0,  # normalized position placeholder
            0.0,  # speed placeholder, filled by live SUMO env later
            delay,
            bounds.t_arr_plan,
            bounds.t_dep_plan,
            bounds.t_arr_min,
            bounds.t_dep_max,
            bounds.dwell_min,
            float(bounds.prio_i),
            1.0 if bounds.pass_flag else 0.0,
            remaining_slack,
            float(len(bounds.candidate_platforms)),
            (grant.W_out[0] - bounds.t_dep_plan) if grant else 0.0,
        ]

        neighbors = _neighbor_block(train_id, requests, conflict_graph)
        resource_state = _resource_state(cfg, grants, t_now)
        obs = self_state + neighbors + resource_state
        if len(obs) != OBS_DIM:
            raise ValueError(f"Observation length for {train_id} is {len(obs)}")
        observations[train_id] = obs

    return observations


def critic_input(
    train_id: str,
    observations: Dict[str, List[float]],
    conflict_graph: Dict[str, set],
    k_max: int = K_MAX,
) -> List[float]:
    group = [train_id] + sorted(conflict_graph.get(train_id, []))[:k_max]
    chunks = [observations.get(tid, [0.0] * OBS_DIM) for tid in group]
    while len(chunks) < k_max + 1:
        chunks.append([0.0] * OBS_DIM)
    return [x for chunk in chunks[:k_max + 1] for x in chunk]


def _neighbor_block(
    train_id: str,
    requests: List[WindowRequest],
    conflict_graph: Dict[str, Set[str]],
) -> List[float]:
    own = next((r for r in requests if r.train_id == train_id), None)
    by_id = {r.train_id: r for r in requests}
    direct = [
        by_id[tid] for tid in sorted(conflict_graph.get(train_id, set()))
        if tid in by_id
    ]
    remaining = [
        r for r in requests
        if r.train_id != train_id and r.train_id not in {d.train_id for d in direct}
    ]
    if own:
        remaining.sort(key=lambda r: abs(r.W_plat[0] - own.W_plat[0]))
    candidates = direct + remaining
    block: List[float] = []
    valid: List[float] = []
    for r in candidates[:K_MAX]:
        block.extend([
            r.W_plat[0],
            r.W_plat[1] - r.W_plat[0],
            r.W_out[0],
            3.0,
            float(r.priority),
            float(r.delay),
            0.0,
            1.0 if own and own.platform == r.platform else 0.0,
        ])
        valid.append(1.0)
    while len(valid) < K_MAX:
        block.extend([0.0] * 8)
        valid.append(0.0)
    return block + valid


def _resource_state(
    cfg: ScenarioConfig,
    grants: List[GrantResult],
    t_now: float,
) -> List[float]:
    occupied: List[float] = []
    for station_id in sorted(cfg.stations.keys()):
        station = cfg.stations[station_id]
        for platform in station.platforms:
            occupied.append(1.0 if any(
                g.platform == platform.id and
                cfg.trains[g.train_id].origin_station == station_id
                and _window_active(g.W_plat, t_now)
                for g in grants
            ) else 0.0)

    conflict_groups = sorted({
        route.conflict_group
        for station in cfg.stations.values()
        for direction in (station.up, station.down)
        for route in direction.in_routes + direction.out_routes
    })
    busy = []
    for group in conflict_groups:
        busy.append(1.0 if any(
            _grant_uses_conflict_group(cfg, grant, group, t_now)
            for grant in grants
        ) else 0.0)

    headway_summary = [0.0]
    state = occupied[:6] + busy[:8] + headway_summary
    while len(state) < 15:
        state.append(0.0)
    return state[:15]


def _window_active(window: tuple[float, float], t_now: float) -> bool:
    return window[0] <= t_now <= window[1]


def _grant_uses_conflict_group(
    cfg: ScenarioConfig,
    grant: GrantResult,
    group: str,
    t_now: float,
) -> bool:
    train = cfg.trains[grant.train_id]
    station = cfg.stations[train.origin_station]
    direction = station.up if train.direction == "up" else station.down
    for route in direction.in_routes:
        if route.platform == grant.platform and route.conflict_group == group:
            return _window_active(grant.W_in, t_now)
    for route in direction.out_routes:
        if route.platform == grant.platform and route.conflict_group == group:
            return _window_active(grant.W_out, t_now)
    return False
