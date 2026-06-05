"""Config-driven observation builder.

Observation dimension is computed from railway_resources.yaml topology.
Adding/removing stations only requires config changes + model retraining.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from ..control.dispatch_bounds import DispatchBounds
from ..control.resource_manager import (
    GrantResult,
    WindowRequest,
    build_conflict_graph,
)
from ..experiment.scenario_loader import ScenarioConfig


K_MAX = 5
SELF_DIM = 14
NEIGHBOR_DIM = 8          # per neighbor
VALIDITY_DIM = K_MAX       # one flag per neighbor slot


# ---------------------------------------------------------------------------
# config-driven dimension computation
# ---------------------------------------------------------------------------

def _platform_count(cfg: ScenarioConfig) -> int:
    return sum(len(s.platforms) for s in cfg.stations.values())


def _scoped_conflict_groups(cfg: ScenarioConfig) -> List[str]:
    """Station-scoped conflict group IDs — globally unique per topology."""
    groups: Set[str] = set()
    for sid, station in cfg.stations.items():
        for direction in (station.up, station.down):
            for route in direction.in_routes + direction.out_routes:
                groups.add(f"{sid}/{route.conflict_group}")
    return sorted(groups)


def compute_obs_dim(cfg: ScenarioConfig) -> int:
    """Total observation vector length for this topology."""
    resource_dim = _platform_count(cfg) + len(_scoped_conflict_groups(cfg)) + 1
    return SELF_DIM + K_MAX * NEIGHBOR_DIM + VALIDITY_DIM + resource_dim


# Backward-compatible cache — set by load_scenario or env init
_OBS_DIM_CACHE: Optional[int] = None


def get_obs_dim(cfg: Optional[ScenarioConfig] = None) -> int:
    """Return cached OBS_DIM, computing from cfg if provided."""
    global _OBS_DIM_CACHE
    if cfg is not None:
        _OBS_DIM_CACHE = compute_obs_dim(cfg)
    if _OBS_DIM_CACHE is None:
        raise RuntimeError("OBS_DIM not initialized — call get_obs_dim(cfg) first")
    return _OBS_DIM_CACHE


# Keep for backward compat in direct imports
OBS_DIM = 74  # fallback; prefer get_obs_dim(cfg)


# ---------------------------------------------------------------------------
# observation builder
# ---------------------------------------------------------------------------

def _station_for(train, station_map: Dict[str, str] | None) -> str:
    """Return the station a train is currently at, falling back to origin."""
    return station_map.get(train.id, train.origin_station) if station_map else train.origin_station


def build_observations(
    cfg: ScenarioConfig,
    requests: List[WindowRequest],
    grants: List[GrantResult],
    t_now: float = 0.0,
    station_map: Optional[Dict[str, str]] = None,
) -> Dict[str, List[float]]:
    obs_dim = get_obs_dim(cfg)
    layer1 = DispatchBounds(cfg)
    grants_by_train = {g.train_id: g for g in grants}
    request_by_train = {r.train_id: r for r in requests}
    conflict_graph = build_conflict_graph(requests)
    observations: Dict[str, List[float]] = {}

    for train_id, train in cfg.trains.items():
        station_id = _station_for(train, station_map)
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
        resource_state = _resource_state(cfg, grants, t_now, station_map)
        obs = self_state + neighbors + resource_state
        if len(obs) != obs_dim:
            raise ValueError(
                f"Observation length for {train_id} is {len(obs)}, expected {obs_dim}")
        observations[train_id] = obs

    return observations


def critic_input(
    train_id: str,
    observations: Dict[str, List[float]],
    conflict_graph: Dict[str, set],
    k_max: int = K_MAX,
) -> List[float]:
    obs_dim = len(next(iter(observations.values()))) if observations else 0
    group = [train_id] + sorted(conflict_graph.get(train_id, []))[:k_max]
    chunks = [observations.get(tid, [0.0] * obs_dim) for tid in group]
    while len(chunks) < k_max + 1:
        chunks.append([0.0] * obs_dim)
    return [x for chunk in chunks[:k_max + 1] for x in chunk]


# ---------------------------------------------------------------------------
# neighbor block
# ---------------------------------------------------------------------------

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
        in_conflict = 1.0 if r.train_id in conflict_graph.get(train_id, set()) else 0.0
        block.extend([
            r.W_plat[0],
            r.W_plat[1] - r.W_plat[0],
            r.W_out[0],
            float(r.priority),
            in_conflict,
            float(r.delay),
            0.0,
            1.0 if own and own.platform == r.platform else 0.0,
        ])
        valid.append(1.0)
    while len(valid) < K_MAX:
        block.extend([0.0] * NEIGHBOR_DIM)
        valid.append(0.0)
    return block + valid


# ---------------------------------------------------------------------------
# resource state — full topology, no truncation
# ---------------------------------------------------------------------------

def _resource_state(
    cfg: ScenarioConfig,
    grants: List[GrantResult],
    t_now: float,
    station_map: Optional[Dict[str, str]] = None,
) -> List[float]:
    # Platform occupancy — one bit per platform, sorted by station then platform
    platform_bits: List[float] = []
    for station_id in sorted(cfg.stations.keys()):
        station = cfg.stations[station_id]
        for platform in station.platforms:
            busy = any(
                g.platform == platform.id
                and _grant_is_at_station(cfg, g, station_id, station_map)
                and _window_active(g.W_plat, t_now)
                for g in grants
            )
            platform_bits.append(1.0 if busy else 0.0)

    # Conflict group occupancy — one bit per station-scoped group
    scoped_groups = _scoped_conflict_groups(cfg)
    group_bits: List[float] = []
    for scoped_group in scoped_groups:
        # Parse "A/A_up_in_g1" → station_id="A", group="A_up_in_g1"
        sid, group = scoped_group.split("/", 1)
        busy = any(
            _grant_uses_scoped_group(cfg, g, sid, group, t_now, station_map)
            for g in grants
        )
        group_bits.append(1.0 if busy else 0.0)

    # Headway placeholder
    return platform_bits + group_bits + [0.0]


def _grant_is_at_station(
    cfg: ScenarioConfig,
    grant: GrantResult,
    station_id: str,
    station_map: Optional[Dict[str, str]] = None,
) -> bool:
    """Does this grant belong to the given station?"""
    train = cfg.trains[grant.train_id]
    sid = station_map.get(grant.train_id, train.origin_station) if station_map else train.origin_station
    return sid == station_id


def _grant_uses_scoped_group(
    cfg: ScenarioConfig,
    grant: GrantResult,
    station_id: str,
    group: str,
    t_now: float,
    station_map: Optional[Dict[str, str]] = None,
) -> bool:
    """Does this grant occupy a specific conflict group at a specific station?"""
    train = cfg.trains[grant.train_id]
    sid = station_map.get(grant.train_id, train.origin_station) if station_map else train.origin_station
    if sid != station_id:
        return False
    station = cfg.stations[sid]
    direction = station.up if train.direction == "up" else station.down
    for route in direction.in_routes:
        if route.platform == grant.platform and route.conflict_group == group:
            return _window_active(grant.W_in, t_now)
    for route in direction.out_routes:
        if route.platform == grant.platform and route.conflict_group == group:
            return _window_active(grant.W_out, t_now)
    return False


def _window_active(window: tuple[float, float], t_now: float) -> bool:
    return window[0] <= t_now <= window[1]
