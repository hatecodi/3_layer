"""Compact actor observation builder (config-driven dimension).

Actor sees only what it needs for decision-making:
  Self (14 dims): time, delay, plan, bounds, priority, class one-hot
  Next station (5 dims): platforms, congestion, est arrival, dwell
  Resource (2 + max_platforms dims): avail, corridor, per-platform blocked flags

Critic gets broader context via critic_input().
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from ..control.dispatch_bounds import DispatchBounds
from ..control.resource_manager import GrantResult, WindowRequest, build_conflict_graph
from ..experiment.scenario_loader import ScenarioConfig

K_MAX = 5
MAX_EPISODE_TIME = 1500.0

ACTOR_SELF_DIM = 14
ACTOR_NEXT_DIM = 5
ACTOR_COMPETITOR_DIM = 1  # n_competitors at current station
_ACTOR_FIXED_RES_DIM = 2  # avail_now, corridor_delay (per-platform bits added dynamically)

TRAIN_CLASS_MAP = {"G": 0, "D": 1, "Z": 2, "K": 3}


# ---------------------------------------------------------------------------
# config-driven dimension computation
# ---------------------------------------------------------------------------

def _max_platforms(cfg: Optional[ScenarioConfig]) -> int:
    """Maximum number of platforms across all stations."""
    if cfg is None:
        return 4
    return max((len(s.platforms) for s in cfg.stations.values()), default=4)


def compute_obs_dim(cfg: Optional[ScenarioConfig] = None) -> int:
    """Compute observation dimension from config.

    OBS_DIM = SELF(14) + NEXT(5) + COMPETITOR(1) + avail(1) + corridor(1) + per_platform_blocked(max_platforms)
    """
    max_plat = _max_platforms(cfg)
    return ACTOR_SELF_DIM + ACTOR_NEXT_DIM + ACTOR_COMPETITOR_DIM + _ACTOR_FIXED_RES_DIM + max_plat


_OBS_DIM_CACHE: Optional[int] = None

def get_obs_dim(cfg: Optional[ScenarioConfig] = None) -> int:
    global _OBS_DIM_CACHE
    if cfg is not None:
        _OBS_DIM_CACHE = compute_obs_dim(cfg)
    if _OBS_DIM_CACHE is None:
        return compute_obs_dim(cfg)
    return _OBS_DIM_CACHE


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _station_for(train, station_map):
    return station_map.get(train.id, train.origin_station) if station_map else train.origin_station

def _next_station(train_id, direction, station_map):
    st_list = ["A","B","C","D","E"] if direction == "up" else ["E","D","C","B","A"]
    if station_map:
        current = station_map.get(train_id, st_list[0])
    else:
        current = st_list[0]
    try:
        idx = st_list.index(current)
        return st_list[idx+1] if idx+1 < len(st_list) else None
    except ValueError:
        return None

def _available_platforms(station_id, cfg, visitor_windows):
    station = cfg.stations[station_id]
    total = len(station.platforms)
    occupied = sum(1 for p in station.platforms
                   if f"{station_id}/{p.id}" in (visitor_windows or {}))
    return total - occupied

def _one_hot(cls_val):
    idx = TRAIN_CLASS_MAP.get(cls_val, 0)
    return [1.0 if i == idx else 0.0 for i in range(4)]


# ---------------------------------------------------------------------------
# main observation builder
# ---------------------------------------------------------------------------

def build_observations(
    cfg: ScenarioConfig,
    requests: List[WindowRequest],
    grants: List[GrantResult],
    t_now: float = 0.0,
    station_map: Optional[Dict[str, str]] = None,
    context: Optional[Dict] = None,
) -> Dict[str, List[float]]:
    """Build compact actor observations (~22 dims).

    Args:
        context: optional dict with keys:
            - visitor_windows: {resource_id: (start, end)} visitor pre-bookings
            - visitors: {station_id: [(platform, start, end, priority), ...]} visitor notification
            - cumulative_delays: {train_id: delay_sec} from TrainRuntimeState
            - corridor_delays: {train_id: extra_sec} from last step
            - journeys: {train_id: {current, next, etc.}} from env
    """
    context = context or {}
    layer1 = DispatchBounds(cfg)
    grants_by_train = {g.train_id: g for g in grants}
    visitor_windows = context.get("visitor_windows", {})
    visitors = context.get("visitors", {})  # {station_id: [(platform, start, end, priority), ...]}
    cum_delays = context.get("cumulative_delays", {})
    corridor_delays = context.get("corridor_delays", {})
    journeys = context.get("journeys", {})
    max_plat = _max_platforms(cfg)

    observations: Dict[str, List[float]] = {}

    for train_id, train in cfg.trains.items():
        station_id = _station_for(train, station_map)
        bounds = layer1.for_train(train_id, station_id)
        grant = grants_by_train.get(train_id)

        # ?? Self state (14 dims) ??
        cum_delay = cum_delays.get(train_id, 0.0)
        slack = bounds.t_dep_max - max(grant.W_out[0] if grant else t_now, bounds.t_dep_plan)
        slack = max(0.0, min(slack, 300.0))

        self_state = [
            min(t_now / MAX_EPISODE_TIME, 1.0),
            min(cum_delay / 600.0, 1.0),
            bounds.t_arr_plan / MAX_EPISODE_TIME,
            bounds.t_dep_plan / MAX_EPISODE_TIME,
            bounds.t_arr_min / MAX_EPISODE_TIME,
            bounds.t_dep_max / MAX_EPISODE_TIME,
            bounds.dwell_min / 120.0,
            bounds.prio_i / 3.0,
            float(bounds.pass_flag),
            slack / 300.0,
        ] + _one_hot(train.train_class)

        # ?? Competitors at current station (1 dim) ??
        pending_at_station = context.get("pending_at_station", {})
        n_at_station = pending_at_station.get(train_id, 0)
        competitor_info = [min(n_at_station / 4.0, 1.0)]  # normalized, cap at 4+

        # ?? Next station info (5 dims) ??
        next_st = _next_station(train_id, train.direction, station_map)
        if next_st and next_st in cfg.stations:
            max_plat_nxt = len(cfg.stations[next_st].platforms)
            avail_nxt = _available_platforms(next_st, cfg, visitor_windows)

            n_heading = 0
            for tid2 in cfg.trains:
                if tid2 == train_id:
                    continue
                nxt2 = _next_station(tid2, cfg.trains[tid2].direction, station_map)
                if nxt2 == next_st:
                    n_heading += 1

            congestion = min(n_heading / max(max_plat_nxt, 1), 1.0)
            est_arrival = (grant.W_out[0] if grant else bounds.t_dep_plan) + 90.0
            nxt_bounds = layer1.for_train(train_id, next_st)

            next_info = [
                avail_nxt / max(max_plat_nxt, 1),
                n_heading / max(len(cfg.trains), 1),
                min(est_arrival / MAX_EPISODE_TIME, 1.0),
                congestion,
                (nxt_bounds.dwell_min / 120.0) if nxt_bounds else 0.0,
            ]
        else:
            next_info = [0.0] * ACTOR_NEXT_DIM

        # ?? Resource (2 + max_platforms dims) ??
        avail_now = _available_platforms(station_id, cfg, visitor_windows)
        max_plat_now = len(cfg.stations[station_id].platforms)
        corr_delay = corridor_delays.get(train_id, 0.0)

        # Per-platform blocked flags from visitor advance notification
        station_visitors = visitors.get(station_id, [])
        platform_blocked = []
        for p in cfg.stations[station_id].platforms:
            blocked = any(
                v_start <= t_now + 600 and v_end >= t_now
                for v_plat, v_start, v_end, _ in station_visitors
                if v_plat == p.id
            )
            platform_blocked.append(1.0 if blocked else 0.0)

        # Pad to max_plat
        while len(platform_blocked) < max_plat:
            platform_blocked.append(0.0)

        res_info = [
            avail_now / max(max_plat_now, 1),
            min(corr_delay / 180.0, 1.0) if corr_delay > 0 else 0.0,
        ] + platform_blocked

        obs = self_state + competitor_info + next_info + res_info
        observations[train_id] = obs

    return observations


# ---------------------------------------------------------------------------
# critic input (broader context for value estimation)
# ---------------------------------------------------------------------------

def critic_input(
    train_id: str,
    observations: Dict[str, List[float]],
    conflict_graph: Dict[str, set],
    k_max: int = K_MAX,
) -> List[float]:
    """Build critic input: self obs + k_max neighbor obs + padding."""
    obs_dim = len(next(iter(observations.values()))) if observations else compute_obs_dim()
    group = [train_id] + sorted(conflict_graph.get(train_id, []))[:k_max]
    chunks = [observations.get(tid, [0.0] * obs_dim) for tid in group]
    while len(chunks) < k_max + 1:
        chunks.append([0.0] * obs_dim)
    return [x for chunk in chunks[:k_max + 1] for x in chunk]
