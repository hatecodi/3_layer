"""Compact actor observation builder (~22 dims).

Actor sees only what it needs for decision-making:
  Self (14 dims): time, delay, plan, bounds, priority, class one-hot
  Next station (5 dims): platforms, congestion, est arrival, dwell
  Resource (3 dims): current platforms, corridor delay, parked blocking

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
ACTOR_RES_DIM = 3
ACTOR_OBS_DIM = ACTOR_SELF_DIM + ACTOR_NEXT_DIM + ACTOR_RES_DIM  # 22

TRAIN_CLASS_MAP = {"G": 0, "D": 1, "Z": 2, "K": 3}


# ---------------------------------------------------------------------------
# config-driven dimension computation (kept for backward compatibility)
# ---------------------------------------------------------------------------

def compute_obs_dim(cfg: Optional[ScenarioConfig] = None) -> int:
    return ACTOR_OBS_DIM

_OBS_DIM_CACHE: Optional[int] = None

def get_obs_dim(cfg: Optional[ScenarioConfig] = None) -> int:
    global _OBS_DIM_CACHE
    if cfg is not None:
        _OBS_DIM_CACHE = ACTOR_OBS_DIM
    if _OBS_DIM_CACHE is None:
        return ACTOR_OBS_DIM
    return _OBS_DIM_CACHE


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _station_for(train, station_map):
    return station_map.get(train.id, train.origin_station) if station_map else train.origin_station

def _next_station(train_id, direction, station_map):
    st_list = ["A","B","C","D","E"] if direction == "up" else ["E","D","C","B","A"]
    current = station_map.get(train_id, st_list[0])
    try:
        idx = st_list.index(current)
        return st_list[idx+1] if idx+1 < len(st_list) else None
    except ValueError:
        return None

def _available_platforms(station_id, cfg, parked_windows):
    station = cfg.stations[station_id]
    total = len(station.platforms)
    occupied = sum(1 for p in station.platforms
                   if f"{station_id}/{p.id}" in (parked_windows or {}))
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
            - parked_windows: {resource_id: (start, end)} from env
            - cumulative_delays: {train_id: delay_sec} from TrainRuntimeState
            - corridor_delays: {train_id: extra_sec} from last step
            - journeys: {train_id: {current, next, etc.}} from env
    """
    context = context or {}
    layer1 = DispatchBounds(cfg)
    grants_by_train = {g.train_id: g for g in grants}
    parked_windows = context.get("parked_windows", {})
    cum_delays = context.get("cumulative_delays", {})
    corridor_delays = context.get("corridor_delays", {})
    journeys = context.get("journeys", {})

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

        # ?? Next station info (5 dims) ??
        next_st = _next_station(train_id, train.direction, station_map)
        if next_st and next_st in cfg.stations:
            max_plat_nxt = len(cfg.stations[next_st].platforms)
            avail_nxt = _available_platforms(next_st, cfg, parked_windows)

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

        # ?? Resource + corridor (3 dims) ??
        avail_now = _available_platforms(station_id, cfg, parked_windows)
        max_plat_now = len(cfg.stations[station_id].platforms)
        corr_delay = corridor_delays.get(train_id, 0.0)
        parked_flag = 1.0 if any(
            f"{station_id}/{p.id}" in (parked_windows or {})
            for p in cfg.stations[station_id].platforms
        ) else 0.0

        res_info = [
            avail_now / max(max_plat_now, 1),
            min(corr_delay / 180.0, 1.0) if corr_delay > 0 else 0.0,
            parked_flag,
        ]

        obs = self_state + next_info + res_info
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
    obs_dim = len(next(iter(observations.values()))) if observations else ACTOR_OBS_DIM
    group = [train_id] + sorted(conflict_graph.get(train_id, []))[:k_max]
    chunks = [observations.get(tid, [0.0] * obs_dim) for tid in group]
    while len(chunks) < k_max + 1:
        chunks.append([0.0] * obs_dim)
    return [x for chunk in chunks[:k_max + 1] for x in chunk]
