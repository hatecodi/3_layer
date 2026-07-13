"""Reward and metric helpers for the Layer 2 learning environment."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from ..control.dispatch_bounds import DispatchBounds
from ..control.resource_manager import GrantResult
from ..experiment.scenario_loader import ScenarioConfig


ONTIME_THRESHOLD_SEC = 15.0   # within 15s of plan = on-time (both early and late)
ONTIME_BONUS = 2.0            # per-train local reward bonus
AVOIDANCE_BONUS = 2.0         # bonus for proactively choosing non-p1 when p1 is contested


@dataclass
class Metrics:
    n_granted: int
    n_held: int
    n_active: int
    total_dep_delay: float
    total_arr_delay: float
    mean_dep_delay: float
    mean_arr_delay: float
    max_dep_delay: float
    max_arr_delay: float
    unresolved_conflicts: int
    deadlock_flag: int
    completion_rate: float
    hard_bound_infeasible_count: int
    n_ontime_dep: int = 0          # trains departing within threshold
    n_ontime_arr: int = 0          # trains arriving within threshold
    ontime_rate: float = 0.0       # ontine departures / n_granted
    platform_utilization: float = 0.0  # distinct platforms used / total platforms (per station avg)


@dataclass
class RewardBreakdown:
    local: float
    global_: float
    total: float


def compute_metrics(
    cfg: ScenarioConfig,
    grants: List[GrantResult],
    held_train_ids: Iterable[str],
    active_train_ids: Iterable[str] | None = None,
    unresolved_conflicts: int = 0,
    hard_bound_infeasible_count: int = 0,
    station_map: Optional[Dict[str, str]] = None,
) -> Metrics:
    """Compute scheduling metrics.

    If station_map is provided (train_id → current station), delays are
    computed relative to that station's timetable.  Otherwise, each train's
    origin_station is used (backward-compatible for two-station env).
    """
    layer1 = DispatchBounds(cfg)
    held: Set[str] = set(held_train_ids)
    active: Set[str] = set(active_train_ids or cfg.trains.keys())

    dep_delays: List[float] = []
    arr_delays: List[float] = []
    for grant in grants:
        train = cfg.trains[grant.train_id]
        sid = station_map.get(grant.train_id) if station_map else train.origin_station
        bounds = layer1.for_train(grant.train_id, sid)
        dep_delays.append(abs(grant.W_out[0] - bounds.t_dep_plan))
        arr_delays.append(abs(grant.W_in[0] - bounds.t_arr_plan))

    total_dep = sum(dep_delays)
    total_arr = sum(arr_delays)
    n_granted = len(grants)
    n_active = max(len(active), 1)
    n_ontime_dep = sum(1 for d in dep_delays if d <= ONTIME_THRESHOLD_SEC)
    n_ontime_arr = sum(1 for d in arr_delays if d <= ONTIME_THRESHOLD_SEC)

    # Platform utilization: per-station average of (distinct_platforms_used / total_platforms)
    plat_by_station: Dict[str, set] = {}
    for grant in grants:
        sid = station_map.get(grant.train_id) if station_map else "?"
        plat_by_station.setdefault(sid, set()).add(grant.platform)
    util_sum = 0.0
    n_stations = 0
    for sid, used in plat_by_station.items():
        if sid and sid in cfg.stations:
            total = len(cfg.stations[sid].platforms)
            if total > 0:
                util_sum += len(used) / total
                n_stations += 1
    platform_util = util_sum / max(n_stations, 1)
    return Metrics(
        n_granted=n_granted,
        n_held=len(held),
        n_active=len(active),
        total_dep_delay=total_dep,
        total_arr_delay=total_arr,
        mean_dep_delay=total_dep / max(n_granted, 1),
        mean_arr_delay=total_arr / max(n_granted, 1),
        max_dep_delay=max(dep_delays) if dep_delays else 0.0,
        max_arr_delay=max(arr_delays) if arr_delays else 0.0,
        unresolved_conflicts=unresolved_conflicts,
        deadlock_flag=1 if len(held) >= n_active else 0,
        completion_rate=n_granted / n_active,
        hard_bound_infeasible_count=hard_bound_infeasible_count,
        n_ontime_dep=n_ontime_dep,
        n_ontime_arr=n_ontime_arr,
        ontime_rate=n_ontime_dep / max(n_granted, 1),
        platform_utilization=platform_util,
    )


def compute_layered_rewards(
    cfg: ScenarioConfig,
    grants: List[GrantResult],
    held_train_ids: Iterable[str],
    active_train_ids: Iterable[str] | None = None,
    unresolved_conflicts: int = 0,
    hard_bound_infeasible_count: int = 0,
    lambda_team: float = 0.3,
    station_map: Optional[Dict[str, str]] = None,
    visitor_windows: Optional[Dict[str, Tuple[float, float]]] = None,
    retried_train_ids: Optional[Iterable[str]] = None,
    corridor_delays: Optional[Dict[str, float]] = None,
) -> Dict[str, RewardBreakdown]:
    layer1 = DispatchBounds(cfg)
    held = set(held_train_ids)
    active = set(active_train_ids or cfg.trains.keys())
    grant_by_train = {g.train_id: g for g in grants}
    retried = set(retried_train_ids or [])
    visitor_windows = visitor_windows or {}
    corridor_delays = corridor_delays or {}
    metrics = compute_metrics(
        cfg,
        grants,
        held,
        active,
        unresolved_conflicts=unresolved_conflicts,
        hard_bound_infeasible_count=hard_bound_infeasible_count,
        station_map=station_map,
    )

    global_reward = (
        -0.02 * (metrics.total_dep_delay / max(metrics.n_active, 1))
        -20.0 * metrics.deadlock_flag
        +2.0 * metrics.ontime_rate
    )

    rewards: Dict[str, RewardBreakdown] = {}
    for train_id in active:
        train = cfg.trains[train_id]
        sid = station_map.get(train_id) if station_map else train.origin_station
        bounds = layer1.for_train(train_id, sid)
        grant = grant_by_train.get(train_id)
        if train_id in held or grant is None:
            local = -3.0
        else:
            dep_delay = abs(grant.W_out[0] - bounds.t_dep_plan)
            prio_weight = {1: 2.0, 2: 1.0, 3: 0.5}.get(bounds.prio_i, 1.0)
            # Progressive penalty: every 300s of delay doubles the marginal cost
            dep_mult = 1.0 + dep_delay // 300.0
            # Continuous improvement: exponential decay so every second of
            # delay reduction gives measurable gradient
            dep_improve = math.exp(-dep_delay / 60.0)
            local = (
                -0.02 * dep_delay * prio_weight * dep_mult
                + ONTIME_BONUS * dep_improve * prio_weight
            )
            # Early/late arrival penalty (intermediate stations only).
            # L3 decomposition can make transit faster than planned; penalising
            # arrival deviation gives MAPPO incentive to adjust departure timing
            # to match the physical corridor transit time.
            if sid != train.origin_station:
                arr_delay = abs(grant.W_in[0] - bounds.t_arr_plan)
                local += -0.01 * arr_delay * prio_weight
            # Corridor headway penalty: penalise trains that experienced
            # headway delay, giving L2 incentive to space departures.
            hw_delay = corridor_delays.get(train_id, 0.0)
            if hw_delay > 1e-6:
                local += -0.02 * hw_delay * prio_weight
        total = local + lambda_team * global_reward
        rewards[train_id] = RewardBreakdown(local, global_reward, total)
    return rewards


def compute_rewards(
    cfg: ScenarioConfig,
    grants: List[GrantResult],
    held_train_ids: Iterable[str],
    active_train_ids: Iterable[str] | None = None,
    lambda_team: float = 0.3,
    station_map: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    return {
        tid: reward.total
        for tid, reward in compute_layered_rewards(
            cfg, grants, held_train_ids, active_train_ids,
            lambda_team=lambda_team, station_map=station_map,
        ).items()
    }


# ---------------------------------------------------------------------------
# L3 Worker reward (corridor-level objectives)
# ---------------------------------------------------------------------------


def compute_l3_reward(
    headway_delay_sec: float,
    arrival_deviation_sec: float,
    recovery_used_sec: float,
) -> float:
    """Compute L3 Worker reward for one corridor transit.

    Objectives:
      - Minimise headway waiting time
      - Minimise arrival time deviation (early or late)
      - Reward using recovery margin only when it helps (net time saved)

    All terms are in seconds, scaled to comparable magnitudes.
    """
    return (
        -0.02 * headway_delay_sec
        - 0.01 * abs(arrival_deviation_sec)
        + 0.01 * recovery_used_sec
    )
