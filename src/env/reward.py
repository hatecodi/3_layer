"""Reward and metric helpers for the Layer 2 learning environment."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

from ..control.dispatch_bounds import DispatchBounds
from ..control.resource_manager import GrantResult
from ..experiment.scenario_loader import ScenarioConfig


ONTIME_THRESHOLD_SEC = 30.0   # within 30s of plan = on-time
ONTIME_BONUS = 2.0            # per-train local reward bonus


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
        dep_delays.append(max(0.0, grant.W_out[0] - bounds.t_dep_plan))
        arr_delays.append(max(0.0, grant.W_in[0] - bounds.t_arr_plan))

    total_dep = sum(dep_delays)
    total_arr = sum(arr_delays)
    n_granted = len(grants)
    n_active = max(len(active), 1)
    n_ontime_dep = sum(1 for d in dep_delays if d <= ONTIME_THRESHOLD_SEC)
    n_ontime_arr = sum(1 for d in arr_delays if d <= ONTIME_THRESHOLD_SEC)
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
) -> Dict[str, RewardBreakdown]:
    layer1 = DispatchBounds(cfg)
    held = set(held_train_ids)
    active = set(active_train_ids or cfg.trains.keys())
    grant_by_train = {g.train_id: g for g in grants}
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
        -0.01 * (metrics.total_arr_delay / max(metrics.n_active, 1))
        -20.0 * metrics.deadlock_flag
        -0.5 * unresolved_conflicts
        -1.0 * hard_bound_infeasible_count
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
            dep_delay = max(0.0, grant.W_out[0] - bounds.t_dep_plan)
            arr_delay = max(0.0, grant.W_in[0] - bounds.t_arr_plan)
            prio_weight = {1: 2.0, 2: 1.0, 3: 0.5}.get(bounds.prio_i, 1.0)
            wait = max(0.0, grant.W_in[0] - bounds.t_arr_plan)
            dwell = max(0.0, grant.W_out[0] - grant.W_plat[0])
            excessive_hold = max(0.0, dwell - bounds.dwell_min - 60.0)
            slack_span = max(bounds.t_dep_max - bounds.t_arr_min, 1.0)
            bound_slack = (bounds.t_dep_max - grant.W_out[0]) / slack_span
            # Continuous improvement bonus: reward ANY delay reduction, not just
            # binary on-time (<30s).  Uses exponential decay so every second of
            # improvement gives measurable gradient, even for heavily delayed trains.
            dep_improve = math.exp(-dep_delay / 60.0)
            arr_improve = math.exp(-arr_delay / 60.0)
            # Progressive penalty: every 300s of delay doubles the marginal cost
            dep_mult = 1.0 + dep_delay // 300.0
            arr_mult = 1.0 + arr_delay // 300.0
            local = (
                2.0
                -0.01 * wait
                -0.02 * dep_delay * prio_weight * dep_mult
                -0.02 * arr_delay * prio_weight * arr_mult
                -0.01 * excessive_hold
                +0.05 * bound_slack
                +ONTIME_BONUS * dep_improve * prio_weight
                +ONTIME_BONUS * arr_improve * prio_weight
            )
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
