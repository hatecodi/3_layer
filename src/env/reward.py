"""Reward and metric helpers for the Layer 2 learning environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set

from ..control.dispatch_bounds import DispatchBounds
from ..control.resource_manager import GrantResult
from ..experiment.scenario_loader import ScenarioConfig


@dataclass
class Metrics:
    n_granted: int
    n_held: int
    n_active: int
    total_delay: float
    mean_delay: float
    max_delay: float
    unresolved_conflicts: int
    deadlock_flag: int
    completion_rate: float
    hard_bound_infeasible_count: int


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
) -> Metrics:
    layer1 = DispatchBounds(cfg)
    held: Set[str] = set(held_train_ids)
    active: Set[str] = set(active_train_ids or cfg.trains.keys())

    delays: List[float] = []
    for grant in grants:
        train = cfg.trains[grant.train_id]
        bounds = layer1.for_train(grant.train_id, train.origin_station)
        delays.append(max(0.0, grant.W_out[0] - bounds.t_dep_plan))

    total_delay = sum(delays)
    n_active = max(len(active), 1)
    return Metrics(
        n_granted=len(grants),
        n_held=len(held),
        n_active=len(active),
        total_delay=total_delay,
        mean_delay=total_delay / max(len(grants), 1),
        max_delay=max(delays) if delays else 0.0,
        unresolved_conflicts=unresolved_conflicts,
        deadlock_flag=1 if len(held) >= n_active else 0,
        completion_rate=len(grants) / n_active,
        hard_bound_infeasible_count=hard_bound_infeasible_count,
    )


def compute_layered_rewards(
    cfg: ScenarioConfig,
    grants: List[GrantResult],
    held_train_ids: Iterable[str],
    active_train_ids: Iterable[str] | None = None,
    unresolved_conflicts: int = 0,
    hard_bound_infeasible_count: int = 0,
    lambda_team: float = 0.3,
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
    )

    global_reward = (
        -0.02 * (metrics.total_delay / max(metrics.n_active, 1))
        -2.0 * metrics.deadlock_flag
        -0.5 * unresolved_conflicts
        -1.0 * hard_bound_infeasible_count
    )

    rewards: Dict[str, RewardBreakdown] = {}
    for train_id in active:
        train = cfg.trains[train_id]
        bounds = layer1.for_train(train_id, train.origin_station)
        grant = grant_by_train.get(train_id)
        if train_id in held or grant is None:
            local = -1.0
        else:
            delay = max(0.0, grant.W_out[0] - bounds.t_dep_plan)
            prio_weight = {1: 2.0, 2: 1.0, 3: 0.5}.get(bounds.prio_i, 1.0)
            wait = max(0.0, grant.W_in[0] - bounds.t_arr_plan)
            dwell = max(0.0, grant.W_out[0] - grant.W_plat[0])
            excessive_hold = max(0.0, dwell - bounds.dwell_min - 60.0)
            slack_span = max(bounds.t_dep_max - bounds.t_arr_min, 1.0)
            bound_slack = (bounds.t_dep_max - grant.W_out[0]) / slack_span
            local = (
                5.0
                -0.01 * wait
                -0.02 * delay * prio_weight
                -0.01 * excessive_hold
                +0.02 * bound_slack
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
) -> Dict[str, float]:
    return {
        tid: reward.total
        for tid, reward in compute_layered_rewards(
            cfg, grants, held_train_ids, active_train_ids, lambda_team=lambda_team
        ).items()
    }
