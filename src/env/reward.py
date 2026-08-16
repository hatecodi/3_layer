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
# Hard ontime trigger aligned with ONTIME_THRESHOLD_SEC.  The exp-decay bonus
# below rewards "less delay" continuously, so a policy can shrink 40->20s delay
# (reward improves) without ever crossing the 15s punctuality window (ontime rate
# flat) — training optimises delay reduction, not punctuality.  This hard bonus
# makes the eval threshold the explicit objective so the reward tracks ontime.
ONTIME_HARD_BONUS = 2.0
AVOIDANCE_BONUS = 2.0         # bonus for proactively choosing non-p1 when p1 is contested
# Per-section energy cost of the FAST mode (per block section).  Without it FAST
# is free and the policy collapses to "always FAST under any stress" — no
# intelligent speed choice.  With it, FAST is used only when recovering delay is
# worth the energy.  0.25/section was calibrated: 0.5 over-corrected (policy gave
# up on FAST entirely, ontime stuck ~58%, entropy collapsed to 0).
# 0.35 (2026-08-14): the always-FAST collapse (84% FAST, 0% CRUISE) in the D-scarcity
# scenario needs a slightly heavier FAST cost than 0.25, but below the 0.5 cliff.
# 0.30 (2026-08-14, v3): 0.35 + always-early over-corrected (avoided FAST on restricted
# sections -> lost delay recovery, affected 98s vs SA 30s).  Gentler FAST cost with the
# always-early penalty (0.05) should keep recovery while stopping the overshoot.
L3_FAST_PENALTY = 0.30

# Punctuality demand by train priority (railway rule: higher-class more on-time).
# Priority 1 = 复兴号 (highest), 2 = 和谐号, 3 = 普速 (lowest).
PRIORITY_WEIGHT = {1: 1.4, 2: 1.0, 3: 0.6}
# Cap the late-penalty magnitude so a few extreme-late (physics-bound) trains don't
# dwarf the ontime-rate incentive (railway KPI = rate primary).  The superlinear
# penalty still applies up to ~85s; beyond that it's flat, keeping the tail
# SECONDARY to the ontime rate.
L3_LATE_CAP = 20.0


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
    n_ontime_dep: int = 0          # trains departing within threshold (L2 grant)
    n_ontime_arr: int = 0          # trains arriving within threshold (realized)
    n_arrivals: int = 0            # realized corridor arrivals (transit events)
    ontime_rate: float = 0.0       # realized ontime arrivals / n_arrivals
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
    transit_events: Optional[List[dict]] = None,
) -> Metrics:
    """Compute scheduling + realised-execution metrics.

    If station_map is provided (train_id → current station), delays are
    computed relative to that station's timetable.  Otherwise, each train's
    origin_station is used (backward-compatible for two-station env).

    Arrival metrics use the REALISED arrivals from `transit_events` (L3 corridor
    execution: actual_arrival - planned_arrival) when provided — the physically
    meaningful punctuality.  Without events, they fall back to the L2 grant
    window (W_in), which is scheduling-level, not execution.
    """
    layer1 = DispatchBounds(cfg)
    held: Set[str] = set(held_train_ids)
    active: Set[str] = set(active_train_ids or cfg.trains.keys())

    dep_delays: List[float] = []
    for grant in grants:
        train = cfg.trains[grant.train_id]
        sid = station_map.get(grant.train_id) if station_map else train.origin_station
        bounds = layer1.for_train(grant.train_id, sid)
        dep_delays.append(abs(grant.W_out[0] - bounds.t_dep_plan))

    total_dep = sum(dep_delays)
    n_granted = len(grants)
    n_active = max(len(active), 1)
    n_ontime_dep = sum(1 for d in dep_delays if d <= ONTIME_THRESHOLD_SEC)

    # ---- realised-arrival metrics (L3 execution) ----
    if transit_events:
        arr_delays = [ev["arr_delay"] for ev in transit_events]
        n_arrivals = len(arr_delays)
        n_ontime_arr = sum(1 for d in arr_delays if abs(d) <= ONTIME_THRESHOLD_SEC)
        total_arr = sum(max(0.0, d) for d in arr_delays)   # late-only realized delay
        max_arr = max(arr_delays) if arr_delays else 0.0
    else:
        arr_delays = []
        for g in grants:
            train = cfg.trains[g.train_id]
            sid = station_map.get(g.train_id) if station_map else train.origin_station
            arr_delays.append(abs(g.W_in[0] - layer1.for_train(g.train_id, sid).t_arr_plan))
        n_arrivals = n_granted
        n_ontime_arr = sum(1 for d in arr_delays if d <= ONTIME_THRESHOLD_SEC)
        total_arr = sum(arr_delays)
        max_arr = max(arr_delays) if arr_delays else 0.0

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
        mean_arr_delay=total_arr / max(n_arrivals, 1),
        max_dep_delay=max(dep_delays) if dep_delays else 0.0,
        max_arr_delay=max_arr,
        unresolved_conflicts=unresolved_conflicts,
        deadlock_flag=1 if len(held) >= n_active else 0,
        completion_rate=n_granted / n_active,
        hard_bound_infeasible_count=hard_bound_infeasible_count,
        n_ontime_dep=n_ontime_dep,
        n_ontime_arr=n_ontime_arr,
        n_arrivals=n_arrivals,
        ontime_rate=n_ontime_arr / max(n_arrivals, 1),
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
    throat_delays: Optional[Dict[str, float]] = None,
) -> Dict[str, RewardBreakdown]:
    layer1 = DispatchBounds(cfg)
    held = set(held_train_ids)
    active = set(active_train_ids or cfg.trains.keys())
    grant_by_train = {g.train_id: g for g in grants}
    retried = set(retried_train_ids or [])
    visitor_windows = visitor_windows or {}
    corridor_delays = corridor_delays or {}
    throat_delays = throat_delays or {}
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
            # Hard ontime bonus: crossing the 15s threshold is explicitly rewarded
            # (aligns training with the eval metric), on top of the smooth decay.
            if dep_delay <= ONTIME_THRESHOLD_SEC:
                local += ONTIME_HARD_BONUS * prio_weight
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
            # Throat delay penalty: shared out-throat (e.g. C station) serialisation
            # is invisible to dep_delay (which uses the granted W_out[0]); surfacing
            # it lets L2 learn to avoid granting same-throat departures too close.
            thr_delay = throat_delays.get(train_id, 0.0)
            if thr_delay > 1e-6:
                local += -0.02 * thr_delay * prio_weight
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
    throat_delay_sec: float = 0.0,
) -> float:
    """Compute L3 Worker reward for one corridor transit.

    Objectives:
      - Minimise corridor headway and throat waiting time
      - Penalise late arrival only (early is acceptable)
      - Reward using recovery margin only when catching up from delay

    All terms are in seconds, scaled to comparable magnitudes.
    """
    total_delay = headway_delay_sec + throat_delay_sec
    late = max(arrival_deviation_sec, 0.0)
    # Reward recovery only when it helps reduce existing delay.
    # Small fuel penalty makes FAST a net cost when there is no delay to recover.
    effective_recovery = min(recovery_used_sec, total_delay + late)
    fuel_penalty = 0.003 * recovery_used_sec
    return (
        -0.02 * headway_delay_sec
        - 0.02 * throat_delay_sec
        - 0.01 * late
        + 0.01 * effective_recovery
        - fuel_penalty
    )


# ---------------------------------------------------------------------------
# L3 v3 sequential per-section reward (headway is a HARD boundary, not a reward)
# ---------------------------------------------------------------------------

def compute_l3_terminal_reward(arrival_deviation_sec: float, platform_free: bool,
                               priority: int = 2,
                               ontime_sigma: float = 10.0,
                               ontime_bonus: float = 3.0,
                               late_coef: float = 0.15,
                               late_sq_coef: float = 0.001,
                               early_coef: float = 0.05,
                               global_delay_sec: float = 0.0,
                               global_coef: float = 0.002) -> float:
    """Terminal reward at corridor end — railway KPI = ontime ARRIVALS / total.

    HIERARCHY (per railway ontime-rate assessment):
      1. PRIMARY = maximise the ontime arrival RATE: a LARGE smooth ontime bonus
         (Gaussian around dev=0) applies to EVERY arrival, so in expected reward it
         dominates (frequent term) and the policy learns to get as many trains on
         time as possible.
      2. SECONDARY = minimise the tail: superlinear late penalty (late_coef*dev +
         late_sq_coef*dev^2) is applied only to the late few, shaping behaviour
         among unavoidably-late trains without outranking the ontime rate.
    - PRIORITY: the whole local reward is scaled by PRIORITY_WEIGHT[priority] so
      higher-class trains (复兴号 > 和谐号 > 普速) are held to higher punctuality.
    - early: penalised REGARDLESS of platform state (2026-08-14) — the old
      "only if platform not free" rule let the policy collapse to always-FAST
      (arrive early when platforms are free = unpunished).  early_coef kept at
      0.05 (always-early at 0.1 over-corrected in v2: avoided FAST on restricted).
    - GLOBAL team term: -global_coef * system_delay for cross-train coordination.
    """
    w = PRIORITY_WEIGHT.get(priority, 1.0)
    r = ontime_bonus * math.exp(-(arrival_deviation_sec / ontime_sigma) ** 2)
    if arrival_deviation_sec > 0:
        late = late_coef * arrival_deviation_sec + late_sq_coef * arrival_deviation_sec ** 2
        r -= min(late, L3_LATE_CAP)
    else:
        r -= early_coef * abs(arrival_deviation_sec)
    if global_delay_sec > 0:
        r -= global_coef * global_delay_sec
    return w * r


def compute_l3_team_reward(episode_ontime_rate: float, episode_mean_dep: float,
                           ontime_bonus: float = 2.0,
                           delay_coef: float = 0.05) -> float:
    """Episode-level team reward for L3 coordination.

    Rewards the SYSTEM's running ontime rate and penalises the SYSTEM's mean
    departure delay (cumulative episode averages — smooth, low-variance).  Shared
    across workers via the env's running episode stats, it gives the learner an
    incentive to coordinate (e.g. slow into a congested downstream station to
    protect system punctuality / reduce system delay) rather than only optimising
    its own arrival.  This is what lets the learned policy beat the greedy
    per-transit SA baseline under congestion.  Still MAPPO: PPO + centralised
    critic + a team reward, standard for cooperative MARL.
    """
    return ontime_bonus * episode_ontime_rate - delay_coef * episode_mean_dep
