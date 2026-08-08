"""Five-station (A-B-C-D-E) scheduling environment.

Each train visits 5 stations (A→B→C→D→E or reverse). At each station,
the train competes for platform and throat resources. After being granted,
it transits to the next station via a corridor.

No delay injection — conflicts arise from natural timetable overlaps
(tight departure spacing at A, cross-direction at C, 2-platform bottleneck at D).

Key difference from two-station: trains have state transitions between
stations, creating a proper MDP where decisions at station N affect the
state at station N+1.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from ..common.base_env import BaseRailwayEnv
from ..control.action_mask import (
    ENTRY_CHOICES,
    THROAT_CLEARANCE_TIME,
    ActionSpec,
    action_to_request,
    compute_joint_action_mask,
    heuristic_action,
    is_legal_action,
)
from ..control.dispatch_bounds import DispatchBounds
from ..control.l3_sim import (
    L3Simulator,
    SpeedMode,
    calibrate_corridor_sections,
    compute_corridor_distances,
    compute_corridor_headway,
    load_l3_physics,
)
from ..control.resource_manager import (
    GrantResult,
    ResourceCalendar,
    WindowRequest,
    build_conflict_graph,
    resolve_all,
)
from ..experiment.heuristic_planner import make_rebuild_fn
from ..experiment.scenario_loader import ScenarioConfig, TrainBound
from .disturbance import DEADLOCK_THRESHOLD, inject_bunched_delay
from .observation_builder import build_observations
from .reward import compute_layered_rewards, compute_metrics

# ---------------------------------------------------------------------------
# Multi-station train tracking
# ---------------------------------------------------------------------------

@dataclass
class TrainJourney:
    """Tracks a train's progress through its multi-station route."""
    train_id: str
    route: List[str]           # ordered list of station IDs
    current_idx: int = 0       # index into route
    arrival_at_current: float = 0.0  # sim time when train arrived at current station
    corridor_runtime: float = 90.0   # normal corridor transit time (legacy)

    @property
    def current_station(self) -> str:
        if self.is_done:
            return ""
        return self.route[self.current_idx]

    @property
    def next_station(self) -> str | None:
        if self.current_idx + 1 < len(self.route):
            return self.route[self.current_idx + 1]
        return None

    @property
    def is_done(self) -> bool:
        return self.current_idx >= len(self.route)

    @property
    def is_in_transit(self) -> bool:
        """Train has been granted but hasn't arrived at next station yet."""
        return self.arrival_at_current < 0  # negative means hasn't arrived

    def set_arrival(self, t_arrival: float) -> None:
        """Set arrival time at the next station after corridor transit."""
        self.arrival_at_current = t_arrival

    def advance(self) -> None:
        """Move to next station. caller must set arrival_at_current afterward."""
        self.current_idx += 1


# ---------------------------------------------------------------------------
# Episode status
# ---------------------------------------------------------------------------

MAX_EPISODE_TIME = 1500.0
TRAVEL_TIME_TO_THROAT = 0.0   # zero — compressed timetable includes overhead


class EpisodeStatus(Enum):
    RUNNING = "running"
    SUCCESS = "success"
    TRUNCATED = "truncated"
    DEADLOCK = "deadlock"


# ---------------------------------------------------------------------------
# Multi-station environment
# ---------------------------------------------------------------------------

class FiveStationEnv(BaseRailwayEnv):
    def __init__(
        self,
        cfg: ScenarioConfig,
        decision_interval_sec: float = 10.0,
        seed: Optional[int] = None,
        batch_window_sec: float = 120.0,
        enable_vt: bool = True,
        enable_initial_delay: bool = True,
        l3_worker: Optional[object] = None,  # L3WorkerTrainer or None for rule-based
        l3_sa: bool = False,                 # SA per-transit baseline (eval only)
        l3_team_lambda: float = 0.3,         # L3 episode-level team-reward weight
    ):
        self.cfg = cfg
        self.layer1 = DispatchBounds(cfg)
        self.decision_interval_sec = decision_interval_sec
        self.rng = random.Random(seed)
        self.batch_window_sec = batch_window_sec
        self._enable_initial_delay = enable_initial_delay
        self._l3_worker = l3_worker  # None = rule-based cruise
        self._l3_sa = l3_sa
        self._l3_team_lambda = l3_team_lambda
        # Running episode system stats (for the L3 team reward): cumulative
        # over the episode, reset each reset().
        self._l3_ep_grants = 0
        self._l3_ep_ontime = 0
        self._l3_ep_delay = 0.0
        # Visitor trains: config-driven, can be disabled per experiment.
        self._visitor_config = cfg.time_generation.get("visitors", {})
        self._visitor_enabled = self._visitor_config.get("enabled", True) and enable_vt
        self._visitor_windows: Dict[str, Tuple[float, float]] = {}
        self._visitor_list: Dict[str, list] = {}  # {station_id: [(plat, start, end, prio), ...]}

        # Build routes for each train
        self._routes: Dict[str, List[str]] = {}
        for tid, train in cfg.trains.items():
            stops = self._get_train_route(train)
            self._routes[tid] = stops

        # Reserve p1 for pass-through trains only (except at D which only
        # has 2 platforms).  Stopping trains compete for p2/p3/p4.
        for tid, train in cfg.trains.items():
            if not train.pass_through:
                for st in list(train.candidate_platforms.keys()):
                    if st == 'D':
                        continue
                    plats = train.candidate_platforms[st]
                    if plats and st in cfg.stations:
                        p1_id = cfg.stations[st].platforms[0].id
                        train.candidate_platforms[st] = [p for p in plats if p != p1_id]

        # L3 physical simulator — per-corridor headway derived from timetable.
        # If config defines block sections, use calibrated block-section physics
        # (fixed-block headway + per-section speed profile); else fall back to
        # single-segment distance-based transit.
        l3_physics = load_l3_physics(cfg.project_root / "configs" / "five_station")
        corridor_hw = compute_corridor_headway(cfg.trains, l3_physics)
        sections = None
        if l3_physics.corridor_sections:
            sections = calibrate_corridor_sections(cfg.trains, l3_physics)
            corridor_dist = {}
        elif l3_physics.corridor_distances_m:
            corridor_dist = dict(l3_physics.corridor_distances_m)
        else:
            corridor_dist = compute_corridor_distances(cfg.trains, l3_physics)
        self._l3_sim = L3Simulator(l3_physics, corridor_hw, corridor_dist,
                                   sections=sections)
        self._corridor_delays: Dict[str, float] = {}  # train_id -> headway delay
        self._throat_delays: Dict[str, float] = {}    # train_id -> throat delay
        self._hard_guard_violations = 0
        self._l3_action_counts: Dict[int, int] = {}   # per-section speed choices (episode)
        self._l3_ep_reward: float = 0.0               # cumulative L3 worker reward this episode
        self._persistent_cal = ResourceCalendar(min_gap_sec=5.0)

        # State
        self._t_now = 0.0
        self._step_idx = 0
        self._journeys: Dict[str, TrainJourney] = {}
        self._held_steps: Dict[str, int] = {}
        self._deadlock_counter = 0
        self._episode_status = EpisodeStatus.RUNNING
        self._last_requests: List[WindowRequest] = []
        self._last_grants: List[GrantResult] = []
        self._last_held: List[str] = []
        self._completed_ids: List[str] = []

    def _get_train_route(self, train) -> List[str]:
        """Extract ordered station list for a train, from origin to destination only."""
        full_up = ["A", "B", "C", "D", "E"]
        full_down = ["E", "D", "C", "B", "A"]
        route = full_up if train.direction == "up" else full_down
        origin = train.origin_station
        dest = train.destination_station
        try:
            start = route.index(origin)
            end = route.index(dest)
            return route[start:end + 1]
        except (ValueError, IndexError):
            return route

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[Dict[str, List[float]], Dict]:
        self._t_now = 0.0
        self._step_idx = 0
        self._deadlock_counter = 0
        self._episode_status = EpisodeStatus.RUNNING
        self._completed_ids = []
        self._held_steps = {}
        self._corridor_delays = {}
        self._throat_delays = {}
        self._hard_guard_violations = 0
        self._l3_action_counts = {}
        self._l3_ep_reward = 0.0
        self._l3_ep_grants = 0
        self._l3_ep_ontime = 0
        self._l3_ep_delay = 0.0
        self._l3_sim.reset()
        self._persistent_cal = ResourceCalendar(min_gap_sec=5.0)

        all_ids = sorted(self.cfg.trains.keys())
        self._journeys = {}
        for tid in all_ids:
            j = TrainJourney(train_id=tid, route=list(self._routes[tid]))
            # Trains enter the line at their planned departure from origin.
            # Setting arrival_at_current = planned_dep prevents early polling.
            # Exception: single-station trains (origin==dest) use planned_arr
            # so they occupy the platform from arrival time (fixed visitors).
            origin = self.cfg.trains[tid].origin_station
            dest = self.cfg.trains[tid].destination_station
            if origin == dest:
                plan_arr = self.cfg.trains[tid].planned[origin].arr
                j.arrival_at_current = float(plan_arr)
            else:
                plan_dep = self.cfg.trains[tid].planned[origin].dep
                j.arrival_at_current = float(plan_dep)
            self._journeys[tid] = j

        # Inject bunched infrastructure delay: all trains in a random
        # time window at one origin station are held until the window
        # clears, creating corridor bunching and headway coupling.
        if self._enable_initial_delay:
            inject_bunched_delay(self._journeys, self.cfg.trains, self.rng)

        self._last_requests = []
        self._last_grants = []
        self._last_held = []
        # Generate visitor trains for this episode
        self._visitor_list, self._visitor_windows = self._generate_visitors()

        ctx = self._obs_context()
        obs = build_observations(self.cfg, [], [], self._t_now,
                                 station_map=self._journey_station_map(),
                                 context=ctx)

        pending = self._pending_trains()
        info = {
            "action_masks": self._action_masks(pending),
            "pending": pending,
            "completed": self._completed_ids,
            "active_train_ids": pending,
            "active_l2": pending,
            "requests": self._last_requests,
            "episode_status": self._episode_status.value,
            "t_now": self._t_now,
            "metrics": {
                "n_active": len(pending),
                "n_granted": 0, "n_held": 0,
                "mean_dep_delay": 0.0, "mean_arr_delay": 0.0,
                "total_dep_delay": 0.0,
            },
        }
        return obs, info

    def step(self, action_dict: Optional[Dict[str, int]] = None,
             lambda_team: float = 0.3) -> Tuple:
        if self._episode_status != EpisodeStatus.RUNNING:
            raise RuntimeError("Episode terminated; call reset().")

        action_dict = action_dict or {}

        # 1. Build requests for all pending trains (within batch window)
        requests, held, infeasible, retried = self._build_requests(
            action_dict, lookahead=self.batch_window_sec)
        conflict_graph = build_conflict_graph(requests)

        # 2. Resolve
        grants = self._resolve_requests(requests)
        granted_ids = {g.train_id for g in grants}
        held.extend(r.train_id for r in requests if r.train_id not in granted_ids)
        held = sorted(set(held))

        # 3. Capture grant-time station before advancing (for reward metrics)
        grant_station = {
            g.train_id: self._journeys[g.train_id].current_station
            for g in grants if not self._journeys[g.train_id].is_done
        }

        # 4. Advance trains
        newly_advanced, hard_guard_held = self._advance_trains(grants)
        effective_grants = [g for g in grants if g.train_id not in hard_guard_held]

        # 5. Update held_steps for deadlock tracking
        for tid in granted_ids:
            self._held_steps[tid] = 0
        for tid in held:
            self._held_steps[tid] = self._held_steps.get(tid, 0) + 1

        # 6. Check completion
        self._completed_ids = [
            tid for tid, j in self._journeys.items() if j.is_done
        ]

        # 7. Rewards & metrics (based on this decision's outcome)
        pending = self._pending_trains()
        unresolved = sum(len(v) for v in conflict_graph.values()) // 2
        rewards = compute_layered_rewards(
            self.cfg, effective_grants, held,
            active_train_ids=pending,
            unresolved_conflicts=unresolved,
            hard_bound_infeasible_count=infeasible + self._hard_guard_violations,
            station_map=grant_station,
            visitor_windows=self._visitor_windows if self._visitor_enabled else {},
            retried_train_ids=retried,
            lambda_team=lambda_team,
            corridor_delays=dict(self._corridor_delays),
            throat_delays=dict(self._throat_delays),
        )
        metrics = compute_metrics(
            self.cfg, effective_grants, held,
            active_train_ids=pending,
            unresolved_conflicts=unresolved,
            hard_bound_infeasible_count=infeasible + self._hard_guard_violations,
            station_map=grant_station,
        )
        # Accumulate running episode system stats (L3 team reward signal)
        self._l3_ep_grants += metrics.n_granted
        self._l3_ep_ontime += metrics.n_ontime_dep
        self._l3_ep_delay += metrics.total_dep_delay

        # 8. Advance time (single unified block)
        self._step_idx += 1
        if len(pending) == 0 and not all(j.is_done for j in self._journeys.values()):
            # All trains in transit — jump to next arrival
            future_arrivals = [
                j.arrival_at_current for j in self._journeys.values()
                if not j.is_done and j.arrival_at_current > self._t_now
            ]
            if future_arrivals:
                self._t_now = min(future_arrivals)
            else:
                self._t_now += self.decision_interval_sec
        else:
            self._t_now += self.decision_interval_sec

        # 9. Deadlock detection (global NO_LEGAL_ACTION count)
        masks = self._action_masks(self._pending_trains())
        no_legal = all(
            not any(masks.get(tid, [False]))
            for tid in self._pending_trains()
        ) if self._pending_trains() else False

        if no_legal:
            self._deadlock_counter += 1
        else:
            self._deadlock_counter = 0

        if self._deadlock_counter >= DEADLOCK_THRESHOLD:
            self._episode_status = EpisodeStatus.DEADLOCK
        elif all(j.is_done for j in self._journeys.values()):
            self._episode_status = EpisodeStatus.SUCCESS
        elif self._t_now >= MAX_EPISODE_TIME:
            self._episode_status = EpisodeStatus.TRUNCATED

        # 10. Observation (reflects state at new t_now for next decision)
        ctx = self._obs_context()
        obs = build_observations(self.cfg, requests, grants, self._t_now,
                                 station_map=self._journey_station_map(),
                                 context=ctx)

        self._last_requests = requests
        self._last_grants = grants
        self._last_held = held
        self._corridor_delays.clear()
        self._throat_delays.clear()

        terminated = self._episode_status in (EpisodeStatus.SUCCESS, EpisodeStatus.DEADLOCK)
        truncated = self._episode_status == EpisodeStatus.TRUNCATED

        info = {
            "action_masks": self._action_masks(self._pending_trains()),
            "requests": requests,
            "active_train_ids": self._pending_trains(),
            "pending": self._pending_trains(),
            "completed": self._completed_ids,
            "active_l2": pending,
            "granted_windows": grants,
            "held_train_ids": held,
            "reward_breakdown": {tid: asdict(r) for tid, r in rewards.items()},
            "metrics": asdict(metrics),
            "episode_status": self._episode_status.value,
            "deadlock_counter": self._deadlock_counter,
            "l3_speed_choices": getattr(self, "_l3_speed_choices", {}),
            "l3_action_counts": dict(getattr(self, "_l3_action_counts", {})),
            "retried_train_ids": retried,  # trains whose platform was changed by resolver
            "hard_guard_violations": self._hard_guard_violations,
            "hard_guard_held": hard_guard_held,
            "t_now": self._t_now,
        }
        return (
            obs,
            {tid: reward.total for tid, reward in rewards.items()},
            terminated,
            truncated,
            info,
        )

    # ------------------------------------------------------------------
    # request building
    # ------------------------------------------------------------------


    def _generate_visitors(self) -> tuple:
        """Generate visitor trains at intermediate stations.

        Visitor info is visible in observation as advance notification,
        allowing the policy to learn to avoid blocked platforms.

        Returns:
            visitors: {station_id: [(platform_id, start, end, priority), ...]}  for observation
            windows: {resource_id: (start, end)}  for resource calendar pre-booking
        """
        visitors: Dict[str, list] = {}
        windows: Dict[str, Tuple[float, float]] = {}
        if not self._visitor_enabled:
            return visitors, windows

        target_stations = self._visitor_config.get("target_stations", ["B", "C", "D"])
        prob_per_slot = self._visitor_config.get("prob_per_slot", 0.5)
        slots_range = self._visitor_config.get("slots_per_station", [1, 2])
        k_ratio = self._visitor_config.get("k_ratio", 0.6)
        k_dwell = self._visitor_config.get("k_dwell", [300, 900])
        g_dwell = self._visitor_config.get("g_dwell", [120, 300])

        for st in target_stations:
            if st not in self.cfg.stations:
                continue
            station = self.cfg.stations[st]
            if len(slots_range) == 1:
                n_slots = slots_range[0]
            else:
                n_slots = self.rng.randint(slots_range[0], slots_range[1])
            for _ in range(n_slots):
                if self.rng.random() >= prob_per_slot:
                    continue
                is_k = self.rng.random() < k_ratio
                prio = 3 if is_k else 2
                dwell = self.rng.uniform(k_dwell[0], k_dwell[1]) if is_k \
                    else self.rng.uniform(g_dwell[0], g_dwell[1])
                plat = self.rng.choice(station.platforms)
                # Bias: preferentially occupy p1 (main line), forcing policy
                # to learn platform switching when the main line is blocked.
                p1_bias = self._visitor_config.get("p1_bias", 0.0)
                if p1_bias > 0 and self.rng.random() < p1_bias and station.platforms:
                    plat = station.platforms[0]  # p1 is always first
                else:
                    # Exclude p1 if biased, so other platforms get the slot
                    candidates = station.platforms
                    if p1_bias > 0 and len(candidates) > 1:
                        candidates = station.platforms[1:]
                    plat = self.rng.choice(candidates)
                start = self.rng.uniform(0, max(0, MAX_EPISODE_TIME - dwell))
                end = start + dwell

                visitors.setdefault(st, []).append((plat.id, start, end, prio))
                windows[f"{st}/{plat.id}"] = (start, end)

        return visitors, windows


    def _obs_context(self) -> Dict:
        """Build context dict for observation builder."""
        pending = self._pending_trains(lookahead=self.batch_window_sec)
        # Count pending trains at each station for competitor awareness
        station_counts: Dict[str, int] = {}
        station_pending: Dict[str, List[Tuple[str, float]]] = {}
        for tid in pending:
            st = self._journeys[tid].current_station
            station_counts[st] = station_counts.get(st, 0) + 1
            planned_dep = self.cfg.trains[tid].planned[st].dep
            station_pending.setdefault(st, []).append((tid, planned_dep))
        pending_at_station = {}
        for tid in pending:
            st = self._journeys[tid].current_station
            pending_at_station[tid] = station_counts.get(st, 0) - 1

        # Cumulative delays from initial delay + propagated scheduling delays
        cumulative_delays = {}
        for tid, j in self._journeys.items():
            if j.is_done:
                continue
            train = self.cfg.trains[tid]
            st = j.current_station
            planned = train.planned.get(st)
            if planned:
                # Origin: delay = actual - planned_dep.  Intermediate: delay = actual_arrival - planned_arr.
                ref = planned.dep if j.current_idx == 0 else planned.arr
                cumulative_delays[tid] = max(0.0, j.arrival_at_current - ref)

        return {
            "visitor_windows": self._visitor_windows if self._visitor_enabled else {},
            "visitors": self._visitor_list if self._visitor_enabled else {},
            "cumulative_delays": cumulative_delays,
            "corridor_delays": dict(self._corridor_delays),
            "journeys": {tid: {"current": j.current_station, "next": j.next_station}
                        for tid, j in self._journeys.items() if not j.is_done},
            "pending_at_station": pending_at_station,
            "station_pending": station_pending,  # {st: [(tid, planned_dep), ...]}
        }

    def _pending_trains(self, lookahead: float = 0.0) -> List[str]:
        """Trains that have arrived (or will arrive within lookahead) and need scheduling.

        Args:
            lookahead: Include trains with arrival_at_current <= t_now + lookahead.
                       Used for batch scheduling at intermediate stations.
        """
        horizon = self._t_now + lookahead
        pending = []
        for tid, j in self._journeys.items():
            if j.is_done:
                continue
            if horizon >= j.arrival_at_current:
                pending.append(tid)
        return pending

    def _build_requests(self, action_dict: Dict[str, int], lookahead: float = 0.0) -> Tuple[
        List[WindowRequest], List[str], int, List[str]
    ]:
        """Build WindowRequests with intra-station sequential ordering.

        Trains at the same station are processed in priority order.  Higher-priority
        trains commit first; lower-priority trains see committed choices and avoid
        picking the same platform, reducing intra-step collisions that would otherwise
        be resolved by fallback.

        Returns:
            (requests, held, infeasible, retried)
            retried: train IDs that had their platform swapped by _retry_action
                     (distinguishes proactive avoidance from rescue retry).
        """
        requests: List[WindowRequest] = []
        held: List[str] = []
        infeasible = 0
        retried: List[str] = []

        # Group pending trains by station
        pending = self._pending_trains(lookahead=lookahead)
        by_station: Dict[str, List[str]] = defaultdict(list)
        for train_id in pending:
            if self._journeys[train_id].is_done:
                continue
            sid = self._journeys[train_id].current_station
            by_station.setdefault(sid, []).append(train_id)

        for station_id, station_trains in by_station.items():
            # Sort: priority (1 > 2 > 3), tiebreak by planned departure
            station_trains.sort(key=lambda tid: (
                self.cfg.trains[tid].priority,
                self.cfg.trains[tid].planned.get(station_id).dep
                if station_id in self.cfg.trains[tid].planned else 0.0,
            ))

            # Intra-step calendar tracks committed platform/throat windows
            station_cal = ResourceCalendar(min_gap_sec=5.0)
            # Seed with cross-step persistent bookings (skip self)
            pending_set = set(station_trains)
            # Hold each platform booking for min_gap AFTER the train leaves, so a
            # following train can't enter within the safety gap even when the
            # occupying train's window is zero-length / already in the past.
            cal_gap = self._persistent_cal._min_gap
            for rid, windows in self._persistent_cal._windows.items():
                if rid.startswith(f"{station_id}/"):
                    for w in windows:
                        if w.end + cal_gap > self._t_now and w.train_id not in pending_set:
                            station_cal.book(rid, w.train_id, w.start, w.end)

            for train_id in station_trains:
                train = self.cfg.trains[train_id]
                bounds = self._get_effective_bounds(train_id, station_id, self._t_now)

                try:
                    action = self._policy_action(train_id, station_id, bounds, action_dict)
                except ValueError:
                    held.append(train_id)
                    infeasible += 1
                    continue

                request = action_to_request(
                    train, station_id, action, self._t_now,
                    TRAVEL_TIME_TO_THROAT,
                    current_delay=0.0,
                    cfg=self.cfg,
                    bounds_override=bounds,
                )
                if request.W_in[0] < bounds.arr_min - 1e-6 or request.W_out[0] > bounds.dep_max + 1e-6:
                    held.append(train_id)
                    infeasible += 1
                    continue

                # Check against already-committed higher-priority trains
                if self._request_conflicts(request, station_cal):
                    request = self._retry_action(
                        train_id, station_id, bounds, action_dict, station_cal)
                    if request is None:
                        held.append(train_id)
                        infeasible += 1
                        continue
                    retried.append(train_id)

                # Commit to intra-step calendar
                self._book_to_calendar(request, station_cal)
                requests.append(request)

        return requests, held, infeasible, retried

    # ------------------------------------------------------------------
    # intra-step coordination helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _request_conflicts(request: WindowRequest, calendar: ResourceCalendar) -> bool:
        """True if any of *request*'s resource windows overlap with *calendar*."""
        for rid, (s, e) in request._resource_windows.items():
            if not calendar.is_free(rid, s, e):
                return True
        return False

    @staticmethod
    def _book_to_calendar(request: WindowRequest, calendar: ResourceCalendar) -> None:
        for rid, (s, e) in request._resource_windows.items():
            calendar.book(rid, request.train_id, s, e)

    def _retry_action(self, train_id: str, station_id: str, bounds,
                      action_dict, calendar: ResourceCalendar) -> Optional[WindowRequest]:
        """Try alternative (platform, entry) combos avoiding the committed calendar.

        Prefers the policy's entry_choice over other entries.  Falls through to
        heuristic_action if no policy-derived action fits.
        """
        train = self.cfg.trains[train_id]
        candidates = train.candidate_platforms.get(station_id, [])

        # Policy's preferred entry_choice
        entry_choice = "plan"
        if train_id in action_dict:
            _, entry_idx = action_dict.get(train_id, (0, 0))
            if 0 <= entry_idx < len(ENTRY_CHOICES):
                entry_choice = ENTRY_CHOICES[entry_idx]

        # Try policy's entry with every platform
        for entry in [entry_choice] + [e for e in ENTRY_CHOICES if e != entry_choice]:
            for plat in candidates:
                action = (plat, entry, 0)
                if not is_legal_action(
                    train, station_id, action, self._t_now,
                    TRAVEL_TIME_TO_THROAT, self.cfg, bounds_override=bounds,
                ):
                    continue
                request = action_to_request(
                    train, station_id, action, self._t_now,
                    TRAVEL_TIME_TO_THROAT, current_delay=0.0,
                    cfg=self.cfg, bounds_override=bounds,
                )
                if not self._request_conflicts(request, calendar):
                    return request

        # Fall through: heuristic
        try:
            plat, entry, _dwell = heuristic_action(
                train, station_id, self._t_now, TRAVEL_TIME_TO_THROAT,
                self.cfg, bounds_override=bounds)
            action = (plat, entry, 0)
            request = action_to_request(
                train, station_id, action, self._t_now,
                TRAVEL_TIME_TO_THROAT, current_delay=0.0,
                cfg=self.cfg, bounds_override=bounds,
            )
            if not self._request_conflicts(request, calendar):
                return request
        except ValueError:
            pass

        return None

    def _policy_action(self, train_id: str, station_id: str, bounds,
                       action_dict) -> Tuple:
        train = self.cfg.trains[train_id]
        mask_set = compute_joint_action_mask(
            train, station_id, self._t_now, TRAVEL_TIME_TO_THROAT, self.cfg,
            bounds_override=bounds)
        if mask_set.status == "NO_LEGAL_ACTION":
            raise ValueError(f"No legal action for {train_id}")

        action = action_dict.get(train_id)
        if action is None:
            return heuristic_action(
                train, station_id, self._t_now, TRAVEL_TIME_TO_THROAT,
                self.cfg, bounds_override=bounds)
        
        # Factored action: (platform_idx, entry_idx), dwell always 0
        platform_idx, entry_idx = action
        platforms = train.candidate_platforms.get(station_id, [])
        if platform_idx < 0 or platform_idx >= len(platforms):
            raise ValueError(f"Invalid platform index for {train_id}")
        if entry_idx < 0 or entry_idx >= len(ENTRY_CHOICES):
            raise ValueError(f"Invalid entry index for {train_id}")
        
        platform = platforms[platform_idx]
        entry_choice = ENTRY_CHOICES[entry_idx]
        return platform, entry_choice, 0  # min dwell always

    # ------------------------------------------------------------------
    # resolver + corridor
    # ------------------------------------------------------------------

    def _resolve_requests(self, requests: List[WindowRequest]) -> List[GrantResult]:
        grants: List[GrantResult] = []
        # Build parked windows from persistent calendar. Hold each platform for
        # min_gap after the train leaves (same fix as _build_requests).
        parked: Dict[str, Tuple[float, float]] = {}
        cal_gap = self._persistent_cal._min_gap
        for rid, windows in self._persistent_cal._windows.items():
            for w in windows:
                if w.end + cal_gap > self._t_now:
                    parked[rid] = (w.start, w.end)
        if self._visitor_enabled:
            for rid, (s, e) in self._visitor_windows.items():
                if e > self._t_now:
                    parked[rid] = (s, e)
        for station_id in sorted(self.cfg.stations):
            st_reqs = [r for r in requests
                       if self._journeys[r.train_id].current_station == station_id]
            if st_reqs:
                candidate_platforms = {
                    tid: self.layer1.for_train(
                        tid, self._journeys[tid].current_station).candidate_platforms
                    for tid in [r.train_id for r in st_reqs]
                }
                grants.extend(resolve_all(
                    st_reqs, candidate_platforms,
                    make_rebuild_fn(self.layer1, station_id),
                    parked_windows=parked if parked else None))
        # Book grants into persistent calendar (old bookings expire via t_now filter)
        granted_ids = {g.train_id for g in grants}
        for rid in self._persistent_cal._windows:
            self._persistent_cal._windows[rid] = [
                w for w in self._persistent_cal._windows[rid]
                if w.train_id not in granted_ids
            ]
        for g in grants:
            station_id = self._journeys[g.train_id].current_station
            self._persistent_cal.book(
                f"{station_id}/{g.platform}", g.train_id, g.W_plat[0], g.W_plat[1])
        return grants

    def _free_platforms_at(self, station_id: str, est_arrival: float,
                           dwell_est: float = 20.0) -> int:
        """Count platforms at a station that are free around an arrival window,
        based on the persistent calendar and visitor bookings.  Honest L3 obs
        input (replaces the previous max(total-1,1) guess)."""
        station = self.cfg.stations.get(station_id)
        if station is None:
            return 0
        cal = ResourceCalendar(min_gap_sec=5.0)
        for rid, ws in self._persistent_cal._windows.items():
            if rid.startswith(f"{station_id}/"):
                for w in ws:
                    cal.book(rid, w.train_id, w.start, w.end)
        if self._visitor_enabled:
            for rid, (s, e) in self._visitor_windows.items():
                if rid.startswith(f"{station_id}/"):
                    cal.book(rid, "PARKED", s, e)
        free = 0
        for plat in station.platforms:
            rid = f"{station_id}/{plat.id}"
            if cal.is_free(rid, est_arrival, est_arrival + dwell_est):
                free += 1
        return free

    def _system_delay(self) -> float:
        """Sum of all trains' current cumulative delay (live global signal for the
        L3 team reward, so the worker learns cross-train coordination)."""
        total = 0.0
        for tid, j in self._journeys.items():
            if j.is_done:
                continue
            train = self.cfg.trains[tid]
            planned = train.planned.get(j.current_station)
            if planned:
                ref = planned.dep if j.current_idx == 0 else planned.arr
                total += max(0.0, j.arrival_at_current - ref)
        return total

    def _advance_trains(self, grants: List[GrantResult]) -> int:
        """Move granted trains to their next station, setting arrival time.

        If an L3 Worker is attached, samples a speed_choice from the worker
        policy and records the transition for later training.
        """
        advanced = 0
        hard_guard_held: List[str] = []
        self._l3_speed_choices: Dict[str, int] = {}
        grants = sorted(grants, key=lambda g: g.W_out[0])
        for g in grants:
            journey = self._journeys[g.train_id]
            if journey.next_station is None:
                journey.advance()
                advanced += 1
                continue

            cur = journey.current_station
            nxt = journey.next_station
            train_plan = self.cfg.trains[g.train_id].planned
            corridor_key = f"{cur}_to_{nxt}"
            if nxt not in train_plan or cur not in train_plan:
                t_corridor_exit = g.W_out[0] + self._corridor_runtime(
                    cur, nxt, self.cfg.trains[g.train_id].direction)
                journey.advance()
                journey.set_arrival(t_corridor_exit)
                advanced += 1
                continue

            planned_transit = max(train_plan[nxt].arr - train_plan[cur].dep, 1.0)
            p = self._l3_sim.p
            sections = self._l3_sim._sections.get(corridor_key)
            bounds = self._l3_sim._sec_bounds.get(corridor_key, [0.0])
            distance = (sum(s.length_m for s in sections) if sections
                        else self._l3_sim.corridor_length(corridor_key))

            train_prio = self.cfg.trains[g.train_id].priority
            speeds = p.get_speeds(train_prio)

            # Throat conflict group for this platform+direction
            station_cfg = self.cfg.stations[cur]
            dir_cfg = station_cfg.up if self.cfg.trains[g.train_id].direction == "up" else station_cfg.down
            out_route = next((r for r in dir_cfg.out_routes if r.platform == g.platform), None)
            throat_group = out_route.conflict_group if out_route else None
            throat_id = f"{cur}/{throat_group}" if throat_group else None

            # Throat headway (before corridor entry) — hard serialization at the throat
            throat_exit = g.W_out[0] + p.throat_out_sec
            throat_delay = 0.0
            if throat_group:
                earliest_throat = self._l3_sim._throats.earliest_exit(
                    throat_id, p.throat_headway_sec)
                if throat_exit < earliest_throat:
                    throat_delay = earliest_throat - throat_exit
                    throat_exit = earliest_throat
                self._l3_sim._throats.record_exit(throat_id, throat_exit)

            headway_delay = 0.0
            flat_speeds = getattr(self, '_flat_speed_choices', {})
            flat_mode = None
            if g.train_id in flat_speeds and self._l3_worker is None:
                flat_mode = SpeedMode(flat_speeds[g.train_id])

            use_worker = self._l3_worker is not None and sections and distance > 0
            use_sa = (not use_worker) and self._l3_sa and sections and distance > 0
            t_corridor_exit = None

            if use_worker:
                # ---- L3 v3: sequential per-section speed-curve learning ----
                # headway is a HARD moving-block boundary (enforced below, never a
                # reward term); the worker optimises terminal punctuality + a
                # global team term (no energy objective).
                from ..env.observation_builder import build_l3_section_observation
                from ..env.reward import compute_l3_terminal_reward, compute_l3_team_reward
                from ..marl.l3_worker import L3Transition
                t = throat_exit
                accumulated_wait = 0.0
                total_run = 0.0
                entry_times: List[float] = []
                exit_times: List[float] = []
                n_sections = len(sections)
                nxt_station = self.cfg.stations.get(nxt)
                nxt_total = len(nxt_station.platforms) if nxt_station else 4
                planned_arrival = train_plan[nxt].arr
                speed_ratio = speeds.cruise_mps / max(speeds.max_mps, 1.0)
                last_action = 0
                for k, sec in enumerate(sections):
                    pos_k = bounds[k] if k < len(bounds) else 0.0
                    # HARD moving-block wait at this section (mode-independent)
                    wait_k, _ = self._l3_sim.section_progress(
                        corridor_key, t, pos_k, sec, SpeedMode.CRUISE, train_prio)
                    cum_delay = max(0.0, g.W_out[0] - train_plan[cur].dep) + accumulated_wait
                    nxt_free = self._free_platforms_at(nxt, t + planned_transit)
                    obs_k = build_l3_section_observation(
                        headway_gap_sec=wait_k,
                        accumulated_wait_sec=accumulated_wait,
                        cumulative_delay=cum_delay,
                        section_idx=k, n_sections=n_sections,
                        section_len_m=sec.length_m,
                        total_len_m=distance,
                        remaining_m=max(0.0, distance - pos_k),
                        max_distance_m=self._l3_sim.max_corridor_length(),
                        speed_ratio=speed_ratio,
                        next_platforms_free=nxt_free,
                        next_platforms_total=nxt_total,
                        throat_gap=self._l3_sim.throat_gap(throat_id, g.W_out[0]) if throat_id else 0.0)
                    obs_t = torch.tensor(obs_k, dtype=torch.float32)
                    action_k, logprob_k, value_k = self._l3_worker.select_action(obs_t)
                    last_action = int(action_k)
                    self._l3_action_counts[last_action] = self._l3_action_counts.get(last_action, 0) + 1

                    run_k = self._l3_sim.section_run_time(sec, SpeedMode(action_k), train_prio)

                    entry_times.append(t)
                    t += wait_k + run_k
                    exit_times.append(t)
                    accumulated_wait += wait_k
                    total_run += run_k

                    done = (k == n_sections - 1)
                    reward_k = 0.0  # no per-section reward (no energy objective)
                    if done:
                        arrival_dev = t + p.approach_sec + p.throat_in_sec - planned_arrival
                        platform_free = self._free_platforms_at(nxt, t) > 0
                        # local punctuality/delay + episode-level team reward
                        reward_k = compute_l3_terminal_reward(arrival_dev, platform_free)
                        if self._l3_ep_grants > 0:
                            running_ontime = self._l3_ep_ontime / self._l3_ep_grants
                            running_dep = self._l3_ep_delay / self._l3_ep_grants
                            reward_k += self._l3_team_lambda * compute_l3_team_reward(
                                running_ontime, running_dep)
                        self._l3_ep_reward += reward_k
                    # done=False between sections (sections ARE a Markov chain under
                    # the worker's policy); done=True only at the corridor end.
                    self._l3_worker.buffer.add(L3Transition(
                        obs=obs_t, action=action_k, logprob=logprob_k,
                        reward=reward_k, value=value_k, done=done))
                headway_delay = accumulated_wait
                t_corridor_exit = t + p.approach_sec + p.throat_in_sec
                self._l3_speed_choices[g.train_id] = last_action
                self._l3_sim._corridors.record_occupancy_abs(
                    corridor_key, g.train_id, entry_times, exit_times)
            elif use_sa:
                # ---- SA baseline: per-transit simulated annealing over the SAME
                # per-section speed space, SAME local objective (own arrival
                # deviation, headway hard). Greedy local upper bound for the eval.
                from ..control.l3_sa import sa_optimize_profile
                from ..env.reward import compute_l3_terminal_reward
                planned_arrival = train_plan[nxt].arr
                reward_fn = lambda dev, t: compute_l3_terminal_reward(
                    dev, self._free_platforms_at(nxt, t) > 0)
                modes, _ = sa_optimize_profile(
                    self._l3_sim, corridor_key, throat_exit, train_prio,
                    sections, bounds, planned_arrival, reward_fn)
                t = throat_exit
                accumulated_wait = 0.0
                entry_times: List[float] = []
                exit_times: List[float] = []
                for k, sec in enumerate(sections):
                    pos_k = bounds[k] if k < len(bounds) else 0.0
                    wait_k, run_k = self._l3_sim.section_progress(
                        corridor_key, t, pos_k, sec, SpeedMode(modes[k]), train_prio)
                    entry_times.append(t)
                    t += wait_k + run_k
                    exit_times.append(t)
                    accumulated_wait += wait_k
                headway_delay = accumulated_wait
                t_corridor_exit = t + p.approach_sec + p.throat_in_sec
                self._l3_speed_choices[g.train_id] = modes[-1]
                self._l3_sim._corridors.record_occupancy_abs(
                    corridor_key, g.train_id, entry_times, exit_times)
            else:
                # ---- backward compat: single-shot transit (CRUISE / flat) ----
                speed_choice = flat_mode or SpeedMode.CRUISE
                self._l3_speed_choices[g.train_id] = int(speed_choice)
                l3_result = self._l3_sim.transit(
                    g.train_id, cur, nxt, corridor_key,
                    planned_total_sec=planned_transit,
                    actual_departure=g.W_out[0], speed_choice=speed_choice,
                    throat_conflict_group=throat_group,
                    priority=train_prio)
                t_corridor_exit = l3_result.actual_arrival
                headway_delay = l3_result.headway_delay_sec
                throat_delay = l3_result.throat_delay_sec

            if headway_delay > 1e-6:
                self._corridor_delays[g.train_id] = headway_delay
            if throat_delay > 1e-6:
                self._throat_delays[g.train_id] = throat_delay

            # L3 hard guard: hold for replan only when headway delay is excessive
            # (fixed threshold, independent of per-corridor timetable-derived headway)
            replan_limit = p.replan_max_headway_delay_sec
            if headway_delay > replan_limit:
                self._hard_guard_violations += 1
                hard_guard_held.append(g.train_id)
                journey.arrival_at_current = max(
                    journey.arrival_at_current,
                    self._t_now + self.decision_interval_sec)
                continue

            journey.advance()
            journey.set_arrival(t_corridor_exit)
            advanced += 1
        return advanced, hard_guard_held

    def _get_effective_bounds(self, train_id: str, station_id: str, t_now: float):
        """Get scheduling bounds.

        Origin stations: use precomputed timetable bounds.
        Intermediate stations: shift to actual arrival time, with dep_max
        allowing scheduling from the later of actual arrival and current time.
        """
        train = self.cfg.trains[train_id]
        journey = self._journeys[train_id]
        orig_bounds = train.bounds.get(station_id)

        if journey.current_idx == 0 or orig_bounds is None:
            # At origin, arr_min should only shift when the train is actually
            # delayed (arrival > planned departure).  A non-delayed train starts
            # at the platform at t=0 and can depart on schedule — dwell_min is
            # satisfied by the planned dwell (planned_dep - planned_arr).
            if orig_bounds is not None:
                actual_arrival = journey.arrival_at_current
                planned_dep = train.planned[station_id].dep
                if actual_arrival > planned_dep + 1e-6:
                    slack = self.cfg.time_generation.get("dep_max_slack_sec", 120)
                    dwell = orig_bounds.dwell_min
                    return TrainBound(
                        arr_min=actual_arrival,
                        dep_max=max(actual_arrival + dwell + slack, orig_bounds.dep_max),
                        dwell_min=dwell,
                    )
            return orig_bounds

        actual_arrival = journey.arrival_at_current
        t_arr_plan = train.planned[station_id].arr if station_id in train.planned else 0.0
        dwell = orig_bounds.dwell_min
        slack = self.cfg.time_generation.get("dep_max_slack_sec", 120)
        # Floor at planned arrival — trains must not depart before schedule
        ref_time = max(actual_arrival, t_now, t_arr_plan)
        physical_overhead = TRAVEL_TIME_TO_THROAT + 2 * THROAT_CLEARANCE_TIME
        physical_min_dep = ref_time + physical_overhead + dwell
        return TrainBound(
            arr_min=ref_time,
            dep_max=max(ref_time + dwell + slack, physical_min_dep),
            dwell_min=dwell,
        )

    def _corridor_runtime(self, from_st: str, to_st: str, direction: str) -> float:
        """Look up corridor runtime from config."""
        time_gen = self.cfg.time_generation
        runtimes = time_gen.get("corridor_runtime_sec", {})
        key = f"{from_st}_to_{to_st}"
        return float(runtimes.get(key, 90.0))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _journey_station_map(self) -> Dict[str, str]:
        """train_id → current_station for all non-done trains."""
        return {
            tid: j.current_station
            for tid, j in self._journeys.items()
            if not j.is_done and j.current_station
        }

    def _action_masks(self, pending_ids: List[str]):
        """Return factored masks: {tid: (platform_mask, entry_mask)}.
        
        platform_mask: list of bool, length = candidate_platforms count (<= 4)
        entry_mask: list of bool, length = ENTRY_CHOICES count (6)
        Dwell is always 0 (min dwell).
        """
        masks = {}
        for tid in pending_ids:
            train = self.cfg.trains[tid]
            st = self._journeys[tid].current_station
            bounds = self._get_effective_bounds(tid, st, self._t_now)
            mask_set = compute_joint_action_mask(
                train, st, t_now=self._t_now,
                travel_time_to_throat=TRAVEL_TIME_TO_THROAT, cfg=self.cfg,
                bounds_override=bounds)
            
            # Compute factored masks: platform + entry, dwell=0 always
            platforms = train.candidate_platforms.get(st, [])
            n_plat = len(platforms)
            n_entry = len(ENTRY_CHOICES)
            p_mask = [False] * n_plat
            e_mask = [False] * n_entry
            
            for i, a in enumerate(mask_set.actions):
                if mask_set.mask[i]:
                    if a.dwell_extend == 0:  # only consider min dwell
                        plat_idx = platforms.index(a.platform) if a.platform in platforms else -1
                        entry_idx = ENTRY_CHOICES.index(a.entry_choice) if a.entry_choice in ENTRY_CHOICES else -1
                        if plat_idx >= 0:
                            p_mask[plat_idx] = True
                        if entry_idx >= 0:
                            e_mask[entry_idx] = True

            masks[tid] = (p_mask, e_mask)
        return masks

    # ------------------------------------------------------------------
    # BaseRailwayEnv interface properties
    # ------------------------------------------------------------------

    @property
    def pending_trains(self) -> List[str]:
        return self._pending_trains()

    @property
    def last_requests(self) -> List[WindowRequest]:
        return self._last_requests
