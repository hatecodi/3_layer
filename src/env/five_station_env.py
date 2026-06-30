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

from ..common.base_env import BaseRailwayEnv
from ..control.action_mask import (
    ENTRY_CHOICES,
    THROAT_CLEARANCE_TIME,
    ActionSpec,
    action_to_request,
    compute_joint_action_mask,
    heuristic_action,
)
from ..control.dispatch_bounds import DispatchBounds
from ..control.resource_manager import (
    GrantResult,
    WindowRequest,
    build_conflict_graph,
    resolve_all,
)
from ..experiment.heuristic_planner import make_rebuild_fn
from ..experiment.scenario_loader import ScenarioConfig, TrainBound
from .disturbance import DEADLOCK_THRESHOLD
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
    ):
        self.cfg = cfg
        self.layer1 = DispatchBounds(cfg)
        self.decision_interval_sec = decision_interval_sec
        self.rng = random.Random(seed)
        self.batch_window_sec = batch_window_sec
        # Visitor trains: pre-booked trains occupying platforms at intermediate stations.
        # Config-driven: read from timetable.yaml time_generation.visitors, with fallback defaults.
        self._visitor_config = cfg.time_generation.get("visitors", {})
        self._visitor_enabled = self._visitor_config.get("enabled", True)
        self._visitor_windows: Dict[str, Tuple[float, float]] = {}
        self._visitor_list: Dict[str, list] = {}  # {station_id: [(plat, start, end, prio), ...]}

        # Build routes for each train
        self._routes: Dict[str, List[str]] = {}
        for tid, train in cfg.trains.items():
            stops = self._get_train_route(train)
            self._routes[tid] = stops

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

    def step(self, action_dict: Optional[Dict[str, int]] = None) -> Tuple:
        if self._episode_status != EpisodeStatus.RUNNING:
            raise RuntimeError("Episode terminated; call reset().")

        action_dict = action_dict or {}

        # 1. Build requests for all pending trains (within batch window)
        requests, held, infeasible = self._build_requests(
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

        # 4. Advance trains (no corridor delays)
        newly_advanced = self._advance_trains(grants)

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
            self.cfg, grants, held,
            active_train_ids=pending,
            unresolved_conflicts=unresolved,
            hard_bound_infeasible_count=infeasible,
            station_map=grant_station,
        )
        metrics = compute_metrics(
            self.cfg, grants, held,
            active_train_ids=pending,
            unresolved_conflicts=unresolved,
            hard_bound_infeasible_count=infeasible,
            station_map=grant_station,
        )

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

        return {
            "visitor_windows": self._visitor_windows if self._visitor_enabled else {},
            "visitors": self._visitor_list if self._visitor_enabled else {},
            "cumulative_delays": {},
            "corridor_delays": {},
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
        List[WindowRequest], List[str], int
    ]:
        requests: List[WindowRequest] = []
        held: List[str] = []
        infeasible = 0

        for train_id in self._pending_trains(lookahead=lookahead):
            if self._journeys[train_id].is_done:
                continue

            train = self.cfg.trains[train_id]
            station_id = self._journeys[train_id].current_station
            bounds = self._get_effective_bounds(train_id, station_id, self._t_now)

            try:
                # All trains use policy (no NOMINAL/heuristic branch)
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

            requests.append(request)
        return requests, held, infeasible

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
                    parked_windows=self._visitor_windows if self._visitor_enabled else None))
        return grants

    def _advance_trains(self, grants: List[GrantResult]) -> int:
        """Move granted trains to their next station, setting arrival time."""
        advanced = 0
        for g in grants:
            journey = self._journeys[g.train_id]
            if journey.next_station is None:
                # Final station — mark as done
                journey.advance()
                advanced += 1
                continue

            # Compute arrival time at next station
            # corridor = planned transit time from timetable (platform-to-platform).
            # Micro physics (throat traversal, speed profiles) deferred to L3.
            cur = journey.current_station
            nxt = journey.next_station
            train_plan = self.cfg.trains[g.train_id].planned
            if nxt in train_plan and cur in train_plan:
                planned_transit = train_plan[nxt].arr - train_plan[cur].dep
                runtime = max(planned_transit, 1.0)
            else:
                runtime = self._corridor_runtime(cur, nxt, self.cfg.trains[g.train_id].direction)
            t_corridor_exit = g.W_out[0] + runtime  # departure + platform-to-platform transit

            journey.advance()
            journey.set_arrival(t_corridor_exit)
            advanced += 1
        return advanced

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
            return orig_bounds

        actual_arrival = journey.arrival_at_current
        dwell = orig_bounds.dwell_min
        slack = self.cfg.time_generation.get("dep_max_slack_sec", 120)
        ref_time = max(actual_arrival, t_now)
        physical_overhead = TRAVEL_TIME_TO_THROAT + 2 * THROAT_CLEARANCE_TIME
        physical_min_dep = ref_time + physical_overhead + dwell
        return TrainBound(
            arr_min=actual_arrival,
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
