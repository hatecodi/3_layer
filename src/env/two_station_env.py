"""Two-station (A-B) scheduling environment.

Each episode schedules trains through their origin station. NOMINAL trains
follow the timetable plan; L2 ACTIVE trains use the policy.
"""
from __future__ import annotations

import random
from dataclasses import asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple

from ..common.base_env import BaseRailwayEnv
from .disturbance import (
    DEADLOCK_THRESHOLD,
    ControlStatus,
    CorridorDisturbance,
    DisturbanceInjector,
    TrainRuntimeState,
    compute_corridor_delay,
    layer2_control_status,
)
from .observation_builder import build_observations
from .reward import compute_layered_rewards, compute_metrics
from ..control.action_mask import (
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
from ..experiment.scenario_loader import ScenarioConfig, load_scenario

# ---------------------------------------------------------------------------
# Episode termination
# ---------------------------------------------------------------------------

MAX_EPISODE_TIME = 900.0           # V4: TRUNCATED threshold
TRAVEL_TIME_TO_THROAT = 30.0


class EpisodeStatus(Enum):
    RUNNING = "running"
    SUCCESS = "success"             # all trains scheduled
    TRUNCATED = "truncated"         # timeout
    DEADLOCK = "deadlock"           # irrecoverable


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------

class TwoStationEnv(BaseRailwayEnv):
    def __init__(
        self,
        cfg: Optional[ScenarioConfig] = None,
        decision_interval_sec: float = 10.0,
        disturbance_prob: float = 0.03,
        corridor_disturbances: Optional[List[CorridorDisturbance]] = None,
        initial_train_delays: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
    ):
        self.cfg = cfg or load_scenario()
        self.layer1 = DispatchBounds(self.cfg)
        self.decision_interval_sec = decision_interval_sec
        self.rng = random.Random(seed)
        self.disturbance = DisturbanceInjector(
            prob_per_step=disturbance_prob,
            rng=self.rng,
        )
        self.corridor_disturbances = corridor_disturbances or []
        self.initial_train_delays = initial_train_delays or {}
        self._force_policy = True  # training: all trains use policy
        self._t_now = 0.0
        self._step_idx = 0
        self._states: Dict[str, TrainRuntimeState] = {}
        self._pending_ids: List[str] = []       # trains not yet scheduled
        self._scheduled_ids: List[str] = []     # trains with granted windows
        self._deadlock_counter = 0
        self._episode_status = EpisodeStatus.RUNNING
        self._last_requests: List[WindowRequest] = []
        self._last_grants: List[GrantResult] = []
        self._last_held: List[str] = []
        self._last_infeasible = 0
        self._last_metrics = None

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[Dict[str, List[float]], Dict]:
        self._t_now = 0.0
        self._step_idx = 0
        self._deadlock_counter = 0
        self._episode_status = EpisodeStatus.RUNNING
        all_ids = sorted(self.layer1.train_ids)
        self._states = {
            tid: TrainRuntimeState(
                train_id=tid,
                delay_sec=self.initial_train_delays.get(tid, 0.0),
            )
            for tid in all_ids
        }
        self._pending_ids = list(all_ids)
        self._scheduled_ids = []
        self._last_requests = []
        self._last_grants = []
        self._last_held = []
        self._last_infeasible = 0
        self._last_metrics = None

        # Inject initial disturbances
        injected = self.disturbance.step(all_ids, self._states)
        obs = build_observations(self.cfg, [], [], self._t_now)
        active_ids = list(all_ids) if self._force_policy else self._active_l2_train_ids({})
        info = {
            "action_masks": self._action_masks(active_ids),
            "active_train_ids": active_ids,
            "pending_train_ids": self._pending_ids,
            "scheduled_train_ids": self._scheduled_ids,
            "control_status": self._control_status_snapshot({}),
            "disturbances": injected,
            "episode_status": self._episode_status.value,
            "t_now": self._t_now,
            "metrics": {},
        }
        return obs, info

    def step(
        self, action_dict: Optional[Dict[str, int]] = None
    ) -> Tuple[
        Dict[str, List[float]],    # observations
        Dict[str, float],          # rewards (train_id → scalar)
        bool,                      # terminated
        bool,                      # truncated
        Dict,                      # info
    ]:
        if self._episode_status != EpisodeStatus.RUNNING:
            raise RuntimeError(
                f"Episode already terminated ({self._episode_status.value}); "
                f"call reset().")

        action_dict = action_dict or {}

        # 1. Inject disturbances for this step
        injected = self.disturbance.step(self._pending_ids, self._states)

        # 2. Build requests: NOMINAL trains use plan, L2 ACTIVE use policy
        requests, held, infeasible = self._build_requests(action_dict)
        conflict_graph = build_conflict_graph(requests)
        # held from _build_requests already includes NO_LEGAL_ACTION trains

        # 3. Resolve
        grants = self._resolve_requests(requests)
        granted_ids = {g.train_id for g in grants}
        # trains that sent requests but got no grant are also held
        held.extend(r.train_id for r in requests if r.train_id not in granted_ids)
        held = sorted(set(held))

        # 4. Update train lifecycle
        newly_scheduled = []
        for tid in list(self._pending_ids):
            if tid in granted_ids:
                newly_scheduled.append(tid)
                self._states[tid].pending_layer2_target = False
                self._states[tid].held_steps = 0
            elif tid in held:
                self._states[tid].held_steps += 1
            else:
                # NOMINAL trains that didn't need a request: they are scheduled
                # via _build_requests adding them to requests with plan action
                pass

        for tid in newly_scheduled:
            self._pending_ids.remove(tid)
            self._scheduled_ids.append(tid)

        unresolved = sum(len(v) for v in conflict_graph.values()) // 2

        # 4.5. Corridor disturbance — apply speed restriction delay
        corridor_delays: Dict[str, float] = {}
        if self.corridor_disturbances:
            for g in grants:
                train = self.cfg.trains[g.train_id]
                corridor_id = train.corridor
                corridor_runtime = float(
                    self.cfg.time_generation.get("corridor_runtime_sec", {})
                    .get("A_to_B" if train.direction == "up" else "B_to_A", 90.0)
                )
                t_corridor_entry = g.W_out[1]  # t_out_end = corridor entry
                extra = compute_corridor_delay(
                    g.train_id, corridor_id, t_corridor_entry,
                    corridor_runtime, self.corridor_disturbances)
                if extra > 0:
                    corridor_delays[g.train_id] = extra
                    self._states[g.train_id].delay_sec += extra

        # 5. Deadlock detection
        self._deadlock_counter = (
            self._deadlock_counter + 1
            if len(self._pending_ids) > 0 and len(granted_ids) == 0
            else 0
        )

        # 6. Rewards & metrics — only for trains processed this step
        step_train_ids = self._step_train_ids(newly_scheduled, held)
        rewards = compute_layered_rewards(
            self.cfg, grants, held,
            active_train_ids=step_train_ids,
            unresolved_conflicts=unresolved,
            hard_bound_infeasible_count=infeasible,
        )
        metrics = compute_metrics(
            self.cfg, grants, held,
            active_train_ids=step_train_ids,
            unresolved_conflicts=unresolved,
            hard_bound_infeasible_count=infeasible,
        )

        # 7. Episode termination
        if self._deadlock_counter >= DEADLOCK_THRESHOLD:
            self._episode_status = EpisodeStatus.DEADLOCK
        elif len(self._pending_ids) == 0:
            self._episode_status = EpisodeStatus.SUCCESS
        elif self._t_now >= MAX_EPISODE_TIME:
            self._episode_status = EpisodeStatus.TRUNCATED

        # 8. Build next observation
        obs = build_observations(self.cfg, requests, grants, self._t_now)
        next_active_ids = list(self._pending_ids) if self._force_policy else self._active_l2_train_ids(conflict_graph)

        # 9. Advance time
        self._step_idx += 1
        self._t_now += self.decision_interval_sec

        # Save for debugging
        self._last_requests = requests
        self._last_grants = grants
        self._last_held = held
        self._last_infeasible = infeasible
        self._last_metrics = metrics

        terminated = self._episode_status in (
            EpisodeStatus.SUCCESS, EpisodeStatus.DEADLOCK)
        truncated = self._episode_status == EpisodeStatus.TRUNCATED

        info = {
            "action_masks": self._action_masks(next_active_ids),
            "requests": requests,
            "active_train_ids": next_active_ids,
            "pending_train_ids": self._pending_ids,
            "scheduled_train_ids": self._scheduled_ids,
            "control_status": self._control_status_snapshot(conflict_graph),
            "granted_windows": grants,
            "held_train_ids": held,
            "corridor_delays": corridor_delays,
            "disturbances": injected,
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

    def _build_requests(
        self, action_dict: Dict[str, int]
    ) -> Tuple[List[WindowRequest], List[str], int]:
        requests: List[WindowRequest] = []
        held: List[str] = []
        infeasible = 0
        empty_conflicts: Dict[str, set] = {}

        for train_id in list(self._pending_ids):
            train = self.cfg.trains[train_id]
            station_id = train.origin_station
            bounds = self.layer1.for_train(train_id, station_id)
            status = layer2_control_status(
                self._states[train_id], bounds, empty_conflicts.get(train_id))

            if self._force_policy or status == ControlStatus.LAYER2_ACTIVE:
                requests, held, infeasible = self._handle_l2_active(
                    train, train_id, station_id, action_dict,
                    requests, held, infeasible)
            else:
                requests = self._handle_nominal(
                    train, train_id, station_id, requests)

        return requests, held, infeasible

    def _handle_l2_active(
        self, train, train_id, station_id, action_dict,
        requests, held, infeasible,
    ) -> Tuple[List[WindowRequest], List[str], int]:
        """L2 ACTIVE train: use policy action or fall back to heuristic."""
        try:
            action = self._policy_action(train_id, action_dict)
        except ValueError:
            held.append(train_id)
            infeasible += 1
            return requests, held, infeasible

        request = action_to_request(
            train, station_id, action, self._t_now,
            TRAVEL_TIME_TO_THROAT,
            current_delay=self._states[train_id].delay_sec,
            cfg=self.cfg,
        )
        bounds = self.layer1.for_train(train_id, station_id)
        if request.W_in[0] < bounds.t_arr_min or request.W_out[0] > bounds.t_dep_max:
            held.append(train_id)
            infeasible += 1
            return requests, held, infeasible

        self._states[train_id].pending_layer2_target = True
        requests.append(request)
        return requests, held, infeasible

    def _handle_nominal(
        self, train, train_id, station_id, requests,
    ) -> List[WindowRequest]:
        """NOMINAL train: follow original timetable plan.

        Uses: first candidate platform, plan entry time, min dwell.
        """
        try:
            action = heuristic_action(
                train, station_id, self._t_now, TRAVEL_TIME_TO_THROAT, self.cfg)
        except ValueError:
            # Should not happen for nominal trains with valid timetable
            return requests

        request = action_to_request(
            train, station_id, action, self._t_now,
            TRAVEL_TIME_TO_THROAT,
            current_delay=self._states[train_id].delay_sec,
            cfg=self.cfg,
        )
        # Nominal trains are always allowed (they follow the plan exactly,
        # the plan is pre-validated to be feasible)
        requests.append(request)
        return requests

    def _policy_action(
        self, train_id: str, action_dict: Dict[str, int]
    ) -> Tuple[str, str, int]:
        train = self.cfg.trains[train_id]
        station_id = train.origin_station
        mask_set = compute_joint_action_mask(
            train, station_id, self._t_now, TRAVEL_TIME_TO_THROAT, self.cfg)
        if mask_set.status == "NO_LEGAL_ACTION":
            raise ValueError(f"No legal action for {train_id}")

        idx = action_dict.get(train_id)
        if idx is None:
            # No policy action provided → fall back to heuristic
            return heuristic_action(
                train, station_id, self._t_now, TRAVEL_TIME_TO_THROAT, self.cfg)
        if idx < 0 or idx >= len(mask_set.actions):
            raise ValueError(f"Action index out of range for {train_id}")
        if not mask_set.mask[idx]:
            raise ValueError(f"Masked action selected for {train_id}")

        spec: ActionSpec = mask_set.actions[idx]
        return spec.platform, spec.entry_choice, spec.dwell_extend

    # ------------------------------------------------------------------
    # resolver
    # ------------------------------------------------------------------

    def _resolve_requests(self, requests: List[WindowRequest]) -> List[GrantResult]:
        candidate_platforms = {
            tid: self.layer1.for_train(
                tid, self.cfg.trains[tid].origin_station).candidate_platforms
            for tid in self.layer1.train_ids
        }
        grants: List[GrantResult] = []
        for station_id in sorted(self.cfg.stations):
            station_requests = [
                r for r in requests
                if self.cfg.trains[r.train_id].origin_station == station_id
                or any(rid.startswith(station_id) for rid in r._resource_windows)
            ]
            # Deduplicate: a request may appear in multiple station groups
            # if its resources span stations; assign to its actual origin
            station_requests = [
                r for r in station_requests
                if self.cfg.trains[r.train_id].origin_station == station_id
            ]
            if station_requests:
                grants.extend(resolve_all(
                    station_requests,
                    candidate_platforms,
                    make_rebuild_fn(self.layer1, station_id),
                ))
        return grants

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _step_train_ids(
        self, newly_scheduled: List[str], held: List[str]
    ) -> List[str]:
        """Trains that were processed this step: scheduled + held."""
        return sorted(set(newly_scheduled) | set(held))

    def _active_l2_train_ids(self, conflict_graph: Dict[str, set]) -> List[str]:
        active = []
        for train_id in self._pending_ids:
            train = self.cfg.trains[train_id]
            status = layer2_control_status(
                self._states[train_id],
                self.layer1.for_train(train_id, train.origin_station),
                conflict_graph.get(train_id),
            )
            if status == ControlStatus.LAYER2_ACTIVE:
                active.append(train_id)
        return active

    def _control_status_snapshot(
        self, conflict_graph: Dict[str, set]
    ) -> Dict[str, str]:
        snapshot: Dict[str, str] = {}
        for train_id in self.layer1.train_ids:
            train = self.cfg.trains[train_id]
            snapshot[train_id] = layer2_control_status(
                self._states[train_id],
                self.layer1.for_train(train_id, train.origin_station),
                conflict_graph.get(train_id),
            ).value
        return snapshot

    def _action_masks(self, train_ids: List[str]) -> Dict[str, List[bool]]:
        masks: Dict[str, List[bool]] = {}
        for train_id in train_ids:
            train = self.cfg.trains[train_id]
            mask_set = compute_joint_action_mask(
                train, train.origin_station,
                t_now=self._t_now,
                travel_time_to_throat=TRAVEL_TIME_TO_THROAT,
                cfg=self.cfg,
            )
            masks[train_id] = mask_set.mask
        return masks

    # ------------------------------------------------------------------
    # BaseRailwayEnv interface properties
    # ------------------------------------------------------------------

    @property
    def pending_trains(self) -> List[str]:
        return self._pending_ids

    @property
    def last_requests(self) -> List[WindowRequest]:
        return self._last_requests
