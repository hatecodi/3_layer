"""Two-station (A-B) scheduling environment — simplified, no delay injection.

Each episode schedules trains through their origin station.
All trains use the policy (no NOMINAL path).
"""
from __future__ import annotations

import random
from dataclasses import asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple

from ..common.base_env import BaseRailwayEnv
from .disturbance import DEADLOCK_THRESHOLD
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
        seed: Optional[int] = None,
    ):
        self.cfg = cfg or load_scenario()
        self.layer1 = DispatchBounds(self.cfg)
        self.rng = random.Random(seed)
        self._t_now = 0.0
        self._step_idx = 0
        self._pending_ids: List[str] = []       # trains not yet scheduled
        self._scheduled_ids: List[str] = []     # trains with granted windows
        self._held_steps: Dict[str, int] = {}
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
        self._held_steps = {}
        all_ids = sorted(self.layer1.train_ids)
        self._pending_ids = list(all_ids)
        self._scheduled_ids = []
        self._last_requests = []
        self._last_grants = []
        self._last_held = []
        self._last_infeasible = 0
        self._last_metrics = None

        obs = build_observations(self.cfg, [], [], self._t_now)
        info = {
            "action_masks": self._action_masks(all_ids),
            "active_train_ids": all_ids,
            "pending_train_ids": self._pending_ids,
            "scheduled_train_ids": self._scheduled_ids,
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

        # 1. Build requests: all trains use policy
        requests, held, infeasible = self._build_requests(action_dict)
        conflict_graph = build_conflict_graph(requests)

        # 2. Resolve
        grants = self._resolve_requests(requests)
        granted_ids = {g.train_id for g in grants}
        held.extend(r.train_id for r in requests if r.train_id not in granted_ids)
        held = sorted(set(held))

        # 3. Update train lifecycle
        newly_scheduled = []
        for tid in list(self._pending_ids):
            if tid in granted_ids:
                newly_scheduled.append(tid)
                self._held_steps[tid] = 0
            elif tid in held:
                self._held_steps[tid] = self._held_steps.get(tid, 0) + 1

        for tid in newly_scheduled:
            self._pending_ids.remove(tid)
            self._scheduled_ids.append(tid)

        unresolved = sum(len(v) for v in conflict_graph.values()) // 2

        # 4. Deadlock detection
        self._deadlock_counter = (
            self._deadlock_counter + 1
            if len(self._pending_ids) > 0 and len(granted_ids) == 0
            else 0
        )

        # 5. Rewards & metrics
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

        # 6. Episode termination
        if self._deadlock_counter >= DEADLOCK_THRESHOLD:
            self._episode_status = EpisodeStatus.DEADLOCK
        elif len(self._pending_ids) == 0:
            self._episode_status = EpisodeStatus.SUCCESS
        elif self._t_now >= MAX_EPISODE_TIME:
            self._episode_status = EpisodeStatus.TRUNCATED

        # 7. Build next observation
        obs = build_observations(self.cfg, requests, grants, self._t_now)
        next_active_ids = list(self._pending_ids)

        # 8. Advance time
        self._step_idx += 1
        self._t_now += self.decision_interval_sec if hasattr(self, 'decision_interval_sec') else 10.0

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

    def _build_requests(
        self, action_dict: Dict[str, int]
    ) -> Tuple[List[WindowRequest], List[str], int]:
        requests: List[WindowRequest] = []
        held: List[str] = []
        infeasible = 0

        for train_id in list(self._pending_ids):
            train = self.cfg.trains[train_id]
            station_id = train.origin_station
            bounds = self.layer1.for_train(train_id, station_id)

            try:
                action = self._policy_action(train_id, action_dict)
            except ValueError:
                held.append(train_id)
                infeasible += 1
                continue

            request = action_to_request(
                train, station_id, action, self._t_now,
                TRAVEL_TIME_TO_THROAT,
                current_delay=0.0,
                cfg=self.cfg,
            )
            if request.W_in[0] < bounds.t_arr_min or request.W_out[0] > bounds.t_dep_max:
                held.append(train_id)
                infeasible += 1
                continue

            requests.append(request)
        return requests, held, infeasible

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
        return sorted(set(newly_scheduled) | set(held))

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
