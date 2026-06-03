"""Layer 2 multi-agent scheduling environment.

This environment trains the Layer 2 dispatcher at the scheduling level.  Layer 3
SUMO execution remains a separate controller/replay layer that consumes
ExecutionTargets generated from granted windows.
"""
from __future__ import annotations

import random
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from .disturbance import (
    ControlStatus,
    DisturbanceInjector,
    TrainRuntimeState,
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


class RailwaySumoEnv:
    def __init__(
        self,
        cfg: Optional[ScenarioConfig] = None,
        decision_interval_sec: float = 10.0,
        control_step_sec: float = 1.0,
        disturbance_prob: float = 0.03,
        max_steps: int = 1,
        seed: Optional[int] = None,
    ):
        self.cfg = cfg or load_scenario()
        self.layer1 = DispatchBounds(self.cfg)
        self.decision_interval_sec = decision_interval_sec
        self.control_step_sec = control_step_sec
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.disturbance = DisturbanceInjector(
            prob_per_step=disturbance_prob,
            rng=self.rng,
        )
        self._done = False
        self._step_idx = 0
        self._t_now = 0.0
        self._states: Dict[str, TrainRuntimeState] = {}
        self._last_requests: List[WindowRequest] = []
        self._last_grants: List[GrantResult] = []
        self._last_held: List[str] = []
        self._active_train_ids: List[str] = []

    def reset(self) -> Tuple[Dict[str, List[float]], Dict]:
        self._done = False
        self._step_idx = 0
        self._t_now = 0.0
        self._states = {
            tid: TrainRuntimeState(train_id=tid)
            for tid in self.layer1.train_ids
        }
        injected = self.disturbance.step(self.layer1.train_ids, self._states)
        self._last_requests = []
        self._last_grants = []
        self._last_held = []
        self._active_train_ids = self._compute_active_train_ids({})
        obs = build_observations(self.cfg, [], [], self._t_now)
        info = {
            "action_masks": self._action_masks(self._active_train_ids),
            "active_train_ids": self._active_train_ids,
            "control_status": self._control_status_snapshot({}),
            "disturbances": injected,
            "metrics": {},
        }
        return obs, info

    def step(self, action_dict: Optional[Dict[str, int]] = None):
        if self._done:
            raise RuntimeError("Episode already terminated; call reset().")
        action_dict = action_dict or {}

        requests, held, hard_bound_infeasible = self._build_requests(action_dict)
        conflict_graph = build_conflict_graph(requests)
        grants = self._resolve_requests(requests)
        granted_ids = {g.train_id for g in grants}
        held.extend(r.train_id for r in requests if r.train_id not in granted_ids)
        held = sorted(set(held))

        self._last_requests = requests
        self._last_grants = grants
        self._last_held = held
        for tid in granted_ids:
            self._states[tid].pending_layer2_target = True
            self._states[tid].held_steps = 0
        for tid in held:
            self._states[tid].held_steps += 1

        unresolved_conflicts = sum(len(v) for v in conflict_graph.values()) // 2
        rewards = compute_layered_rewards(
            self.cfg,
            grants,
            held,
            active_train_ids=self._active_train_ids,
            unresolved_conflicts=unresolved_conflicts,
            hard_bound_infeasible_count=hard_bound_infeasible,
        )
        metrics = compute_metrics(
            self.cfg,
            grants,
            held,
            active_train_ids=self._active_train_ids,
            unresolved_conflicts=unresolved_conflicts,
            hard_bound_infeasible_count=hard_bound_infeasible,
        )

        obs = build_observations(self.cfg, requests, grants, self._t_now)
        self._step_idx += 1
        self._t_now += self.decision_interval_sec
        self._done = self._step_idx >= self.max_steps or metrics.deadlock_flag == 1

        info = {
            "action_masks": self._action_masks(self._active_train_ids),
            "active_train_ids": self._active_train_ids,
            "control_status": self._control_status_snapshot(conflict_graph),
            "granted_windows": grants,
            "held_train_ids": held,
            "reward_breakdown": {tid: asdict(r) for tid, r in rewards.items()},
            "metrics": asdict(metrics),
        }
        return (
            obs,
            {tid: reward.total for tid, reward in rewards.items()},
            self._done and metrics.deadlock_flag == 0,
            self._done and metrics.deadlock_flag == 1,
            info,
        )

    def _build_requests(
        self,
        action_dict: Dict[str, int],
    ) -> Tuple[List[WindowRequest], List[str], int]:
        requests: List[WindowRequest] = []
        held: List[str] = []
        hard_bound_infeasible = 0
        empty_conflicts: Dict[str, set] = {}
        self._active_train_ids = self._compute_active_train_ids(empty_conflicts)

        for train_id in self.layer1.train_ids:
            train = self.cfg.trains[train_id]
            station_id = train.origin_station
            status = layer2_control_status(
                self._states[train_id],
                self.layer1.for_train(train_id, station_id),
                empty_conflicts.get(train_id),
            )

            try:
                if status == ControlStatus.LAYER2_ACTIVE:
                    action = self._policy_action(train_id, action_dict)
                else:
                    action = heuristic_action(
                        train, station_id, self._t_now, 30.0, self.cfg)
            except ValueError:
                held.append(train_id)
                hard_bound_infeasible += 1
                continue

            bounds = self.layer1.for_train(train_id, station_id)
            request = action_to_request(
                train,
                station_id,
                action,
                self._t_now,
                30.0,
                current_delay=self._states[train_id].delay_sec,
                cfg=self.cfg,
            )
            if status == ControlStatus.LAYER2_ACTIVE:
                self._states[train_id].pending_layer2_target = True
            if request.W_in[0] < bounds.t_arr_min or request.W_out[0] > bounds.t_dep_max:
                held.append(train_id)
                hard_bound_infeasible += 1
                continue
            requests.append(request)
        return requests, held, hard_bound_infeasible

    def _policy_action(
        self,
        train_id: str,
        action_dict: Dict[str, int],
    ) -> tuple[str, str, int]:
        train = self.cfg.trains[train_id]
        station_id = train.origin_station
        mask_set = compute_joint_action_mask(
            train, station_id, self._t_now, 30.0, self.cfg)
        if mask_set.status == "NO_LEGAL_ACTION":
            raise ValueError(f"No legal action for {train_id}")
        action_index = action_dict.get(train_id)
        if action_index is None:
            action_index = next(i for i, ok in enumerate(mask_set.mask) if ok)
        if action_index < 0 or action_index >= len(mask_set.actions):
            raise ValueError(f"Action index out of range for {train_id}")
        if not mask_set.mask[action_index]:
            raise ValueError(f"Masked action selected for {train_id}")
        spec: ActionSpec = mask_set.actions[action_index]
        return spec.platform, spec.entry_choice, spec.dwell_extend

    def _resolve_requests(self, requests: List[WindowRequest]) -> List[GrantResult]:
        candidate_platforms = {
            tid: self.layer1.for_train(
                tid, self.cfg.trains[tid].origin_station).candidate_platforms
            for tid in self.layer1.train_ids
        }
        requests_by_station: Dict[str, List[WindowRequest]] = {}
        for request in requests:
            station_id = self.cfg.trains[request.train_id].origin_station
            requests_by_station.setdefault(station_id, []).append(request)

        grants: List[GrantResult] = []
        for station_id, station_requests in requests_by_station.items():
            grants.extend(resolve_all(
                station_requests,
                candidate_platforms,
                make_rebuild_fn(self.layer1, station_id),
            ))
        return grants

    def _compute_active_train_ids(self, conflict_graph: Dict[str, set]) -> List[str]:
        active = []
        for train_id in self.layer1.train_ids:
            train = self.cfg.trains[train_id]
            status = layer2_control_status(
                self._states[train_id],
                self.layer1.for_train(train_id, train.origin_station),
                conflict_graph.get(train_id),
            )
            if status == ControlStatus.LAYER2_ACTIVE:
                active.append(train_id)
        return active

    def _control_status_snapshot(self, conflict_graph: Dict[str, set]) -> Dict[str, str]:
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
                train,
                train.origin_station,
                t_now=self._t_now,
                travel_time_to_throat=30.0,
                cfg=self.cfg,
            )
            masks[train_id] = mask_set.mask
        return masks
