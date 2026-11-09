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
from collections import defaultdict, deque
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
from ..experiment.scenario_loader import (
    ScenarioConfig,
    TrainBound,
    TrainPlan,
    TrainTimetable,
    apply_pass_platform_policy,
    is_station_pass,
    pass_stations_for,
)
from .disturbance import (
    DEADLOCK_THRESHOLD,
    inject_bunched_delay,
    inject_corridor_closure,
    inject_corridor_incident,
)
from .observation_builder import (
    build_l3_section_observation,
    build_observations,
)
from .reward import (
    L2_PRIORITY_WEIGHT,
    YIELD_TAIL_THRESHOLD,
    compute_layered_rewards,
    compute_metrics,
)

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
        enable_corridor_incident: Optional[bool] = None,  # None -> follow config
        enable_corridor_closure: Optional[bool] = None,   # None -> follow config
        l3_worker: Optional[object] = None,  # L3WorkerTrainer or None for rule-based
        l3_sa: bool = False,                 # SA per-transit baseline (eval only)
        l3_sa_iters: int = 300,              # SA annealing budget per transit
                                             # (lower => weaker SA baseline)
        yield_entry_cap: bool = False,       # cap entry delay for yield cases
                                             # to plan/+10/+20 (exclude +30/+50)
        l3_sa_online: bool = False,          # partial-observation SA baseline (eval only)
        l3_sa_nofc: bool = False,            # SA WITHOUT incident forecast: plans as if
                                             # clear (factor=1) but executes the real
                                             # restriction -> the cost of needing a forecast
        record_l3_bc: bool = False,          # record (10-dim section obs, SA mode)
                                             # pairs for behavior cloning of L3
        enable_extra_train: bool = False,    # random mid-operation extra P2 train: spawns at
                                             # a random station (empty platform), transits
                                             # to D, stops at D (tests new-instance adaptation)
        extra_train_prob: float = 0.5,
        incident_scale: float = 1.0,         # >1 strengthens the incident (max_added x,
                                             # factor /, duration x) WITHOUT changing the
                                             # base config — opens the L2 re-dispatch window
        l3_team_lambda: float = 1.0,         # L3 episode-level team-reward weight
                                             # (>=1 so coordination pays; greedy SA
                                             # does NOT see this term -> asymmetric)
        control_coverage: str = "base",      # which decisions are exposed as
                                             # coordination-relevant in info:
                                             # base = pass / higher-prio follower
                                             # plus = + closure + any co-pending
                                             # full = every active train
    ):
        self.cfg = cfg
        self.layer1 = DispatchBounds(cfg)
        self.decision_interval_sec = decision_interval_sec
        self.rng = random.Random(seed)
        self.batch_window_sec = batch_window_sec
        self._enable_initial_delay = enable_initial_delay
        inc_cfg = cfg.time_generation.get("corridor_incident", {})
        self._incident_enabled = (
            enable_corridor_incident if enable_corridor_incident is not None
            else inc_cfg.get("enabled", False))
        clo_cfg = cfg.time_generation.get("corridor_closure", {})
        self._closure_enabled = (
            enable_corridor_closure if enable_corridor_closure is not None
            else clo_cfg.get("enabled", False))
        self._l3_worker = l3_worker  # None = rule-based cruise
        self._l3_sa = l3_sa
        self._l3_sa_iters = l3_sa_iters
        self._yield_entry_cap = yield_entry_cap
        self._l3_sa_online = l3_sa_online
        self._l3_sa_nofc = l3_sa_nofc
        self._record_l3_bc = record_l3_bc
        self._enable_extra_train = enable_extra_train
        self._extra_train_prob = extra_train_prob
        self._incident_scale = incident_scale
        self._l3_team_lambda = l3_team_lambda
        self._control_coverage = control_coverage
        # (L3 team reward now uses a sliding window of recent arrivals — see
        # _l3_win_delays — for a low-variance system punctuality signal.)
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

        # Resolve per-station pass-through and platform roles.  At stations
        # with a reserved through platform, pass trains use it and stopping
        # trains use the remaining platforms; uniform-platform stations (D)
        # leave candidates unchanged.  Config-driven, idempotent.
        for tid, train in cfg.trains.items():
            if not train.pass_stations and not train.pass_through:
                train.pass_stations = pass_stations_for(
                    train, cfg.stations, cfg.corridors)
            apply_pass_platform_policy(train, cfg.stations)

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
        # Moving-block externality ledger: front train -> sum(wait * w_follower),
        # charged at that train's next reward step (see scenario_loader/rewward).
        self._leader_externality: Dict[str, float] = {}
        # Yield-amplitude soft penalty: train_id -> avoidable excess seconds.
        self._yield_tail_charges: Dict[str, float] = {}
        # Station-level re-sequencing instrumentation.
        self._overtake_events: List[dict] = []
        self._overtake_total: int = 0
        self._yield_total: int = 0
        self._corridor_incidents: Dict[str, Tuple[float, float, float, float]] = {}
        # Corridor closure windows: corridor_id -> (t0, t1), no NEW entries in
        # [t0, t1].  Trains already inside continue.
        self._corridor_closures: Dict[str, Tuple[float, float]] = {}
        self._closure_held: Dict[str, float] = {}  # train_id -> hold seconds
        self._hard_guard_violations = 0
        # Mask diagnostics: how often the Layer-1 action mask rejects candidate
        # actions (and by which bound).  See _policy_action / _build_requests.
        self._mask_stats: Dict[str, int] = defaultdict(int)
        self._l3_action_counts: Dict[int, int] = {}   # per-section speed choices (episode)
        # behavior instrumentation: (mode, section_speed_factor, priority) per
        # section — for comparing SA vs learned-L3 mode distributions
        self._l3_behavior: List[Tuple[int, float, int]] = []
        self._l3_transit_events: List[dict] = []      # realized corridor transits (this step)
        self._l3_ep_reward: float = 0.0               # cumulative L3 worker reward this episode
        # Sliding window of recent realised arrival delays for the SMOOTH team
        # reward (low-variance system punctuality signal; instantaneous running
        # stats were too noisy early-episode and destabilised training).
        self._l3_win_delays: deque = deque(maxlen=20)
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
        self._leader_externality = {}
        self._yield_tail_charges = {}
        self._overtake_events = []
        self._overtake_total = 0
        self._yield_total = 0
        self._corridor_incidents = {}
        self._corridor_closures = {}
        self._closure_held = {}
        self._hard_guard_violations = 0
        self._mask_stats = defaultdict(int)
        self._l3_action_counts = {}
        self._l3_behavior = []
        self._l3_bc_samples: List[Tuple[List[float], int]] = []
        self._l3_transit_events = []
        self._l3_ep_reward = 0.0
        self._l3_win_delays.clear()
        self._l3_sim.reset()
        self._persistent_cal = ResourceCalendar(min_gap_sec=5.0)

        # random mid-operation extra P2 train: drop the previous episode's, add
        # one if triggered (a NEW instance out of the training distribution).
        for tid in [t for t in self.cfg.trains if t.startswith("EXTRA_")]:
            del self.cfg.trains[tid]
            self._routes.pop(tid, None)
        if self._enable_extra_train and self.rng.random() < self._extra_train_prob:
            etid = "EXTRA_0"
            extra = self._build_extra_train(etid, self.rng)
            self.cfg.trains[etid] = extra
            self._routes[etid] = self._get_train_route(extra)

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
        if self._incident_enabled:
            self._corridor_incidents = inject_corridor_incident(
                self.cfg, self.rng, severity_scale=self._incident_scale)
        if self._closure_enabled:
            self._corridor_closures = inject_corridor_closure(
                self.cfg, self.rng)

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
        # Policy action set: every train the batch scheduler will consider this
        # step (lookahead = batch_window_sec), not just trains that have already
        # arrived.  Without this the policy only controls ~14% of grants; the
        # rest are heuristic-scheduled 120s ahead.
        policy_ids = self._pending_trains(lookahead=self.batch_window_sec)
        relevant_ids = self._control_relevant_ids(
            policy_ids, ctx.get("station_pending_detail", {}))
        info = {
            "action_masks": self._action_masks(policy_ids),
            "pending": pending,
            "policy_train_ids": policy_ids,
            "control_relevant_ids": relevant_ids,
            "completed": self._completed_ids,
            "active_train_ids": policy_ids,
            "active_l2": policy_ids,
            "requests": self._last_requests,
            "episode_status": self._episode_status.value,
            "t_now": self._t_now,
            "metrics": {
                "n_active": len(policy_ids),
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

        # Policy action set for this step (batch window).  Rewards/metrics must
        # cover the same agents the policy acted for.
        policy_ids_step = self._pending_trains(lookahead=self.batch_window_sec)
        policy_station = {
            tid: self._journeys[tid].current_station
            for tid in policy_ids_step if not self._journeys[tid].is_done
        }
        # Trains currently held by a corridor closure (departure into [t0,t1]).
        self._closure_held = {}
        for tid in policy_ids_step:
            st = self._journeys[tid].current_station
            t1 = self._closure_hold_until(tid, st)
            if t1 is not None:
                plan = self.cfg.trains[tid].planned.get(st)
                planned_dep = plan.dep if plan is not None else self._t_now
                hold = max(0.0, t1 - planned_dep)
                if hold > 1e-6:  # only trains whose departure plan is held
                    self._closure_held[tid] = hold

        # 1. Build requests for all pending trains (within batch window)
        requests, held, infeasible, retried = self._build_requests(
            action_dict, lookahead=self.batch_window_sec)
        conflict_graph = build_conflict_graph(requests)

        # 2. Resolve
        grants = self._resolve_requests(requests)
        granted_ids = {g.train_id for g in grants}
        held.extend(r.train_id for r in requests if r.train_id not in granted_ids)
        held = sorted(set(held))

        # (station map for rewards/metrics was captured before requests;
        #  it covers every policy-active train, granted or held)

        # 4. Advance trains
        pre_grant = self._capture_pre_grant(grants)
        newly_advanced, hard_guard_held = self._advance_trains(grants)
        effective_grants = [g for g in grants if g.train_id not in hard_guard_held]
        self._yield_tail_charges = self._compute_yield_tail_charges(
            effective_grants, pre_grant)
        # Station-level re-sequencing: overtakes (arrived earlier, departed
        # later) and deliberate yields (lower priority delayed while a
        # higher-priority train departs no later).  Same-step detection only.
        self._overtake_events = self._detect_resequencing(
            effective_grants, pre_grant, retried=set(retried),
            requests=requests)
        self._overtake_total += sum(
            1 for e in self._overtake_events if e["is_overtake"])
        self._yield_total += sum(
            1 for e in self._overtake_events if e["is_yield"])
        # feed this step's realised arrivals into the team-reward sliding window
        for ev in self._l3_transit_events:
            self._l3_win_delays.append(ev["arr_delay"])

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
            active_train_ids=policy_ids_step,
            unresolved_conflicts=unresolved,
            hard_bound_infeasible_count=infeasible + self._hard_guard_violations,
            station_map=policy_station,
            visitor_windows=self._visitor_windows if self._visitor_enabled else {},
            retried_train_ids=retried,
            lambda_team=lambda_team,
            corridor_delays=dict(self._corridor_delays),
            throat_delays=dict(self._throat_delays),
            externality_charges=dict(self._leader_externality),
            yield_tail_charges=dict(self._yield_tail_charges),
        )
        # Charges were applied to every active train; drop them so the same
        # wait is never billed twice.  A non-active leader keeps its ledger
        # entry until its next reward step.
        for tid in rewards:
            self._leader_externality.pop(tid, None)
        metrics = compute_metrics(
            self.cfg, effective_grants, held,
            active_train_ids=policy_ids_step,
            unresolved_conflicts=unresolved,
            hard_bound_infeasible_count=infeasible + self._hard_guard_violations,
            station_map=policy_station,
            transit_events=list(self._l3_transit_events),
            overtake_count=sum(
                1 for e in self._overtake_events if e["is_overtake"]),
            yield_count=sum(
                1 for e in self._overtake_events if e["is_yield"]),
            pass_overtake_count=sum(
                1 for e in self._overtake_events
                if e["is_overtake"]
                and (e["pass_overtaken"] or e["pass_overtaker"])),
            priority_overtake_count=sum(
                1 for e in self._overtake_events
                if e.get("is_priority_overtake")),
            deliberate_yield_count=sum(
                1 for e in self._overtake_events
                if e.get("is_deliberate_yield")),
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
        self._corridor_delays.clear()
        self._throat_delays.clear()

        terminated = self._episode_status in (EpisodeStatus.SUCCESS, EpisodeStatus.DEADLOCK)
        truncated = self._episode_status == EpisodeStatus.TRUNCATED

        pending_next = self._pending_trains()
        policy_next = self._pending_trains(lookahead=self.batch_window_sec)
        relevant_next = self._control_relevant_ids(
            policy_next, ctx.get("station_pending_detail", {}))
        info = {
            "action_masks": self._action_masks(policy_next),
            "requests": requests,
            "active_train_ids": policy_next,
            "policy_train_ids": policy_next,
            "control_relevant_ids": relevant_next,
            "pending": pending_next,
            "completed": self._completed_ids,
            "active_l2": policy_next,
            "granted_windows": grants,
            "held_train_ids": held,
            "reward_breakdown": {tid: asdict(r) for tid, r in rewards.items()},
            "metrics": asdict(metrics),
            "episode_status": self._episode_status.value,
            "deadlock_counter": self._deadlock_counter,
            "l3_speed_choices": getattr(self, "_l3_speed_choices", {}),
            "l3_action_counts": dict(getattr(self, "_l3_action_counts", {})),
            "l3_transit_events": list(getattr(self, "_l3_transit_events", [])),
            "retried_train_ids": retried,  # trains whose platform was changed by resolver
            "corridor_closures": dict(self._corridor_closures),
            "closure_held": dict(self._closure_held),
            "closure_count": sum(
                1 for e in self._overtake_events
                if e.get("closure_related")),
            "hard_guard_violations": self._hard_guard_violations,
            "hard_guard_held": hard_guard_held,
            "mask_stats": dict(self._mask_stats),
            "overtake_events": list(self._overtake_events),
            "overtake_count": sum(
                1 for e in self._overtake_events if e["is_overtake"]),
            "yield_count": sum(
                1 for e in self._overtake_events if e["is_yield"]),
            "deliberate_yield_count": sum(
                1 for e in self._overtake_events
                if e.get("is_deliberate_yield")),
            "overtake_total": self._overtake_total,
            "yield_total": self._yield_total,
            "leader_externality": dict(self._leader_externality),
            "yield_tail_charges": dict(self._yield_tail_charges),
            "t_now": self._t_now,
        }
        return (
            obs,
            {tid: reward.total for tid, reward in rewards.items()},
            terminated,
            truncated,
            info,
        )

    def _build_extra_train(self, tid: str, rng: random.Random) -> TrainTimetable:
        """Random mid-operation extra P2 train.

        Spawns at a random station (A/B/C/E — a free platform is handled by the
        scheduler), transits toward D (passing intermediate stations with 0
        dwell), stops at D (经停), and terminates at D.  This is a NEW instance
        out of the training distribution — tests whether the learned L2/L3 adapt
        without retraining while SA must re-solve.
        """
        up = ["A", "B", "C", "D", "E"]
        down = ["E", "D", "C", "B", "A"]
        s = rng.choice(["A", "B", "C", "E"])
        direction = "up" if s in ("A", "B", "C") else "down"
        order = up if direction == "up" else down
        i, j = order.index(s), order.index("D")
        route = order[i:j + 1] if j >= i else order[j:i + 1][::-1]
        spawn_dep = rng.uniform(150.0, 500.0)
        d_dwell = 10.0
        planned: Dict[str, TrainPlan] = {}
        bounds: Dict[str, TrainBound] = {}
        t = spawn_dep
        for k, st in enumerate(route):
            if st == s:
                planned[st] = TrainPlan(arr=0.0, dep=t)
            elif st == "D":
                planned[st] = TrainPlan(arr=t, dep=t + d_dwell)
            else:
                planned[st] = TrainPlan(arr=t, dep=t)  # pass-through
            bounds[st] = TrainBound(arr_min=0.0, dep_max=t + 200.0,
                                    dwell_min=d_dwell if st == "D" else 0.0)
            if k < len(route) - 1:
                t += self._corridor_runtime(st, route[k + 1], direction)
        cand = {
            st: [p.id for p in self.cfg.stations[st].platforms]
            for st in route
        }
        train = TrainTimetable(
            id=tid, direction=direction, train_class="G", priority=2,
            pass_through=False, origin_station=s, destination_station="D",
            corridor=f"{s}_to_{route[1]}" if len(route) > 1 else f"{s}_to_D",
            planned=planned, bounds=bounds, candidate_platforms=cand)
        train.pass_stations = pass_stations_for(
            train, self.cfg.stations, self.cfg.corridors)
        apply_pass_platform_policy(train, self.cfg.stations)
        return train

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
        # Detailed pending list for follower-aware observations (priority,
        # expected arrival, per-station pass role).
        station_pending_detail: Dict[str, List[dict]] = {}
        for tid in pending:
            j = self._journeys[tid]
            st = j.current_station
            train = self.cfg.trains[tid]
            plan = train.planned.get(st)
            station_pending_detail.setdefault(st, []).append({
                "tid": tid,
                "priority": train.priority,
                "direction": train.direction,
                "arrival": j.arrival_at_current,
                "planned_dep": plan.dep if plan else 0.0,
                "is_pass": is_station_pass(train, st),
            })

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

        # Corridor closure reopen time per train (only its NEXT corridor).
        closure_until: Dict[str, float] = {}
        if self._corridor_closures:
            for tid, j in self._journeys.items():
                if j.is_done or j.next_station is None:
                    continue
                win = self._corridor_closures.get(
                    f"{j.current_station}_to_{j.next_station}")
                if win is not None and win[1] > self._t_now + 1e-6:
                    closure_until[tid] = win[1]

        return {
            "visitor_windows": self._visitor_windows if self._visitor_enabled else {},
            "visitors": self._visitor_list if self._visitor_enabled else {},
            "cumulative_delays": cumulative_delays,
            "corridor_delays": dict(self._corridor_delays),
            "journeys": {tid: {"current": j.current_station, "next": j.next_station}
                        for tid, j in self._journeys.items() if not j.is_done},
            "pending_at_station": pending_at_station,
            "station_pending": station_pending,  # {st: [(tid, planned_dep), ...]}
            "station_pending_detail": station_pending_detail,
            "closure_until": closure_until,
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
                    action = self._policy_action(train_id, station_id, bounds,
                                                 action_dict, mask_stats=self._mask_stats)
                except ValueError:
                    held.append(train_id)
                    infeasible += 1
                    self._mask_stats["no_legal_action"] += 1
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
                    # Redundant post-construction re-check: any hit here means the
                    # mask let an out-of-envelope action through (expected 0).
                    self._mask_stats["postcheck_reject"] += 1
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
                       action_dict, mask_stats=None) -> Tuple:
        train = self.cfg.trains[train_id]
        mask_set = compute_joint_action_mask(
            train, station_id, self._t_now, TRAVEL_TIME_TO_THROAT, self.cfg,
            bounds_override=bounds,
            reason_counter=mask_stats if mask_stats is not None else None)
        if mask_stats is not None:
            # Mask diagnostics (paper evidence: the mask is operative, not vacuous)
            n_masked = sum(1 for m in mask_set.mask if not m)
            mask_stats["decisions"] += 1
            mask_stats["candidates"] += len(mask_set.mask)
            mask_stats["masked_out"] += n_masked
            if n_masked:
                mask_stats["decisions_with_mask"] += 1
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
                # Cross-platform fallback pays a real re-routing delay
                # (swap_cost_sec) so a policy picking a colliding platform /
                # shared-throat group sees an actual departure delay, not a free
                # platform-swap.  Config-driven; default 0 = legacy free rescue.
                swap_cost_sec = self.cfg.time_generation.get("swap_cost_sec", 0.0)
                grants.extend(resolve_all(
                    st_reqs, candidate_platforms,
                    make_rebuild_fn(self.layer1, station_id),
                    parked_windows=parked if parked else None,
                    swap_cost_sec=swap_cost_sec))
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
        self._l3_transit_events = []
        # Execution/re-sequencing order is departure time; priority and id make
        # exact ties deterministic (yield schedules often land on the same
        # second as the train being yielded to).
        grants = sorted(grants, key=lambda g: (
            g.W_out[0], self.cfg.trains[g.train_id].priority, g.train_id))
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
            # Front occupant BEFORE this train is recorded on the corridor.
            # It is the train that any headway wait is attributable to.
            front_id = self._l3_sim.front_train_id(corridor_key)
            speeds = p.get_speeds(train_prio)
            # Per-train SCHEDULE (cruise) speed from the INITIAL timetable:
            # running at schedule = on-time arrival for every class (fixes P1/P3
            # structural early/late).  FAST = class max (recovery), SLOW = schedule/1.2.
            schedule_cruise = distance / max(planned_transit - p.corridor_overhead_sec, 1.0)
            schedule_cruise = min(schedule_cruise, speeds.max_mps)

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

            # Corridor speed-restriction incident window (start, end, f_inc,
            # sf_inc) or None.  Applied SECTION-RESOLVED via
            # L3Simulator.incident_section_factors: only the sections whose
            # traversal overlaps the window are slowed.  The old whole-transit
            # rule (any overlap -> slow the entire leg) turned an affected train
            # into a slow-moving front and stacked an unbounded moving-block
            # queue behind it (absurd 400-1900s headway holds).  All branches
            # (worker/SA/rule CRUISE) honor the same section-resolved physics.
            inc = self._corridor_incidents.get(corridor_key)
            affected = False

            headway_delay = 0.0
            flat_speeds = getattr(self, '_flat_speed_choices', {})
            flat_mode = None
            if g.train_id in flat_speeds and self._l3_worker is None:
                flat_mode = SpeedMode(flat_speeds[g.train_id])

            use_worker = self._l3_worker is not None and sections and distance > 0
            use_sa = (not use_worker) and (self._l3_sa or self._l3_sa_online or self._l3_sa_nofc) \
                and sections and distance > 0
            t_corridor_exit = None

            if use_worker:
                # ---- L3 v3: sequential per-section speed-curve learning ----
                # headway is a HARD moving-block boundary (enforced below, never a
                # reward term); the worker optimises terminal punctuality + a
                # global team term (no energy objective).
                from ..env.reward import (
                    L3_FAST_PENALTY,
                    compute_l3_terminal_reward,
                    compute_l3_team_reward,
                )
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
                speed_ratio = schedule_cruise / max(speeds.max_mps, 1.0)
                last_action = 0
                for k, sec in enumerate(sections):
                    pos_k = bounds[k] if k < len(bounds) else 0.0
                    # section-resolved incident factors: this section's traversal
                    # [t, t+run_est] overlaps the window -> slow this section only
                    sf_k, saf_k = 1.0, 1.0
                    if inc:
                        i_start, i_end, f_inc, sf_inc = inc
                        run_est = self._l3_sim.section_run_time(
                            sec, SpeedMode.CRUISE, train_prio, 1.0, schedule_cruise)
                        sf_k, saf_k = self._l3_sim.incident_section_factors(
                            i_start, i_end, f_inc, sf_inc, t, run_est)
                        if sf_k != 1.0:
                            affected = True
                    # HARD moving-block wait at this section (mode-independent)
                    wait_k, _ = self._l3_sim.section_progress(
                        corridor_key, t, pos_k, sec, SpeedMode.CRUISE, train_prio,
                        speed_factor=sf_k, safety_factor=saf_k,
                        schedule_cruise_mps=schedule_cruise)
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
                        throat_gap=self._l3_sim.throat_gap(throat_id, g.W_out[0]) if throat_id else 0.0,
                        speed_factor=sf_k)
                    obs_t = torch.tensor(obs_k, dtype=torch.float32)
                    action_k, logprob_k, v_local, v_global = self._l3_worker.select_action(obs_t)
                    last_action = int(action_k)
                    self._l3_action_counts[last_action] = self._l3_action_counts.get(last_action, 0) + 1
                    self._l3_behavior.append((last_action, sf_k, train_prio))

                    # Execute the CHOSEN mode with the continuous moving-block
                    # hard boundary.  The obs probe (wait_k, CRUISE) above is the
                    # nominal headway gap the worker sees; the actual wait may be
                    # larger when FAST would close the gap mid-section — a FAST
                    # follower cannot catch the front (P2 fix).
                    wait_k_run, run_k = self._l3_sim.section_progress(
                        corridor_key, t, pos_k, sec, SpeedMode(action_k), train_prio,
                        speed_factor=sf_k, safety_factor=saf_k,
                        schedule_cruise_mps=schedule_cruise)

                    entry_times.append(t)
                    t += wait_k_run + run_k
                    exit_times.append(t)
                    accumulated_wait += wait_k_run
                    total_run += run_k

                    done = (k == n_sections - 1)
                    # per-section energy cost of FAST: it is NOT free, so the worker
                    # only runs FAST when recovering delay is worth the energy —
                    # otherwise it collapses to "always FAST under stress".
                    # Decomposed rewards for the TWO critics: reward_local = own
                    # corridor outcome; reward_global = episode-level team term.
                    reward_local_k = -L3_FAST_PENALTY if action_k == 1 else 0.0
                    reward_global_k = 0.0
                    if done:
                        arrival_dev = t + p.approach_sec + p.throat_in_sec - planned_arrival
                        platform_free = self._free_platforms_at(nxt, t) > 0
                        reward_local_k = compute_l3_terminal_reward(
                            arrival_dev, platform_free, priority=train_prio)
                        if self._l3_win_delays:
                            w = self._l3_win_delays
                            win_ontime = sum(1 for d in w if abs(d) <= 15.0) / len(w)
                            win_dep = sum(max(0.0, d) for d in w) / len(w)
                            reward_global_k = self._l3_team_lambda * compute_l3_team_reward(
                                win_ontime, win_dep)
                        self._l3_ep_reward += reward_local_k + reward_global_k
                    # done=False between sections (sections ARE a Markov chain under
                    # the worker's policy); done=True only at the corridor end.
                    self._l3_worker.buffer.add(L3Transition(
                        obs=obs_t, action=action_k, logprob=logprob_k,
                        reward_local=reward_local_k, reward_global=reward_global_k,
                        value_local=v_local, value_global=v_global, done=done))
                headway_delay = accumulated_wait
                t_corridor_exit = t + p.approach_sec + p.throat_in_sec
                self._l3_speed_choices[g.train_id] = last_action
                self._l3_sim._corridors.record_occupancy_abs(
                    corridor_key, g.train_id, entry_times, exit_times)
            elif use_sa:
                # ---- SA baselines over the SAME per-section speed space, SAME
                # local objective (own arrival deviation, headway hard).
                # l3_sa          = full-information offline oracle (jointly anneals
                #                  the whole profile, sees the front occupant).
                # l3_sa_online   = partial-observation: per-section greedy recompute
                #                  from current state only (headway now, delay now,
                #                  geometry ahead), no lookahead — the fair match
                #                  for the learned worker's 10-dim obs.
                # l3_sa_nofc     = SA that does NOT know the incident: it PLANS as if
                #                  clear (speed/safety factor 1.0) but EXECUTES the
                #                  real restriction.  Demonstrates the reactive (vs
                #                  forecast-needing) value: MAPPO sees reality each
                #                  section and adapts; SA is locked to a stale plan.
                if self._l3_sa_online:
                    from ..control.l3_sa import sa_online_profile
                    profile_fn = sa_online_profile
                else:
                    from ..control.l3_sa import sa_optimize_profile
                    profile_fn = sa_optimize_profile
                from ..env.reward import compute_l3_terminal_reward
                planned_arrival = train_plan[nxt].arr
                reward_fn = lambda dev, t: compute_l3_terminal_reward(
                    dev, self._free_platforms_at(nxt, t) > 0, priority=train_prio)
                # SA planning sees the incident as a section-resolved window
                # (same physics as execution); the no-forecast variant plans as
                # clear (window withheld).
                plan_window = None if self._l3_sa_nofc else inc
                modes, _ = profile_fn(
                    self._l3_sim, corridor_key, throat_exit, train_prio,
                    sections, bounds, planned_arrival, reward_fn,
                    n_iter=self._l3_sa_iters,
                    incident_window=plan_window,
                    schedule_cruise_mps=schedule_cruise)
                t = throat_exit
                accumulated_wait = 0.0
                entry_times: List[float] = []
                exit_times: List[float] = []
                n_sections_sa = len(sections)
                nxt_station_sa = self.cfg.stations.get(nxt)
                nxt_total_sa = (len(nxt_station_sa.platforms)
                                if nxt_station_sa else 4)
                speed_ratio_sa = schedule_cruise / max(speeds.max_mps, 1.0)
                for k, sec in enumerate(sections):
                    pos_k = bounds[k] if k < len(bounds) else 0.0
                    sf_k, saf_k = 1.0, 1.0
                    if inc:
                        i_start, i_end, f_inc, sf_inc = inc
                        run_est = self._l3_sim.section_run_time(
                            sec, SpeedMode(modes[k]), train_prio, 1.0, schedule_cruise)
                        sf_k, saf_k = self._l3_sim.incident_section_factors(
                            i_start, i_end, f_inc, sf_inc, t, run_est)
                        if sf_k != 1.0:
                            affected = True
                    if self._record_l3_bc:
                        wait_probe, _ = self._l3_sim.section_progress(
                            corridor_key, t, pos_k, sec, SpeedMode.CRUISE,
                            train_prio, speed_factor=sf_k, safety_factor=saf_k,
                            schedule_cruise_mps=schedule_cruise)
                        cum_delay_sa = (max(0.0, g.W_out[0]
                                            - train_plan[cur].dep)
                                        + accumulated_wait)
                        nxt_free_sa = self._free_platforms_at(
                            nxt, t + planned_transit)
                        obs_k = build_l3_section_observation(
                            headway_gap_sec=wait_probe,
                            accumulated_wait_sec=accumulated_wait,
                            cumulative_delay=cum_delay_sa,
                            section_idx=k, n_sections=n_sections_sa,
                            section_len_m=sec.length_m,
                            total_len_m=distance,
                            remaining_m=max(0.0, distance - pos_k),
                            max_distance_m=self._l3_sim.max_corridor_length(),
                            speed_ratio=speed_ratio_sa,
                            next_platforms_free=nxt_free_sa,
                            next_platforms_total=nxt_total_sa,
                            throat_gap=(self._l3_sim.throat_gap(
                                throat_id, g.W_out[0]) if throat_id else 0.0),
                            speed_factor=sf_k)
                        self._l3_bc_samples.append(
                            (obs_k, int(modes[k])))
                    wait_k, run_k = self._l3_sim.section_progress(
                        corridor_key, t, pos_k, sec, SpeedMode(modes[k]), train_prio,
                        speed_factor=sf_k, safety_factor=saf_k,
                        schedule_cruise_mps=schedule_cruise)
                    self._l3_behavior.append((int(modes[k]), sf_k, train_prio))
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
                # ---- rule-based CRUISE / flat: SAME stepwise per-section
                # physics as the worker and SA (continuous moving-block via
                # section_progress).  Previously this called transit(), which
                # enforced a stricter continuous ENTRY and double-counted the
                # throat headway -> an unfair comparison with worker/SA.  The
                # throat headway is already applied above (throat_exit).
                speed_choice = flat_mode or SpeedMode.CRUISE
                self._l3_speed_choices[g.train_id] = int(speed_choice)
                if sections:
                    t = throat_exit
                    accumulated_wait = 0.0
                    entry_times: List[float] = []
                    exit_times: List[float] = []
                    for k, sec in enumerate(sections):
                        pos_k = bounds[k] if k < len(bounds) else 0.0
                        sf_k, saf_k = 1.0, 1.0
                        if inc:
                            i_start, i_end, f_inc, sf_inc = inc
                            run_est = self._l3_sim.section_run_time(
                                sec, speed_choice, train_prio, 1.0, schedule_cruise)
                            sf_k, saf_k = self._l3_sim.incident_section_factors(
                                i_start, i_end, f_inc, sf_inc, t, run_est)
                            if sf_k != 1.0:
                                affected = True
                        wait_k, run_k = self._l3_sim.section_progress(
                            corridor_key, t, pos_k, sec, speed_choice, train_prio,
                            speed_factor=sf_k, safety_factor=saf_k,
                            schedule_cruise_mps=schedule_cruise)
                        self._l3_behavior.append((int(speed_choice), sf_k, train_prio))
                        entry_times.append(t)
                        t += wait_k + run_k
                        exit_times.append(t)
                        accumulated_wait += wait_k
                    headway_delay = accumulated_wait
                    t_corridor_exit = t + p.approach_sec + p.throat_in_sec
                    self._l3_sim._corridors.record_occupancy_abs(
                        corridor_key, g.train_id, entry_times, exit_times)
                else:
                    # no sections: fall back to the legacy single-shot transit
                    # (whole-transit restriction, as before)
                    w_sf, w_saf, w_aff = 1.0, 1.0, False
                    if inc:
                        i_start, i_end, f_inc, sf_inc = inc
                        if not (throat_exit + planned_transit <= i_start
                                or throat_exit >= i_end):
                            w_sf, w_saf, w_aff = f_inc, sf_inc, True
                            affected = True
                    l3_result = self._l3_sim.transit(
                        g.train_id, cur, nxt, corridor_key,
                        planned_total_sec=planned_transit,
                        actual_departure=g.W_out[0], speed_choice=speed_choice,
                        throat_conflict_group=throat_group,
                        priority=train_prio,
                        speed_factor=w_sf, safety_factor=w_saf,
                        schedule_cruise_mps=schedule_cruise)
                    t_corridor_exit = l3_result.actual_arrival
                    headway_delay = l3_result.headway_delay_sec
                    throat_delay = l3_result.throat_delay_sec

            if headway_delay > 1e-6:
                self._corridor_delays[g.train_id] = headway_delay
                # Internalise the moving-block externality: charge the front
                # train for the priority-weighted wait this follower incurred.
                if front_id and front_id != g.train_id:
                    w_f = L2_PRIORITY_WEIGHT.get(train_prio, 1.0)
                    self._leader_externality[front_id] = (
                        self._leader_externality.get(front_id, 0.0)
                        + headway_delay * w_f)
            if throat_delay > 1e-6:
                self._throat_delays[g.train_id] = throat_delay

            # Realized corridor transit (eval instrumentation: actual vs planned
            # arrival, and whether the incident restriction applied)
            if t_corridor_exit is not None:
                # `affected` above is decided by the REALIZED trajectory
                # (sf_k != 1.0), so the affected SUBSET depends on the policy —
                # a policy that shifts its transits out of the window shrinks it.
                # Comparing "affected delay" across policies would then compare
                # different trains.  `affected_plan` is the same overlap test on
                # the PLANNED timetable window only, so it is policy-independent
                # and can be used as the exogenous subset.  Eval instrumentation
                # only: nothing in the env reads it back.
                affected_plan = False
                if inc:
                    p_entry = train_plan[cur].dep
                    p_exit = p_entry + float(
                        self.cfg.time_generation.get("corridor_runtime_sec", {})
                        .get(corridor_key, 90.0))
                    affected_plan = (p_entry < inc[1]) and (p_exit > inc[0])
                # Closure disturbance: realized hold at this departure (
                # policy-dependent) and the plan-based subset (policy-independent:
                # planned departure inside [t0, t1], i.e. a new entry blocked).
                closure_hold_sec = float(self._closure_held.get(
                    g.train_id, 0.0))
                closure_affected = closure_hold_sec > 1e-6
                cw = self._corridor_closures.get(corridor_key)
                closure_plan = bool(
                    cw is not None
                    and cw[0] - 1e-6 <= train_plan[cur].dep <= cw[1] + 1e-6)
                affected = affected or closure_affected
                affected_plan = affected_plan or closure_plan
                self._l3_transit_events.append({
                    "train_id": g.train_id, "corridor": corridor_key,
                    "entry": throat_exit, "actual_arrival": t_corridor_exit,
                    "planned_arrival": train_plan[nxt].arr,
                    "arr_delay": t_corridor_exit - train_plan[nxt].arr,
                    "headway_delay": headway_delay, "affected": affected,
                    "affected_plan": affected_plan,
                    "closure_affected": closure_affected,
                    "closure_affected_plan": closure_plan,
                    "closure_hold_sec": closure_hold_sec,
                    "priority": train_prio, "front_train": front_id,
                    "dep_station": cur, "arr_station": nxt,
                    "is_pass_dep": is_station_pass(
                        self.cfg.trains[g.train_id], cur),
                })

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

    def _has_higher_prio_follower(self, tid: str, station_id: str) -> bool:
        """True if a higher-priority, later-arriving same-direction train is
        pending at this station (the yield case)."""
        me = self._journeys.get(tid)
        if me is None:
            return False
        train = self.cfg.trains[tid]
        for other, j in self._journeys.items():
            if other == tid or j.is_done or j.current_station != station_id:
                continue
            other_train = self.cfg.trains[other]
            if other_train.direction != train.direction:
                continue
            if (other_train.priority < train.priority
                    and j.arrival_at_current > me.arrival_at_current + 1e-6):
                return True
        return False

    def _compute_yield_tail_charges(self, grants: List[GrantResult],
                                    pre: Dict[str, dict]) -> Dict[str, float]:
        """Avoidable departure delay for yield-relevant trains.

        Only trains that were ahead of a higher-priority, later-arriving
        same-direction train at the pre-grant station are charged.  The charge
        is the part of W_out beyond max(planned_dep, arrival) and beyond
        YIELD_TAIL_THRESHOLD; reward.py prices it softly (YIELD_TAIL_COEF).
        """
        charges: Dict[str, float] = {}
        items = list(pre.values())
        for g in grants:
            p = pre.get(g.train_id)
            if p is None or p["next"] is None:
                continue
            has_higher = any(
                q is not p
                and q["station"] == p["station"]
                and q["direction"] == p["direction"]
                and q["priority"] < p["priority"]
                and q["arrival"] > p["arrival"] + 1e-6
                for q in items)
            if not has_higher:
                continue
            base = max(p["planned_dep"], p["arrival"])
            extra = max(0.0, g.W_out[0] - base)
            if extra > YIELD_TAIL_THRESHOLD:
                charges[g.train_id] = extra - YIELD_TAIL_THRESHOLD
        return charges

    def _control_relevant_ids(self, ids: List[str],
                              detail: Dict[str, List[dict]]) -> List[str]:
        """Trains whose decision is coordination-relevant (for gated control).

        ``base``: a pass-through-at-this-station opportunity, or a train ahead
        of a higher-priority follower at the same station/direction (the
        potential yield case).

        ``plus``: base + any train whose next corridor is closed + any train
        that shares its station/direction with another pending train (any
        priority).  This raises policy coverage from ~24% to ~85% without
        handing every routine decision to the policy.

        ``full``: every active train (100%).
        """
        mode = getattr(self, "_control_coverage", "base")
        if mode == "full":
            return sorted(ids)
        relevant = set()
        for tid in ids:
            j = self._journeys.get(tid)
            if j is None or j.is_done:
                continue
            st = j.current_station
            train = self.cfg.trains[tid]
            if mode == "plus":
                # Closure at this departure: the train may need to yield/plan
                # around the reopening even if no follower is visible yet.
                nxt = j.next_station
                if nxt is not None:
                    win = self._corridor_closures.get(f"{st}_to_{nxt}")
                    if win is not None and win[1] > self._t_now + 1e-6:
                        relevant.add(tid)
                # Any co-pending same-direction train = potential reordering.
                if any(x["tid"] != tid and x["direction"] == train.direction
                       for x in detail.get(st, [])):
                    relevant.add(tid)
            if is_station_pass(train, st):
                relevant.add(tid)
                continue
            me = next((x for x in detail.get(st, []) if x["tid"] == tid), None)
            if me is None:
                continue
            for x in detail.get(st, []):
                if x["tid"] == tid or x["direction"] != train.direction:
                    continue
                if (x["priority"] < train.priority
                        and x["arrival"] > me["arrival"] + 1e-6):
                    relevant.add(tid)
                    break
        return sorted(relevant)

    def _capture_pre_grant(self, grants: List[GrantResult]) -> Dict[str, dict]:
        """Snapshot each granted train's station state BEFORE advancing."""
        info: Dict[str, dict] = {}
        for g in grants:
            j = self._journeys[g.train_id]
            if j.is_done:
                continue
            train = self.cfg.trains[g.train_id]
            st = j.current_station
            plan = train.planned.get(st)
            info[g.train_id] = {
                "station": st,
                "next": j.next_station,
                "direction": train.direction,
                "arrival": j.arrival_at_current,
                "priority": train.priority,
                "is_pass": is_station_pass(train, st),
                "planned_dep": plan.dep if plan else 0.0,
            }
        return info

    def _detect_resequencing(self, grants: List[GrantResult],
                             pre: Dict[str, dict],
                             retried: set | None = None,
                             requests: List[WindowRequest] | None = None
                             ) -> List[dict]:
        """Detect station-level overtakes and yields for this step.

        Overtake: i arrived earlier but departs later than j (i is passed).
        Yield: lower-priority i (ahead/equal arrival) delayed beyond its plan
        by >5 s while higher-priority j departs no later than i.  Both are
        policy-independent and usable for learned/FCFS/heuristic.
        """
        groups: Dict[tuple, List[tuple]] = {}
        for g in grants:
            p = pre.get(g.train_id)
            if p is None:
                continue
            key = (p["station"], p["next"], p["direction"])
            groups.setdefault(key, []).append((g, p))

        events: List[dict] = []
        yielded: set = set()
        retried = retried or set()
        # Requested (pre-resolver) departure times: lets eval separate
        # "policy chose this order" from "resolver changed the order".
        req_w = ({r.train_id: r.W_out[0] for r in requests}
                 if requests is not None else {})
        for (st, nxt, direction), items in groups.items():
            # Terminal arrivals are not corridor re-sequencing events.
            if nxt is None:
                continue
            for g_i, p_i in items:
                for g_j, p_j in items:
                    if g_i.train_id == g_j.train_id:
                        continue
                    base = {
                        "station": st, "next": nxt, "direction": direction,
                        "overtaken": g_i.train_id, "overtaker": g_j.train_id,
                        "arr_overtaken": p_i["arrival"],
                        "arr_overtaker": p_j["arrival"],
                        "dep_overtaken": g_i.W_out[0],
                        "dep_overtaker": g_j.W_out[0],
                        "prio_overtaken": p_i["priority"],
                        "prio_overtaker": p_j["priority"],
                        "pass_overtaken": p_i["is_pass"],
                        "pass_overtaker": p_j["is_pass"],
                        "yield_delay": g_i.W_out[0] - p_i["planned_dep"],
                        "late_before_dep":
                            p_i["arrival"] > p_i["planned_dep"] + 1e-6,
                        "resolver_changed": bool(g_i.fallback_from)
                            or g_i.train_id in retried,
                        "req_dep_overtaken": req_w.get(g_i.train_id),
                        "req_dep_overtaker": req_w.get(g_j.train_id),
                        "order_policy_driven": (
                            None if g_i.train_id not in req_w
                            or g_j.train_id not in req_w
                            else req_w[g_i.train_id] > req_w[g_j.train_id]),
                        "closure_related": (
                            f"{st}_to_{nxt}" in self._corridor_closures
                            and self._t_now <= self._corridor_closures[
                                f"{st}_to_{nxt}"][1]),
                    }
                    if (p_i["arrival"] + 1e-6 < p_j["arrival"]
                            and g_i.W_out[0] > g_j.W_out[0] + 1e-6):
                        events.append({
                            **base, "is_overtake": True, "is_yield": False,
                            "is_priority_overtake":
                                p_j["priority"] < p_i["priority"]})
                    if (p_i["priority"] > p_j["priority"]
                            and p_i["arrival"] <= p_j["arrival"] + 1e-6
                            and g_i.W_out[0] > p_i["planned_dep"] + 5.0
                            and g_j.W_out[0] <= g_i.W_out[0] + 1e-6
                            and g_i.train_id not in yielded):
                        events.append({
                            **base, "is_overtake": False, "is_yield": True,
                            "is_priority_overtake": False,
                            "is_deliberate_yield":
                                (not base["late_before_dep"]
                                 and not base["resolver_changed"])})
                        yielded.add(g_i.train_id)
        return events

    def _closure_hold_until(self, train_id: str, station_id: str) -> Optional[float]:
        """Reopening time t1 if this train's next corridor is closed, else None.

        A closure blocks NEW ENTRIES into a corridor during [t0, t1].  A train
        is held only when its departure could fall in the window: either the
        closure has already started, or its planned departure is at/after t0.
        Trains planned before t0 may still depart early and clear the section.
        """
        if not self._corridor_closures:
            return None
        journey = self._journeys.get(train_id)
        if journey is None or journey.next_station is None:
            return None
        cid = f"{station_id}_to_{journey.next_station}"
        win = self._corridor_closures.get(cid)
        if win is None:
            return None
        t0, t1 = win
        if self._t_now >= t1 - 1e-6:
            return None
        plan = self.cfg.trains[train_id].planned.get(station_id)
        planned_dep = plan.dep if plan is not None else self._t_now
        if self._t_now >= t0 - 1e-6 or planned_dep >= t0 - 1e-6:
            return t1
        return None

    def _get_effective_bounds(self, train_id: str, station_id: str, t_now: float):
        """Get scheduling bounds.

        Origin stations: use precomputed timetable bounds.
        Intermediate stations: shift to actual arrival time, with dep_max
        allowing scheduling from the later of actual arrival and current time.
        A corridor closure adds a hard departure floor (t1 - dwell) so no train
        enters the closed section; the mask enforces it like any Layer-1 bound.
        """
        train = self.cfg.trains[train_id]
        journey = self._journeys[train_id]
        orig_bounds = train.bounds.get(station_id)
        hold_until = self._closure_hold_until(train_id, station_id)

        if journey.current_idx == 0 or orig_bounds is None:
            # At origin, arr_min should only shift when the train is actually
            # delayed (arrival > planned departure).  A non-delayed train starts
            # at the platform at t=0 and can depart on schedule — dwell_min is
            # satisfied by the planned dwell (planned_dep - planned_arr).
            if orig_bounds is not None:
                actual_arrival = journey.arrival_at_current
                planned_dep = train.planned[station_id].dep
                dwell = orig_bounds.dwell_min
                slack = self.cfg.time_generation.get("dep_max_slack_sec", 120)
                if hold_until is not None:
                    arr_min = max(orig_bounds.arr_min, hold_until - dwell)
                    dep_max = max(orig_bounds.dep_max,
                                  hold_until + dwell + slack)
                    # `arrival_at_current` is a planned_dep placeholder at reset;
                    # only a genuine (later) delay shifts the envelope.
                    if actual_arrival > planned_dep + 1e-6:
                        arr_min = max(arr_min, actual_arrival)
                        dep_max = max(dep_max, actual_arrival + dwell + slack)
                    return TrainBound(
                        arr_min=arr_min, dep_max=dep_max, dwell_min=dwell)
                if actual_arrival > planned_dep + 1e-6:
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
        if hold_until is not None:
            ref_time = max(ref_time, hold_until - dwell)
        physical_overhead = TRAVEL_TIME_TO_THROAT + 2 * THROAT_CLEARANCE_TIME
        physical_min_dep = ref_time + physical_overhead + dwell
        dep_max = max(ref_time + dwell + slack, physical_min_dep)
        if hold_until is not None:
            # The exogenous closure must not make the envelope infeasible.
            dep_max = max(dep_max, hold_until + dwell + slack)
        return TrainBound(
            arr_min=ref_time,
            dep_max=dep_max,
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

            # Yield-amplitude cap: when this train is ahead of a higher-priority
            # follower, do not let the policy push it out by +30/+50.  The cap
            # is only applied if a cheaper entry remains legal.
            if self._yield_entry_cap and self._has_higher_prio_follower(tid, st):
                cheap = [i for i, c in enumerate(ENTRY_CHOICES)
                         if c in ("earliest", "plan", "plan+10", "plan+20")]
                if any(e_mask[i] for i in cheap):
                    e_mask = [e_mask[i] if i in cheap else False
                              for i in range(n_entry)]

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
