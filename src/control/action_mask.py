"""
Phase 2: Action space definition, derive_schedule, and joint-action masking.

Implements the conditional discrete action from the design document:

    a_i = (platform_choice, entry_choice, dwell_extend)

All Layer 1 hard bounds (t_arr_min, t_dep_max, dwell_min, route compatibility,
through-track restriction) are enforced by the action mask — actions violating
any constraint are set to -inf logits before softmax.  The policy never sees
an illegal action.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..experiment.scenario_loader import (
    ScenarioConfig,
    StationConfig,
    TrainTimetable,
)
from .resource_manager import WindowRequest

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

THROAT_CLEARANCE_TIME = 20.0        # seconds to traverse throat
THROUGH_PLATFORM_CLEARANCE_TIME = 5.0   # minimum platform pass-through time
MIN_CLEARANCE_SECONDS = 5.0          # safety gap between resource windows

ENTRY_CHOICES = ["earliest", "plan", "plan+1", "plan+2", "plan+4"]  # minutes
DWELL_EXTEND_CHOICES = [0, 60, 120]  # seconds


# ---------------------------------------------------------------------------
# derived action-space size hint
# ---------------------------------------------------------------------------

@dataclass
class ActionSpec:
    index: int
    platform: str
    entry_choice: str
    dwell_extend: int   # seconds


# ---------------------------------------------------------------------------
# schedule derivation
# ---------------------------------------------------------------------------

def requested_entry_time(
    train: TrainTimetable,
    station_id: str,
    entry_choice: str,
    t_now: float,
    travel_time_to_throat: float,
) -> float:
    """Compute the *requested* inbound-throat entry time for a given choice."""
    earliest_physical = t_now + travel_time_to_throat
    bounds = train.bounds.get(station_id)
    earliest_allowed = max(earliest_physical,
                           bounds.arr_min if bounds else 0.0)

    if entry_choice == "earliest":
        return earliest_allowed

    plan = train.planned.get(station_id)
    planned_arr = plan.arr if plan else 0.0

    if entry_choice == "plan":
        offset_sec = 0.0
    else:
        # "plan+N" where N is minutes
        offset_sec = float(entry_choice.split("+")[1]) * 60.0

    return max(planned_arr + offset_sec, earliest_allowed)


def derive_schedule(
    train: TrainTimetable,
    station_id: str,
    action: Tuple[str, str, int],
    t_now: float,
    travel_time_to_throat: float,
) -> Tuple[float, float, float, float, float, float]:
    """Convert an action to concrete time windows.

    Returns:
        (t_in_start, t_in_end, t_plat_start, t_plat_end,
         t_out_start, t_out_end)
    """
    platform, entry_choice, dwell_extend = action

    t_in_start = requested_entry_time(
        train, station_id, entry_choice, t_now, travel_time_to_throat)
    t_in_end = t_in_start + THROAT_CLEARANCE_TIME

    if train.pass_through:
        dwell = 0.0
        t_plat_start = t_in_end
        t_plat_end = t_plat_start + THROUGH_PLATFORM_CLEARANCE_TIME
    else:
        bounds = train.bounds.get(station_id)
        dwell_min = bounds.dwell_min if bounds else 0.0
        dwell = dwell_min + dwell_extend
        t_plat_start = t_in_end
        t_plat_end = t_plat_start + dwell

    t_out_start = t_plat_end
    t_out_end = t_out_start + THROAT_CLEARANCE_TIME

    return t_in_start, t_in_end, t_plat_start, t_plat_end, t_out_start, t_out_end


# ---------------------------------------------------------------------------
# legality checks
# ---------------------------------------------------------------------------

def is_legal_action(
    train: TrainTimetable,
    station_id: str,
    action: Tuple[str, str, int],
    t_now: float,
    travel_time_to_throat: float,
    cfg: ScenarioConfig,
) -> bool:
    """Check an action against all Layer 1 hard bounds."""
    platform, entry_choice, dwell_extend = action
    bounds = train.bounds.get(station_id)
    if bounds is None:
        return True  # no bounds = no constraints (degenerate)

    station = cfg.stations.get(station_id)
    if station is None:
        return False

    # platform must be in candidate set
    candidate = train.candidate_platforms.get(station_id, [])
    if platform not in candidate:
        return False

    # pass-through must use through-allowed platform
    plat_cfg = next((p for p in station.platforms if p.id == platform), None)
    if plat_cfg is None:
        return False
    if train.pass_through and not plat_cfg.through_allowed:
        return False
    if train.pass_through and dwell_extend != 0:
        return False

    # derive windows
    times = derive_schedule(
        train, station_id, action, t_now, travel_time_to_throat)
    t_in_start, t_in_end, _, _, t_out_start, t_out_end = times

    # hard floor: cannot enter before arr_min
    if t_in_start < bounds.arr_min - 1e-6:
        return False

    # hard ceiling: must start exit before dep_max
    if t_out_start > bounds.dep_max + 1e-6:
        return False

    # dwell floor (enforced by dwell_extend >= 0 and dwell_min base)
    actual_dwell = t_out_start - t_in_end - THROUGH_PLATFORM_CLEARANCE_TIME \
        if train.pass_through else t_out_start - t_in_end
    if actual_dwell < bounds.dwell_min - 1e-6:
        return False

    return True


# ---------------------------------------------------------------------------
# candidate action enumeration + mask
# ---------------------------------------------------------------------------

def candidate_actions(
    train: TrainTimetable,
    station_id: str,
    cfg: ScenarioConfig,
) -> List[ActionSpec]:
    """Enumerate all possible (platform, entry, extend) combinations for a train
    at a given station."""
    actions: List[ActionSpec] = []
    candidate = train.candidate_platforms.get(station_id, [])
    for platform in candidate:
        for entry in ENTRY_CHOICES:
            for dwell_ext in DWELL_EXTEND_CHOICES:
                actions.append(ActionSpec(
                    index=len(actions),
                    platform=platform,
                    entry_choice=entry,
                    dwell_extend=dwell_ext,
                ))
    return actions


# ---------------------------------------------------------------------------
# joint action mask for a set of trains at a decision step
# ---------------------------------------------------------------------------

@dataclass
class MaskedActionSet:
    actions: List[ActionSpec]
    mask: List[bool]                  # True = legal
    status: str                       # "OK" or "NO_LEGAL_ACTION"
    train_id: str


def compute_joint_action_mask(
    train: TrainTimetable,
    station_id: str,
    t_now: float,
    travel_time_to_throat: float,
    cfg: ScenarioConfig,
) -> MaskedActionSet:
    """Build the valid-action mask for one train at one station.

    Returns MaskedActionSet with status="NO_LEGAL_ACTION" if every action is
    illegal.  The caller should trigger a deterministic hold/replan transition
    instead of sampling from the policy.
    """
    actions = candidate_actions(train, station_id, cfg)
    mask = [
        is_legal_action(
            train, station_id,
            (a.platform, a.entry_choice, a.dwell_extend),
            t_now, travel_time_to_throat, cfg,
        )
        for a in actions
    ]

    if not any(mask):
        return MaskedActionSet(actions, mask, "NO_LEGAL_ACTION", train.id)

    return MaskedActionSet(actions, mask, "OK", train.id)


# ---------------------------------------------------------------------------
# action -> WindowRequest (for resource manager)
# ---------------------------------------------------------------------------

def action_to_request(
    train: TrainTimetable,
    station_id: str,
    action: Tuple[str, str, int],
    t_now: float,
    travel_time_to_throat: float,
    current_delay: float,
    cfg: ScenarioConfig,
) -> WindowRequest:
    """Convert a legal action to a WindowRequest with resource windows.

    The resource windows encode which physical resources are occupied
    during each phase.  This is used by the resource manager for
    conflict detection and resolution.
    """
    platform, _, _ = action
    times = derive_schedule(
        train, station_id, action, t_now, travel_time_to_throat)
    t_in_start, t_in_end, t_plat_start, t_plat_end, t_out_start, t_out_end = times

    station = cfg.stations[station_id]
    dir_cfg = station.up if train.direction == "up" else station.down

    # Map platform to its in/out routes
    in_route = next((r for r in dir_cfg.in_routes if r.platform == platform), None)
    out_route = next((r for r in dir_cfg.out_routes if r.platform == platform), None)

    resource_windows: Dict[str, Tuple[float, float]] = {
        # platform occupancy
        f"{station_id}/{platform}": (t_plat_start, t_plat_end),
    }

    # throat-in occupancy (by conflict group)
    if in_route:
        resource_windows[f"{station_id}/{in_route.conflict_group}"] = (
            t_in_start, t_in_end)

    # throat-out occupancy (by conflict group)
    if out_route:
        resource_windows[f"{station_id}/{out_route.conflict_group}"] = (
            t_out_start, t_out_end)

    # Corridor timing is recorded for execution/metrics, but not inserted into
    # _resource_windows. Minimum tracking headway is enforced by the execution
    # layer/controller in this design, while the resolver owns station
    # resources: platform, inbound throat, and outbound throat.
    corridor_id = train.corridor
    corridor_window = None
    if corridor_id:
        corr = cfg.corridors.get(corridor_id)
        if corr:
            corridor_start = t_out_end
            corridor_end = corridor_start + sum(
                float(edge_len(corr, e)) / _corridor_speed(corr)
                for e in corr.edges
            ) if corr.edges else corridor_start + 90.0
            corridor_window = (corridor_start, corridor_end)

    req = WindowRequest(
        train_id=train.id,
        platform=platform,
        W_in=(t_in_start, t_in_end),
        W_plat=(t_plat_start, t_plat_end),
        W_out=(t_out_start, t_out_end),
        W_corridor=corridor_window or (t_out_end, t_out_end + 90.0),
        priority=train.priority,
        delay=current_delay,
    )
    req._resource_windows = resource_windows
    return req


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def edge_len(corridor, edge_id: str) -> float:
    """Return edge length if we know it, else a default based on corridor config."""
    # In a full implementation this would read from the SUMO net.
    # For now, use reasonable defaults based on corridor edge count.
    total = 1200.0  # corridor length per trains reference
    n = max(len(corridor.edges), 1)
    return total / n


def _corridor_speed(corridor) -> float:
    """Return corridor speed (m/s).  Default 30 m/s."""
    return 30.0


# ---------------------------------------------------------------------------
# heuristic policy (Phase 2, step 5)
# ---------------------------------------------------------------------------

def heuristic_action(
    train: TrainTimetable,
    station_id: str,
    t_now: float,
    travel_time_to_throat: float,
    cfg: ScenarioConfig,
) -> Tuple[str, str, int]:
    """Deterministic heuristic: earliest legal entry, first legal platform, min dwell.

    Used as the baseline policy before MAPPO is introduced.
    """
    actions = candidate_actions(train, station_id, cfg)
    for a in actions:
        if is_legal_action(
            train, station_id,
            (a.platform, a.entry_choice, a.dwell_extend),
            t_now, travel_time_to_throat, cfg,
        ):
            return (a.platform, a.entry_choice, a.dwell_extend)
    # No legal action — caller must hold/replan
    raise ValueError(f"No legal action for {train.id} at {station_id}")
