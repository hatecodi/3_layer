"""
Phase 2: Resource calendar and deterministic conflict resolver.

Tracks occupation windows for every physical resource (platforms, throat
routes, corridors).  When multiple trains submit overlapping time-window
requests, a priority-weighted deterministic resolver grants one request and
assigns the earliest feasible fallback to the loser.

Key property:  all granted windows respect Layer 1 hard bounds and physical
resource-exclusivity.  The resolver is fixed (not learned); MAPPO learns
*when to yield* rather than the resolution rule itself.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# rebuild callback: (train_id, platform, shift_sec, base_request) -> WindowRequest or None
RebuildFn = Callable[[str, str, float, "WindowRequest"], Optional["WindowRequest"]]


# ---------------------------------------------------------------------------
# data types
# ---------------------------------------------------------------------------

@dataclass
class WindowRequest:
    """A train's time-window request for a specific platform and route."""
    train_id: str
    platform: str
    W_in: Tuple[float, float]       # (t_in_start, t_in_end)
    W_plat: Tuple[float, float]     # (t_plat_start, t_plat_end)
    W_out: Tuple[float, float]      # (t_out_start, t_out_end)
    W_corridor: Optional[Tuple[float, float]] = None  # corridor transit
    priority: int = 99
    delay: float = 0.0
    # resource IDs for conflict checking (set after construction)
    _resource_windows: Dict[str, Tuple[float, float]] = field(
        default_factory=dict, init=False, repr=False)


@dataclass
class OccupancyWindow:
    train_id: str
    start: float
    end: float


# ---------------------------------------------------------------------------
# Resource calendar
# ---------------------------------------------------------------------------

class ResourceCalendar:
    """Tracks granted occupation windows per resource."""

    def __init__(self, min_gap_sec: float = 5.0):
        self._windows: Dict[str, List[OccupancyWindow]] = defaultdict(list)
        self._min_gap = min_gap_sec

    # -- query ---------------------------------------------------------

    def is_free(self, resource_id: str, start: float, end: float) -> bool:
        """Check whether [start, end) conflicts with any existing booking."""
        for w in self._windows.get(resource_id, []):
            if _overlap((start, end), (w.start, w.end), self._min_gap):
                return False
        return True

    def earliest_free(self, resource_id: str, after: float,
                      duration: float) -> float:
        """Return the earliest start time >= *after* for a window of
        *duration* seconds on this resource."""
        t = after
        # simple linear scan — sufficient for small number of bookings
        booked = sorted(self._windows.get(resource_id, []), key=lambda w: w.start)
        for w in booked:
            if t + duration + self._min_gap <= w.start:
                return t
            t = max(t, w.end + self._min_gap)
        return t

    # -- mutate --------------------------------------------------------

    def book(self, resource_id: str, train_id: str,
             start: float, end: float) -> None:
        self._windows[resource_id].append(
            OccupancyWindow(train_id, start, end))

    def clear(self) -> None:
        self._windows.clear()

    def dump(self) -> Dict[str, List[Tuple[str, float, float]]]:
        return {
            rid: [(w.train_id, w.start, w.end) for w in sorted(ws, key=lambda x: x.start)]
            for rid, ws in self._windows.items()
        }


# ---------------------------------------------------------------------------
# conflict detection
# ---------------------------------------------------------------------------

def _overlap(a: Tuple[float, float], b: Tuple[float, float],
             gap: float = 0.0) -> bool:
    """True if intervals [a0,a1) and [b0,b1) overlap (inclusive of gap)."""
    return a[0] < b[1] + gap and b[0] < a[1] + gap


def requests_conflict(r1: WindowRequest, r2: WindowRequest,
                      min_gap: float = 5.0) -> bool:
    """Check whether two requests share any resource with overlapping windows."""
    shared = set(r1._resource_windows.keys()) & set(r2._resource_windows.keys())
    for rid in shared:
        if _overlap(r1._resource_windows[rid], r2._resource_windows[rid], min_gap):
            return True
    return False


def build_conflict_graph(requests: List[WindowRequest],
                         min_gap: float = 5.0) -> Dict[str, Set[str]]:
    """Return adjacency {train_id: {neighbor_train_ids}} for conflicting requests."""
    graph: Dict[str, Set[str]] = defaultdict(set)
    for i, r1 in enumerate(requests):
        for r2 in requests[i + 1:]:
            if requests_conflict(r1, r2, min_gap):
                graph[r1.train_id].add(r2.train_id)
                graph[r2.train_id].add(r1.train_id)
    return dict(graph)


# ---------------------------------------------------------------------------
# priority-weighted deterministic resolver
# ---------------------------------------------------------------------------

def _resolve_score(req: WindowRequest) -> Tuple[int, float, str]:
    """Lower is better.  1st: priority class, 2nd: -delay (more delay = protected),
    3rd: train_id (deterministic tiebreak)."""
    return (req.priority, -req.delay, req.train_id)


def resolve_pair(r1: WindowRequest, r2: WindowRequest) -> Tuple[
    WindowRequest, WindowRequest]:
    """Return (winner, loser).  Deterministic priority-weighted comparison."""
    if _resolve_score(r1) <= _resolve_score(r2):
        return r1, r2
    return r2, r1


# ---------------------------------------------------------------------------
# fallback assignment  (Step 0: rebuilt with proper resource-window support)
# ---------------------------------------------------------------------------

def _compute_shift_for_fallback(
    req: WindowRequest,
    calendar: ResourceCalendar,
    min_gap: float,
) -> float:
    """Find the minimum time shift needed so *req*'s resource windows are
    all free on *calendar*.  Returns 0 if already free."""
    latest_end = 0.0
    for rid, (s, e) in req._resource_windows.items():
        for w in calendar._windows.get(rid, []):
            if _overlap((s, e), (w.start, w.end), min_gap):
                if w.end + min_gap > latest_end:
                    latest_end = w.end + min_gap
    if latest_end <= req.W_in[0]:
        return 0.0
    return latest_end - req.W_in[0]


def assign_fallback(
    loser: WindowRequest,
    candidate_platforms: List[str],
    rebuild_fn: RebuildFn,
    calendar: ResourceCalendar,
    min_gap: float = 5.0,
) -> Optional[WindowRequest]:
    """Assign the earliest feasible window on a different (or later) slot.

    Uses *rebuild_fn* to create fresh WindowRequests with correct
    resource IDs for each candidate platform.  Falls back to a delayed
    slot on the original platform only after all alternatives are tried.

    Returns None if no bounded-feasible slot exists.
    """
    train_id = loser.train_id
    original_platform = loser.platform

    def _first_feasible(platform: str) -> Optional[WindowRequest]:
        """Iteratively rebuild until every resource window is free.

        A single shift can create a new conflict on a later resource window
        (for example, avoiding throat-in may move throat-out into another
        train's window). Iterate a bounded number of times because each pass
        moves forward to the end of an existing booking.
        """
        shift = 0.0
        for _ in range(20):
            req = rebuild_fn(train_id, platform, shift, loser)
            if req is None:
                return None
            extra = _compute_shift_for_fallback(req, calendar, min_gap)
            if extra <= 1e-9:
                req.delay = loser.delay + shift
                req.priority = loser.priority
                return req
            shift += extra
        return None

    # 1) Try other platforms first.
    for plat in candidate_platforms:
        if plat == original_platform:
            continue
        shifted = _first_feasible(plat)
        if shifted is not None:
            return shifted

    # 2) Delay on the same platform
    return _first_feasible(original_platform)


# ---------------------------------------------------------------------------
# batch resolver (resolve all requests -> granted windows)
# ---------------------------------------------------------------------------

@dataclass
class GrantResult:
    train_id: str
    platform: str
    W_in: Tuple[float, float]
    W_plat: Tuple[float, float]
    W_out: Tuple[float, float]
    winner: bool
    fallback_from: Optional[str] = None


def resolve_all(
    requests: List[WindowRequest],
    candidate_platforms: Dict[str, List[str]],
    rebuild_fn: RebuildFn,
    min_gap: float = 5.0,
) -> List[GrantResult]:
    """Batch-resolve all requests.

    1. Sort by priority score (best first).
    2. Try to grant each request. If any resource conflicts with a
       previously granted window, resolve the conflict and assign
       a fallback to the loser via *rebuild_fn*.
    3. Book granted windows in the calendar.

    Returns ordered list of GrantResult (same length as requests).
    """
    calendar = ResourceCalendar(min_gap)
    results: Dict[str, GrantResult] = {}
    sorted_reqs = sorted(requests, key=_resolve_score)

    for req in sorted_reqs:
        if req.train_id in results:
            continue

        # Check conflicts with already-granted bookings
        conflicts: List[str] = []
        for rid, (s, e) in req._resource_windows.items():
            if not calendar.is_free(rid, s, e):
                for w in calendar._windows.get(rid, []):
                    if _overlap((s, e), (w.start, w.end), min_gap):
                        if w.train_id not in conflicts:
                            conflicts.append(w.train_id)

        if not conflicts:
            _grant(req, calendar, results)
            continue

        # Resolve: already-granted trains won on priority, current request loses.
        granted_fallback = False
        for cid in conflicts:
            if cid in results:
                plat_list = candidate_platforms.get(req.train_id, [req.platform])
                fallback = assign_fallback(
                    req, plat_list, rebuild_fn, calendar, min_gap)
                if fallback is not None:
                    _grant(fallback, calendar, results,
                           winner=False, fallback_from=req.platform)
                    granted_fallback = True
                    break
        if not granted_fallback:
            # No bounded feasible fallback exists. The train is intentionally
            # left ungranted so the caller can hold/replan it; granting the
            # original request here would reintroduce a hard conflict.
            continue

    return [results[r.train_id] for r in requests if r.train_id in results]


def _grant(req: WindowRequest, calendar: ResourceCalendar,
           results: Dict[str, GrantResult],
           winner: bool = True,
           fallback_from: Optional[str] = None) -> None:
    for rid, (s, e) in req._resource_windows.items():
        calendar.book(rid, req.train_id, s, e)
    results[req.train_id] = GrantResult(
        train_id=req.train_id,
        platform=req.platform,
        W_in=req.W_in,
        W_plat=req.W_plat,
        W_out=req.W_out,
        winner=winner,
        fallback_from=fallback_from,
    )
