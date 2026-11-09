"""Disturbance module — delay injection and deadlock threshold.

Two disturbance modes:
  - per_train: independent random delays per train (legacy)
  - bunched: infrastructure failure — all trains in a time window at one
    origin station are held until the window clears, creating a bunch of
    trains that must navigate the downstream corridor together.

Layer 2 injects disturbances at initialisation time.  Delayed trains
propagate delay to downstream stations via _get_effective_bounds shift.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from ..experiment.scenario_loader import route_stations

DEADLOCK_THRESHOLD = 15  # global NO_LEGAL_ACTION consecutive steps

# Corridor CLOSURE defaults (区间中断): no new entries during [t0, t1].
CLOSURE_PROB = 0.6
# Hold length for the targeted leader.  Kept short so the window isolates the
# designed meet instead of holding every train planned in a long span (a long
# window makes the priority tie-break do the reordering, which teaches nothing).
CLOSURE_DURATION_RANGE = (10.0, 30.0)
CLOSURE_MIN_GAP_SEC = 15.0      # leader/follower planned departure separation
CLOSURE_FOLLOWER_MARGIN_SEC = 5.0  # keep t1 before follower's planned entry

# Corridor speed-restriction incident defaults (evidence-based, see plan).
INCIDENT_PROB = 0.6
INCIDENT_DURATION_RANGE = (100, 200)
INCIDENT_FACTOR_RANGE = (0.15, 0.4)   # 大暴雨(45km/h=0.15x) .. 暴雨(120km/h=0.4x)
INCIDENT_SAFETY_FACTOR = 1.5          # wet-rail braking distance — ESTIMATED
INCIDENT_MIN_TRAINS = 2               # window must overlap >= this many transits
INCIDENT_MAX_ADDED_DELAY = 200.0      # calibration cap: keep delay recoverable


# ---------------------------------------------------------------------------
# per-train random delay (legacy)
# ---------------------------------------------------------------------------

INITIAL_DELAY_PROB = 0.5
INITIAL_DELAY_RANGE = (30, 180)


def inject_initial_delays(
    journeys: Dict,
    trains: Dict,
    rng: random.Random,
    prob: float = INITIAL_DELAY_PROB,
    delay_range: Tuple[float, float] = INITIAL_DELAY_RANGE,
) -> None:
    """Randomly delay a subset of trains at their origin station."""
    for tid, j in journeys.items():
        if j.is_done:
            continue
        if rng.random() < prob:
            delay = rng.uniform(*delay_range)
            j.arrival_at_current += delay


# ---------------------------------------------------------------------------
# bunched infrastructure delay
# ---------------------------------------------------------------------------

BUNCH_WINDOW_SPAN = 40   # seconds — width of the disruption window
BUNCH_DELAY_MIN = 60     # minimum forced delay after window clears
BUNCH_DELAY_MAX = 120


def inject_bunched_delay(
    journeys: Dict,
    trains: Dict,
    rng: random.Random,
    origin_stations: List[str] | None = None,
    window_span: float = BUNCH_WINDOW_SPAN,
    delay_min: float = BUNCH_DELAY_MIN,
    delay_max: float = BUNCH_DELAY_MAX,
) -> None:
    """Infrastructure failure at a random origin station.

    1. Picks a random origin station from *origin_stations*
    2. Collects all trains departing from that station
    3. Randomly selects a time window [t0, t0 + window_span] covering ≥ 2 trains
    4. All trains in the window are delayed to depart at
       max(planned_dep, window_end) + uniform(delay_min, delay_max)

    This creates genuine corridor bunching — multiple trains released
    close together must navigate headway constraints downstream.
    """
    if origin_stations is None:
        origin_stations = ["A", "E"]

    # Collect trains by origin station with their planned departure
    by_origin: Dict[str, List[Tuple[str, float]]] = {}
    for tid, train in trains.items():
        origin = train.origin_station
        if origin not in origin_stations:
            continue
        j = journeys.get(tid)
        if j is None or j.is_done:
            continue
        planned_dep = train.planned.get(origin)
        if planned_dep is None:
            continue
        by_origin.setdefault(origin, []).append((tid, planned_dep.dep))

    if not by_origin:
        return

    # Pick a random origin station
    chosen = rng.choice(list(by_origin.keys()))
    deps = sorted(by_origin[chosen], key=lambda x: x[1])

    if len(deps) < 2:
        return

    # Pick a random start time that covers at least 2 trains
    t_min = deps[0][1]
    t_max = deps[-1][1]
    if t_max - t_min < 5:
        return

    t0 = rng.uniform(t_min, max(t_min, t_max - window_span * 0.5))
    t1 = t0 + window_span
    bunched = [(tid, d) for tid, d in deps if t0 <= d <= t1]

    if len(bunched) < 2:
        return

    # Delay all bunched trains
    delay = rng.uniform(delay_min, delay_max)
    new_dep = t1 + delay
    for tid, planned_d in bunched:
        j = journeys[tid]
        # Shift arrival_at_current from planned_dep to the new departure time
        extra = new_dep - planned_d
        j.arrival_at_current = max(j.arrival_at_current, new_dep)


# ---------------------------------------------------------------------------
# corridor speed-restriction incident (coordination-rich disturbance)
# ---------------------------------------------------------------------------


def _corridor_transit_windows(cfg) -> Dict[str, List[Tuple[float, float]]]:
    """per corridor_id -> list of (entry, exit) planned transit windows (s).

    Entry = planned departure from the source station; exit = entry + the
    configured corridor runtime.  Placeholder legs (planned arr/dep == 0
    outside a train's real route) are ignored.
    """
    rt = cfg.time_generation.get("corridor_runtime_sec", {})
    wins: Dict[str, List[Tuple[float, float]]] = {}
    full_up = ["A", "B", "C", "D", "E"]
    full_down = ["E", "D", "C", "B", "A"]
    for tid, t in cfg.trains.items():
        route = full_up if t.direction == "up" else full_down
        stops = [s for s in route
                 if s in t.planned and (t.planned[s].dep > 0 or t.planned[s].arr > 0)]
        for i in range(len(stops) - 1):
            a, b = stops[i], stops[i + 1]
            dep = t.planned[a].dep
            if dep <= 0:
                continue
            key = f"{a}_to_{b}"
            wins.setdefault(key, []).append((dep, dep + float(rt.get(key, 90.0))))
    return wins


def inject_corridor_incident(cfg, rng: random.Random,
                             severity_scale: float = 1.0) -> Dict[str, Tuple[float, float, float, float]]:
    """Generate a corridor speed-restriction incident, or {}.

    A random corridor gets a temporary speed limit: a moving bottleneck that
    trains crossing the window must slow for (platoon -> re-dispersion).  The
    window is ANCHORED to the corridor's real transit traffic (>= min_trains
    trains overlap), and the speed factor is CALIBRATED so the worst-case added
    delay stays within max_added_delay_sec (recoverable by L3 speed control —
    unlike origin-bunching).

    severity_scale > 1 STRENGTHENS the incident without touching the base config:
    max_added_delay_sec is multiplied (bigger allowed delay), the factor range is
    divided (slower restriction), and the duration is multiplied — so affected
    trains get delayed past dep_max_slack and L2 must genuinely re-dispatch.

    Returns:
        {corridor_id: (start, end, speed_factor, safety_factor)} or {}.
        speed_factor: multiplier on train speed (1.0 = no restriction, < 1 slower).
        safety_factor: multiplier on moving_block_safety_m (wet-rail braking).
    """
    block = cfg.time_generation.get("corridor_incident", {})
    # `enabled` is gated by the CALLER (env's enable_corridor_incident) so config
    # defaults never silently change existing baselines; here we just generate.
    if rng.random() >= block.get("prob", INCIDENT_PROB):
        return {}

    dur_lo, dur_hi = block.get("duration_range", list(INCIDENT_DURATION_RANGE))
    f_lo, f_hi = block.get("factor_range", list(INCIDENT_FACTOR_RANGE))
    safety = block.get("safety_factor", INCIDENT_SAFETY_FACTOR)
    min_trains = block.get("min_trains", INCIDENT_MIN_TRAINS)
    max_added = block.get("max_added_delay_sec", INCIDENT_MAX_ADDED_DELAY)
    # severity_scale: strengthen WITHOUT editing the base config.
    if severity_scale > 1.0:
        max_added = max_added * severity_scale          # more allowed added delay
        f_lo, f_hi = f_lo / severity_scale, f_hi / severity_scale  # slower restriction
        dur_lo, dur_hi = dur_lo * severity_scale, dur_hi * severity_scale

    wins = _corridor_transit_windows(cfg)
    eligible = {cid: v for cid, v in wins.items() if len(v) >= min_trains}
    if not eligible:
        return {}
    # weight corridor choice by traffic (bigger bursts -> bigger platoons)
    corridor_ids = list(eligible.keys())
    weights = [len(eligible[c]) for c in corridor_ids]
    cid = rng.choices(corridor_ids, weights=weights, k=1)[0]
    transits = sorted(eligible[cid], key=lambda w: w[0])

    duration = rng.uniform(dur_lo, dur_hi)
    # start so that the window overlaps >= min_trains transits
    t0 = None
    for i in range(0, len(transits) - min_trains + 1):
        lo = transits[i][0]
        hi = transits[i + min_trains - 1][1]
        if hi - lo <= duration:
            t0 = rng.uniform(max(0.0, hi - duration), lo)
            break
    if t0 is None:
        t0 = transits[0][0]
    end = t0 + duration

    factor = rng.uniform(f_lo, f_hi)
    # calibrate severity so the worst affected train's added delay stays within
    # max_added_delay_sec (recoverable by L3 speed control).  Two regimes:
    #   transit <= duration : a train can be FULLY covered -> delay ~ transit*(1/f-1)
    #   transit >  duration : only partial overlap -> delay ~ duration*(1-f)
    transit_est = min((b - a) for a, b in transits)  # shortest transit (worst full-cover)
    need_full = need_part = 0.0
    if max_added > 0:
        need_full = (1.0 / (1.0 + max_added / transit_est)
                     if transit_est <= duration else 0.0)
        need_part = (1.0 - max_added / duration
                     if duration < transit_est else 0.0)
    factor = max(factor, need_full, need_part)
    factor = min(max(factor, f_lo), 1.0)
    return {cid: (t0, end, factor, safety)}


# ---------------------------------------------------------------------------
# corridor closure incident (区间中断: no new entries during a window)
# ---------------------------------------------------------------------------


def inject_corridor_closure(
    cfg, rng: random.Random, severity_scale: float = 1.0,
) -> Dict[str, Tuple[float, float]]:
    """Generate a corridor closure that creates a catch-up/overtake opportunity.

    A closure blocks NEW ENTRIES into a corridor during ``[t0, t1]``.  Unlike a
    random window, the window is built around a *designed meet*:

        1. a lower-priority leader is planned to depart the source station at
           ``d_L``;
        2. a higher-priority follower is planned to depart at ``d_F > d_L``;
        3. ``t1`` is placed after ``d_L`` but BEFORE the follower's plan entry
           (``d_F - dwell``), so the leader is held to the reopening while the
           follower can still depart at its planned time.

    After t1 both trains are co-pending at the source station and the leader has
    an earlier feasible slot: FCFS sends the leader first, so only a deliberate
    yield lets the follower pass there.  This is the learnable re-sequencing
    case; a window ending after both planned departures would let the existing
    (W_out, priority, id) tie-break reorder by rule and teach nothing.

    Returns ``{corridor_id: (t0, t1)}`` or ``{}`` when no qualifying meet exists.
    Semantics: trains already inside the corridor continue (no mid-section stop).
    """
    block = cfg.time_generation.get("corridor_closure", {})
    # `enabled` is gated by the caller (env's enable_corridor_closure), same as
    # the speed-restriction incident, so config defaults never change baselines.
    if rng.random() >= block.get("prob", CLOSURE_PROB):
        return {}

    dur_lo, dur_hi = block.get("duration_range", list(CLOSURE_DURATION_RANGE))
    min_gap = block.get("min_gap_sec", CLOSURE_MIN_GAP_SEC)
    margin = block.get("follower_margin_sec", CLOSURE_FOLLOWER_MARGIN_SEC)
    if severity_scale > 1.0:
        dur_lo, dur_hi = dur_lo * severity_scale, dur_hi * severity_scale

    # corridor -> [(planned_dep, priority, train_id, dwell_min), ...]
    events: Dict[str, List[Tuple[float, int, str, float]]] = {}
    for tid, t in cfg.trains.items():
        route = route_stations(t, cfg.corridors)
        for a, b in zip(route, route[1:]):
            pa = t.planned.get(a)
            pb = t.planned.get(b)
            if pa is None or pb is None or pa.dep is None or pa.dep <= 0:
                continue
            # Same key convention as FiveStationEnv's L3 corridor_key and
            # time_generation.corridor_runtime_sec ("A_to_B"), not the
            # configs' corridor graph IDs ("A_B_up").
            cid = f"{a}_to_{b}"
            bounds = t.bounds.get(a)
            dwell = bounds.dwell_min if bounds else 0.0
            events.setdefault(cid, []).append(
                (float(pa.dep), int(t.priority), tid, float(dwell)))

    candidates = []
    for cid, evs in events.items():
        evs = sorted(evs, key=lambda e: (e[0], e[1], e[2]))
        for i in range(len(evs)):
            d_l, p_l, tid_l, _ = evs[i]
            for j in range(i + 1, len(evs)):
                d_f, p_f, tid_f, dwell_f = evs[j]
                if p_l <= p_f:            # leader must be lower priority
                    continue
                if d_f - d_l < min_gap:   # enough catch-up separation
                    continue
                plan_entry_f = d_f - dwell_f
                t1_lo = d_l + 1.0
                t1_hi = plan_entry_f - margin
                if t1_hi - t1_lo < 10.0:  # no room to hold the leader
                    continue
                candidates.append(
                    (cid, t1_lo, t1_hi, d_l, d_f, tid_l, tid_f))

    if not candidates:
        return {}

    cid, t1_lo, t1_hi, d_l, d_f, tid_l, tid_f = rng.choice(candidates)
    # Start just before the leader's planned departure and hold for a short
    # window ending before the follower's plan entry.  This isolates the
    # designed meet: few other trains fall inside [t0, t1], so the leader must
    # YIELD to let the follower pass (rather than a priority tie-break doing it).
    hold = rng.uniform(dur_lo, dur_hi)
    t0 = max(0.0, d_l - 5.0)
    t1 = min(t1_hi, d_l + hold)
    if t1 <= d_l + 1.0:
        return {}
    return {cid: (t0, t1)}

