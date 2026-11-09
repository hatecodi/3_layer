"""Delay-dissipation time (DDT) — the TEMPORAL dimension of recovery.

The existing eval metrics answer "how bad" (tail severity) and "how good on
average" (mean delay, on-time rate).  Neither answers "how fast does the system
come back" — the quantity a resilience claim actually rests on.  DDT is that
third leg.

Definition (pre-registered cell: ``theta = 30`` sim-s, ``k = 3`` arrival events):

    After a disturbance, from the anchor ``t_rec_start`` onward, the
    dissipation time is the elapsed time until ``k`` CONSECUTIVE arrival events
    all satisfy ``|arr_delay| <= theta``.  The clock stops at the CONFIRMING
    (k-th) event; if no such run occurs before the episode ends, the episode is
    RIGHT-CENSORED.

Three deliberate choices, each of which is a bug if reversed:

1. ``|arr_delay|``, not signed delay.  ``five_station_env.py`` already records
   signed delay and ``l3_sim.py:580`` fixed a bug where systematic EARLY arrival
   masked L3's recovery value.  A signed threshold would let a policy that runs
   everything early fake a zero dissipation time.
2. The anchor is the END of the disturbance, never its onset.  A train covered by
   a restriction window is delayed by physics regardless of policy, so anchoring
   at onset folds the disturbance DURATION into the metric instead of measuring
   the response to it.
3. ``theta = 30`` sim-s is deliberately NOT ``ONTIME_THRESHOLD_SEC = 15`` — at 15
   the metric would be nearly a restatement of the on-time rate.  30 is ~12.5
   real min, still well below the incident cap (``max_added_delay_sec = 80``), so
   it is attainable.

Reporting rules: a time-to-event statistic is heavy-tailed and censored, so
report MEDIAN + IQR + the censoring rate — never a mean.  ``theta`` and ``k`` are
chosen, not derived, so the claim to make is RANK STABILITY across the
``DDT_GRID_THETA x DDT_GRID_K`` grid, not the absolute values.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

# Pre-registered cell (see module docstring).
DDT_THETA_SEC = 30.0
DDT_K = 3

# Sensitivity grid — the metric's parameters are chosen, so the paper claim must
# be "the method ordering is stable across this grid", not one arbitrary cell.
DDT_GRID_THETA = (15.0, 30.0, 45.0)
DDT_GRID_K = (2, 3, 5)

def _n_after(events: Sequence[Dict], t_rec_start: float) -> int:
    return sum(1 for e in events if e["actual_arrival"] >= t_rec_start)


def dissipation_time(
    events: Sequence[Dict],
    t_rec_start: float,
    theta: float = DDT_THETA_SEC,
    k: int = DDT_K,
    value_of: Optional[Callable[[Dict], float]] = None,
) -> Optional[float]:
    """Seconds after ``t_rec_start`` at which the run of ``k`` compliant arrivals
    is confirmed, or ``None`` if the episode never recovers (right-censored).

    Events are ordered by REALISED arrival time (the realised stream is what
    "the system is back to normal" means), and the run counter resets on every
    non-compliant arrival.  ``value_of`` defaults to signed ``arr_delay``; the
    paired-excess variant passes a selector for the ``_excess`` field.
    """
    if k <= 0:
        return 0.0
    value_of = value_of or (lambda e: e["arr_delay"])
    seq = sorted((e for e in events if e["actual_arrival"] >= t_rec_start),
                 key=lambda e: e["actual_arrival"])
    run = 0
    for e in seq:
        if abs(value_of(e)) <= theta:
            run += 1
            if run >= k:
                return e["actual_arrival"] - t_rec_start
        else:
            run = 0
    return None


def recovery_stats(
    eps: Sequence[Dict],
    theta: float = DDT_THETA_SEC,
    k: int = DDT_K,
    value_of: Optional[Callable[[Dict], float]] = None,
) -> Dict[str, float]:
    """Median / IQR / censoring of DDT over the DISTURBED episodes.

    Episodes without an anchor (``t_rec_start is None`` — clean timetable, or an
    episode the disturbance never touched) are excluded from ``n_eps_dist``
    rather than counted as censored.  ``short_pct`` isolates the structural
    censoring cause (fewer than ``k`` arrivals remained after the anchor, so no
    run could be confirmed even by a perfect policy).
    """
    dts: List[float] = []
    n_dist = n_cens = n_short = 0
    for r in eps:
        t0 = r.get("t_rec_start")
        if t0 is None:
            continue
        n_dist += 1
        ev = r["arr_events"]
        if _n_after(ev, t0) < k:
            n_short += 1
            n_cens += 1
            continue
        dt = dissipation_time(ev, t0, theta, k, value_of=value_of)
        if dt is None:
            n_cens += 1
        else:
            dts.append(dt)
    dts.sort()
    n = len(dts)
    # Quantile convention matches _dist() in eval_protocol.py (index n//2, not an
    # interpolating quantile) so the DDT table is consistent with delay table.
    return {
        "n_eps_dist": n_dist,
        "n_rec": n,
        "censor_pct": (100.0 * n_cens / n_dist) if n_dist else float("nan"),
        "short_pct": (100.0 * n_short / n_dist) if n_dist else float("nan"),
        "med": dts[n // 2] if n else float("nan"),
        "p25": dts[n // 4] if n else float("nan"),
        "p75": dts[(3 * n) // 4] if n else float("nan"),
    }


def recovery_grid(
    eps: Sequence[Dict],
    value_of: Optional[Callable[[Dict], float]] = None,
) -> List[Dict[str, float]]:
    """Long-format rows over ``DDT_GRID_THETA x DDT_GRID_K`` for the paper's
    parameter-sensitivity table (rank stability, not absolute values)."""
    rows = []
    for theta in DDT_GRID_THETA:
        for k in DDT_GRID_K:
            rows.append({"theta": theta, "k": k, **recovery_stats(eps, theta, k, value_of)})
    return rows


def floor_index(ref_eps: Sequence[Dict]) -> Dict[Tuple, float]:
    """``(seed, train_id, corridor) -> arr_delay`` for a reference execution layer.

    Deliberately NOT keyed by event order: the two execution layers realise
    different arrival times, so the arrival ORDER can differ while the under-
    lying train x corridor leg still corresponds one-to-one.
    """
    idx: Dict[Tuple, float] = {}
    for r in ref_eps:
        for ev in r["arr_events"]:
            idx[(r.get("seed"), ev["train_id"], ev["corridor"])] = ev["arr_delay"]
    return idx


def with_floor(
    eps: Sequence[Dict], idx: Dict[Tuple, float]
) -> Tuple[List[Dict], float]:
    """Attach ``_excess = arr_delay - floor`` to each event, PAIRED by
    (seed, train_id, corridor).

    Needed for the incident arm: the restriction adds delay that L3 cannot fully
    recover (cap ``max_added_delay_sec``), so an ABSOLUTE threshold is censored by
    construction — the coordination value is the excess over the uncoordinated
    floor.  Events whose counterpart is missing (a different grant combination
    under a different speed profile) are dropped; the returned coverage lets the
    caller report how much of the stream was actually paired.
    """
    out: List[Dict] = []
    paired = total = 0
    for r in eps:
        evs = []
        for ev in r["arr_events"]:
            total += 1
            f = idx.get((r.get("seed"), ev["train_id"], ev["corridor"]))
            if f is None:
                continue
            paired += 1
            evs.append({**ev, "_excess": ev["arr_delay"] - f})
        out.append({**r, "arr_events": evs})
    coverage = (100.0 * paired / total) if total else float("nan")
    return out, coverage
