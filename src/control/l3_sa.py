"""SA baseline for L3 speed-curve control.

Simulated annealing over the per-section speed modes (CRUISE/FAST/SLOW) for ONE
corridor transit, optimising the SAME local objective the learned L3 worker is
rewarded on (arrival punctuality, with headway as a hard boundary enforced by
``L3Simulator.section_progress``).  SA has full information (front occupant
profile fixed, all sections considered jointly), so it is the greedy per-transit
local upper bound.  The paper claim is that a globally-coordinated learned L3
can beat this greedy baseline on system-level metrics.
"""
from __future__ import annotations

import math
import random
from typing import Callable, List, Optional, Tuple

from .l3_sim import L3Section, L3Simulator, SpeedMode
from ..env.reward import L3_FAST_PENALTY


def sa_optimize_profile(
    sim: L3Simulator,
    corridor_id: str,
    departure: float,
    priority: int,
    sections: List[L3Section],
    bounds: List[float],
    planned_arrival: float,
    reward_fn: Callable[[float, float], float],
    n_iter: int = 300,
    seed: int = 0,
    t0: float = 10.0,
    t1: float = 0.1,
    incident_window: Optional[Tuple[float, float, float, float]] = None,
    schedule_cruise_mps: Optional[float] = None,
) -> Tuple[List[int], float]:
    """Return (best per-section speed-mode vector, best reward) via SA.

    reward_fn(arrival_deviation_sec, arrival_time) -> reward to MAXIMISE,
    e.g. the L3 terminal reward with headway already enforced by the sim.
    incident_window = (start, end, f_inc, sf_inc) of a speed-restriction incident;
    it is applied SECTION-RESOLVED (only sections whose traversal overlaps the
    window are slowed) — the same physics the worker and rule CRUISE see.
    schedule_cruise_mps (per-train schedule speed) is passed to the sim so the
    greedy baseline sees the SAME physics as the learned worker.
    """
    n = len(sections)
    rng = random.Random(seed)

    def sec_factors(k: int, t: float, mode: int) -> Tuple[float, float]:
        """Section-resolved incident factors for section k traversed from t."""
        if not incident_window:
            return 1.0, 1.0
        i_start, i_end, f_inc, sf_inc = incident_window
        run_est = sim.section_run_time(sections[k], SpeedMode(mode), priority,
                                       1.0, schedule_cruise_mps)
        return sim.incident_section_factors(i_start, i_end, f_inc, sf_inc,
                                            t, run_est)

    def outcome(modes: List[int]) -> float:
        """Arrival time (absolute) for a per-section speed profile."""
        t = departure
        for k, sec in enumerate(sections):
            sf_k, saf_k = sec_factors(k, t, modes[k])
            wait, run = sim.section_progress(
                corridor_id, t, bounds[k], sec, SpeedMode(modes[k]), priority,
                speed_factor=sf_k, safety_factor=saf_k,
                schedule_cruise_mps=schedule_cruise_mps)
            t += wait + run
        return t + sim.p.approach_sec + sim.p.throat_in_sec

    def score(modes: List[int]) -> float:
        arr = outcome(modes)
        # same per-section FAST energy cost as the learned worker — otherwise SA
        # would freely FAST and the comparison would be unfair.
        fast_cost = L3_FAST_PENALTY * sum(1 for m in modes if m == 1)
        return reward_fn(arr - planned_arrival, arr) - fast_cost

    best_modes = [0] * n  # all-CRUISE start
    best_score = score(best_modes)
    cur = list(best_modes)
    cur_score = best_score
    temp = t0
    for it in range(n_iter):
        i = rng.randrange(n)
        new_mode = rng.choice([m for m in (0, 1, 2) if m != cur[i]])
        cand = list(cur)
        cand[i] = new_mode
        s = score(cand)
        if s > cur_score or rng.random() < math.exp((s - cur_score) / temp):
            cur = cand
            cur_score = s
            if s > best_score:
                best_score = s
                best_modes = list(cand)
        temp = t0 * ((t1 / t0) ** ((it + 1) / n_iter))
    return best_modes, best_score


def sa_online_profile(
    sim: L3Simulator,
    corridor_id: str,
    departure: float,
    priority: int,
    sections: List[L3Section],
    bounds: List[float],
    planned_arrival: float,
    reward_fn: Callable[[float, float], float],
    n_iter: int = 300,      # ignored: online baseline has no joint annealing
    seed: int = 0,          # ignored: deterministic
    t0: float = 10.0,       # ignored
    t1: float = 0.1,        # ignored
    incident_window: Optional[Tuple[float, float, float, float]] = None,
    schedule_cruise_mps: Optional[float] = None,
) -> Tuple[List[int], float]:
    """Partial-observation baseline: per-section greedy recompute from the
    CURRENT state only.

    incident_window = (start, end, f_inc, sf_inc) of a speed-restriction incident.
    The CURRENT section applies the section-resolved factor (it sees reality);
    the CRUISE rollout for the remaining sections uses nominal (factor-free) run
    times — a partial-observation baseline cannot foresee a future restriction.

    Unlike :func:`sa_optimize_profile` (a full-information offline oracle that
    jointly anneals the whole profile and can "see" the front occupant's full
    trajectory through ``section_progress``), this baseline picks each section's
    speed mode from what is knowable at that moment: the headway wait NOW (the
    moving-block wait is mode-independent), the delay accrued SO FAR, and the
    corridor geometry ahead.  Remaining sections are rolled out at CRUISE with
    NO future headway wait — i.e. the same partial observability the learned L3
    worker has (10-dim per-section obs: headway gap, accumulated wait, delay,
    remaining distance).  No joint lookahead, no front-occupant profile.

    The comparison is therefore apples-to-apples: if the learned worker beats
    this baseline, the advantage is coordination/learning, not omniscience.
    """
    n = len(sections)
    # geometry run times — NOMINAL (factor-free): the online baseline cannot
    # foresee a future restriction, so the remaining-sections rollout is clear.
    cruise_rt = [sim.section_run_time(s, SpeedMode.CRUISE, priority, 1.0,
                                      schedule_cruise_mps) for s in sections]
    fast_rt = [sim.section_run_time(s, SpeedMode.FAST, priority, 1.0,
                                    schedule_cruise_mps) for s in sections]
    slow_rt = [sim.section_run_time(s, SpeedMode.SLOW, priority, 1.0,
                                    schedule_cruise_mps) for s in sections]
    # remaining-cruise suffix sums, so the rollout is O(1) per candidate
    suffix = [0.0] * (n + 1)
    for k in range(n - 1, -1, -1):
        suffix[k] = suffix[k + 1] + cruise_rt[k]
    tail = sim.p.approach_sec + sim.p.throat_in_sec

    def sec_factors(k: int, t: float) -> Tuple[float, float]:
        if not incident_window:
            return 1.0, 1.0
        i_start, i_end, f_inc, sf_inc = incident_window
        run_est = sim.section_run_time(sections[k], SpeedMode.CRUISE, priority,
                                       1.0, schedule_cruise_mps)
        return sim.incident_section_factors(i_start, i_end, f_inc, sf_inc,
                                            t, run_est)

    modes = [0] * n
    t = departure
    total_wait = 0.0
    total_run = 0.0
    for k, sec in enumerate(sections):
        sf_k, saf_k = sec_factors(k, t)
        # headway wait at this section is mode-independent — one probe suffices.
        wait, _ = sim.section_progress(
            corridor_id, t, bounds[k], sec, SpeedMode.CRUISE, priority,
            speed_factor=sf_k, safety_factor=saf_k,
            schedule_cruise_mps=schedule_cruise_mps)
        best_mode, best_score = 0, None
        for m in (0, 1, 2):
            rt_here = (fast_rt[k] if m == 1 else slow_rt[k] if m == 2 else cruise_rt[k])
            pred_arr = t + wait + rt_here + suffix[k + 1] + tail
            dev = pred_arr - planned_arrival
            score = reward_fn(dev, pred_arr) if reward_fn is not None else -abs(dev)
            if m == 1:
                score -= L3_FAST_PENALTY
            if best_score is None or score > best_score:
                best_score, best_mode = score, m
        modes[k] = best_mode
        wait_k, run_k = sim.section_progress(
            corridor_id, t, bounds[k], sec, SpeedMode(best_mode), priority,
            speed_factor=sf_k, safety_factor=saf_k,
            schedule_cruise_mps=schedule_cruise_mps)
        t += wait_k + run_k
        total_wait += wait_k
        total_run += run_k
    arr = t + tail
    final = reward_fn(arr - planned_arrival, arr) if reward_fn is not None else -1.0
    return modes, final
