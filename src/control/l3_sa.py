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
from typing import Callable, List, Tuple

from .l3_sim import L3Section, L3Simulator, SpeedMode


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
) -> Tuple[List[int], float]:
    """Return (best per-section speed-mode vector, best reward) via SA.

    reward_fn(arrival_deviation_sec, arrival_time) -> reward to MAXIMISE,
    e.g. the L3 terminal reward with headway already enforced by the sim.
    """
    n = len(sections)
    rng = random.Random(seed)

    def outcome(modes: List[int]) -> float:
        """Arrival time (absolute) for a per-section speed profile."""
        t = departure
        for k, sec in enumerate(sections):
            wait, run = sim.section_progress(
                corridor_id, t, bounds[k], sec, SpeedMode(modes[k]), priority)
            t += wait + run
        return t + sim.p.approach_sec + sim.p.throat_in_sec

    def score(modes: List[int]) -> float:
        arr = outcome(modes)
        return reward_fn(arr - planned_arrival, arr)

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
