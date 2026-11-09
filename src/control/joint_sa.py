"""Fleet-level JOINT simulated annealing over per-section speed modes.

Difference from :mod:`src.control.l3_sa` (read this before citing either):

  * ``l3_sa.sa_optimize_profile``  — anneals ONE corridor transit.  Decision
    vector length = n_sections (4 on this network), so its search space is a
    FIXED ``3**4 = 81`` regardless of fleet size.  Its solve time therefore does
    NOT grow with the number of trains.
  * ``joint_sa_optimize`` (here)   — anneals ALL trains' per-section modes in a
    SINGLE decision vector of length ``n_sections * n_trains``, so the search
    space is ``3**(n_sections * n_trains)`` — exponential in fleet size.  This is
    the centralized, full-information oracle whose cost explodes with scale.

Both optimise the SAME objective (terminal punctuality minus the per-FAST energy
penalty) so they are comparable with the learned L3 worker.

Coupling: every candidate is evaluated by replaying all transits, in departure
order, through ONE shared :class:`L3Simulator` corridor state.  Trains sharing a
corridor therefore interact through the moving-block headway; trains on disjoint
corridors do not.  The strength of that coupling is what decides whether the
time-to-target curve actually grows — if it does not, report it as it is.

Gate 1 of the local search perturbs a single section mode of a single train and
re-evaluates, so the per-iteration cost is O(total sections).  With a FIXED
iteration budget the wall-clock is therefore O(n_trains), NOT exponential; an
exponential *time* curve can only come from measuring time-to-target, because
the number of iterations needed grows with the space.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Dict, List, Optional, Tuple

from .l3_sim import L3CorridorState, L3Section, L3Simulator, SpeedMode
from ..env.reward import L3_FAST_PENALTY


@dataclass
class JointTransit:
    """One train's corridor traversal inside a joint instance."""
    train_id: str
    corridor_id: str
    departure: float
    priority: int
    sections: List[L3Section]
    bounds: List[float]
    planned_arrival: float
    schedule_cruise_mps: Optional[float] = None
    incident_window: Optional[Tuple[float, float, float, float]] = None
    # filled by the builder: index of this transit's first variable
    offset: int = 0

    @property
    def n_sections(self) -> int:
        return len(self.sections)


def build_joint_vector(transits: List[JointTransit]) -> None:
    """Assign each transit its slice offset in the concatenated decision vector."""
    off = 0
    for tr in transits:
        tr.offset = off
        off += tr.n_sections


def make_evaluator(sim: L3Simulator, transits: List[JointTransit],
                   reward_fn: Callable[[float, float], float]) -> Callable[[List[int]], float]:
    """Return objective(modes) for the concatenated joint vector.

    Pure function of *modes*: the shared corridor state is rebuilt from scratch
    on every call, so the objective carries no hidden state between candidates.
    """
    order = sorted(range(len(transits)), key=lambda i: transits[i].departure)
    p = sim.p

    def objective(modes: List[int]) -> float:
        sim._corridors = L3CorridorState()
        total = 0.0
        for i in order:
            tr = transits[i]
            seg = modes[tr.offset: tr.offset + tr.n_sections]
            t = tr.departure
            entries: List[float] = []
            exits: List[float] = []
            for k, sec in enumerate(tr.sections):
                sf_k, saf_k = 1.0, 1.0
                if tr.incident_window:
                    i_s, i_e, f_inc, sf_inc = tr.incident_window
                    run_est = sim.section_run_time(
                        sec, SpeedMode(seg[k]), tr.priority, 1.0,
                        tr.schedule_cruise_mps)
                    sf_k, saf_k = sim.incident_section_factors(
                        i_s, i_e, f_inc, sf_inc, t, run_est)
                wait, run = sim.section_progress(
                    tr.corridor_id, t, tr.bounds[k], sec, SpeedMode(seg[k]),
                    tr.priority, speed_factor=sf_k, safety_factor=saf_k,
                    schedule_cruise_mps=tr.schedule_cruise_mps)
                entries.append(t)
                t += wait + run
                exits.append(t)
            sim._corridors.record_occupancy_abs(
                tr.corridor_id, tr.train_id, entries, exits)
            arr = t + p.approach_sec + p.throat_in_sec
            total += reward_fn(arr - tr.planned_arrival, arr)
            total -= L3_FAST_PENALTY * sum(1 for m in seg if m == 1)
        return total

    return objective


@dataclass
class JointSAResult:
    best_modes: List[int]
    best_value: float
    target: float
    reached: bool
    iters_to_target: Optional[int]
    wall_s_to_target: Optional[float]
    total_wall_s: float
    evals: int
    trace: List[Tuple[int, float]] = field(default_factory=list)


def joint_sa_optimize(
    sim: L3Simulator,
    transits: List[JointTransit],
    reward_fn: Callable[[float, float], float],
    target: float,
    n_iter: int = 2000,
    seed: int = 0,
    t0: float = 10.0,
    t1: float = 0.1,
    trace_every: int = 0,
) -> JointSAResult:
    """Anneal the concatenated speed-mode vector of ALL transits jointly.

    Stops early once ``target`` is reached (that is the measurement); returns how
    many iterations and how much wall-clock it took.  ``reached=False`` means the
    budget was exhausted first (censored sample — do not drop it silently).
    """
    build_joint_vector(transits)
    objective = make_evaluator(sim, transits, reward_fn)

    n_vars = sum(tr.n_sections for tr in transits)
    rng = random.Random(seed)

    t_start = perf_counter()
    evals = 0
    best = [0] * n_vars                      # all-CRUISE start
    best_val = objective(best); evals += 1
    cur = list(best); cur_val = best_val
    trace: List[Tuple[int, float]] = []

    iters_to_target: Optional[int] = None
    wall_to_target: Optional[float] = None
    if best_val >= target:
        iters_to_target, wall_to_target = 0, 0.0

    temp = t0
    for it in range(n_iter):
        j = rng.randrange(n_vars)
        new_mode = rng.choice([m for m in (0, 1, 2) if m != cur[j]])
        cand = list(cur)
        cand[j] = new_mode
        val = objective(cand); evals += 1
        if val > cur_val or rng.random() < math.exp((val - cur_val) / max(temp, 1e-9)):
            cur, cur_val = cand, val
            if val > best_val:
                best_val, best = val, list(cand)
                if iters_to_target is None and best_val >= target:
                    iters_to_target = it
                    wall_to_target = perf_counter() - t_start
        temp = t0 * ((t1 / t0) ** ((it + 1) / n_iter))
        if trace_every and (it + 1) % trace_every == 0:
            trace.append((it + 1, best_val))
        if iters_to_target is not None and iters_to_target > 0 and it > iters_to_target:
            break

    return JointSAResult(
        best_modes=best, best_value=best_val, target=target,
        reached=iters_to_target is not None,
        iters_to_target=iters_to_target, wall_s_to_target=wall_to_target,
        total_wall_s=perf_counter() - t_start, evals=evals, trace=trace)


def independent_baseline(sim: L3Simulator, transits: List[JointTransit],
                         reward_fn: Callable[[float, float], float],
                         n_iter: int = 300, seed: int = 0) -> List[int]:
    """Per-transit independent optimum (the DECOMPOSED solution).

    Each transit is annealed alone with an empty corridor, exactly as
    ``l3_sa.sa_optimize_profile`` does — i.e. the "solve each leg on its own"
    strategy.  Used to define the joint target: how long does the joint search
    need merely to MATCH the decomposed answer when both are scored jointly?
    """
    modes: List[int] = []
    p = sim.p
    for idx, tr in enumerate(transits):
        sim._corridors = L3CorridorState()
        best = [0] * tr.n_sections
        best_val = -math.inf
        # index-based (NOT hash(train_id) — str hashing is salted per process,
        # which would make the baseline irreproducible across runs)
        rng = random.Random(seed * 100003 + idx)

        def score(seg: List[int]) -> float:
            sim._corridors = L3CorridorState()
            t = tr.departure
            for k, sec in enumerate(tr.sections):
                wait, run = sim.section_progress(
                    tr.corridor_id, t, tr.bounds[k], sec, SpeedMode(seg[k]),
                    tr.priority, schedule_cruise_mps=tr.schedule_cruise_mps)
                t += wait + run
            arr = t + p.approach_sec + p.throat_in_sec
            return (reward_fn(arr - tr.planned_arrival, arr)
                    - L3_FAST_PENALTY * sum(1 for m in seg if m == 1))

        cur = [0] * tr.n_sections
        cur_val = score(cur)
        best_val = cur_val
        temp = 10.0
        for it in range(n_iter):
            k = rng.randrange(tr.n_sections)
            nm = rng.choice([m for m in (0, 1, 2) if m != cur[k]])
            cand = list(cur)
            cand[k] = nm
            v = score(cand)
            if v > cur_val or rng.random() < math.exp((v - cur_val) / max(temp, 1e-9)):
                cur, cur_val = cand, v
                if v > best_val:
                    best_val, best = v, list(cand)
            temp = 10.0 * ((0.1 / 10.0) ** ((it + 1) / n_iter))
        modes.extend(best)
    return modes
