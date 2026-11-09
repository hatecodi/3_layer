"""Tests for the corridor-closure disturbance (区间中断).

Run:
    python -m pytest tests/test_closure.py -v
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.control.action_mask import ENTRY_CHOICES, derive_schedule
from src.env.disturbance import inject_corridor_closure
from src.env.five_station_env import TRAVEL_TIME_TO_THROAT, FiveStationEnv
from src.env.observation_builder import get_obs_dim
from src.experiment.scenario_loader import load_scenario, route_stations


@pytest.fixture(scope="module")
def cfg():
    return load_scenario(scenario_name="five_station_15t")


def _events(cfg, cid):
    a, b = cid.split("_to_")
    out = []
    for tid, t in cfg.trains.items():
        route = route_stations(t, cfg.corridors)
        if a not in route or b not in route:
            continue
        pa = t.planned.get(a)
        if pa is None or pa.dep is None or pa.dep <= 0:
            continue
        bounds = t.bounds.get(a)
        dwell = bounds.dwell_min if bounds else 0.0
        out.append((float(pa.dep), int(t.priority), tid, float(dwell)))
    return sorted(out)


def test_closure_generator_targets_a_designed_meet(cfg):
    got = None
    for s in range(50):
        res = inject_corridor_closure(cfg, random.Random(s))
        if res:
            got = res
            break
    assert got, "no closure generated in 50 seeds"
    cid, (t0, t1) = next(iter(got.items()))
    assert 0.0 <= t0 < t1
    # There must be a lower-priority leader before t1 and a higher-priority
    # follower whose plan entry is after t1 (so only the leader is held).
    pairs = []
    evs = _events(cfg, cid)
    for d_l, p_l, tid_l, _ in evs:
        for d_f, p_f, tid_f, dwell_f in evs:
            if p_l > p_f and d_l < t1 and d_f - dwell_f > t1:
                pairs.append((tid_l, tid_f))
    assert pairs, f"closure {cid}=({t0},{t1}) does not hold a leader before a follower"


def test_closure_effective_bounds_hold_leader_not_follower(cfg, monkeypatch):
    monkeypatch.setattr(
        "src.env.five_station_env.inject_corridor_closure",
        lambda *a, **k: {"A_to_B": (0.0, 85.0)})
    env = FiveStationEnv(cfg=cfg, seed=0, enable_vt=False,
                         enable_initial_delay=False,
                         enable_corridor_incident=False,
                         enable_corridor_closure=True)
    env.reset()
    t_now = env._t_now
    # G5038 (P2) departs A at 77 -> held to t1=85.
    b = env._get_effective_bounds("G5038", "A", t_now)
    dwell = cfg.trains["G5038"].bounds["A"].dwell_min
    assert b.arr_min >= 85.0 - dwell - 1e-6
    times = derive_schedule(
        cfg.trains["G5038"], "A", ("A_p2", "plan", 0),
        t_now, TRAVEL_TIME_TO_THROAT, bounds_override=b)
    assert times[4] >= 85.0 - 1e-6  # departure (W_out)
    # G1616_s3 (P1) departs A at 95, after t1 -> unaffected.
    bf = env._get_effective_bounds("G1616_s3", "A", t_now)
    tf = derive_schedule(
        cfg.trains["G1616_s3"], "A", ("A_p2", "plan", 0),
        t_now, TRAVEL_TIME_TO_THROAT, bounds_override=bf)
    assert abs(tf[4] - 95.0) <= 1e-6


def test_closure_observation_features(cfg, monkeypatch):
    monkeypatch.setattr(
        "src.env.five_station_env.inject_corridor_closure",
        lambda *a, **k: {"A_to_B": (0.0, 85.0)})
    env = FiveStationEnv(cfg=cfg, seed=0, enable_vt=False,
                         enable_initial_delay=False,
                         enable_corridor_incident=False,
                         enable_corridor_closure=True)
    obs, info = env.reset()
    assert get_obs_dim(cfg) == 37
    assert all(len(v) == 37 for v in obs.values())
    # self_state = 12 base + 4 class one-hot + 3 follower + 3 closure
    closed_idx, remaining_idx, vs_plan_idx = 19, 20, 21
    o = obs["G5038"]
    assert o[closed_idx] == 1.0
    assert o[remaining_idx] == pytest.approx(85.0 / 300.0)
    assert o[vs_plan_idx] == pytest.approx((85.0 - 77.0) / 120.0)
    # A train whose next corridor (E_to_D, down) is not closed sees zeros.
    assert obs["G1347"][closed_idx] == 0.0


def test_closure_yield_is_policy_driven(cfg, monkeypatch):
    """The requested yield (not the resolver) must decide the departure order."""
    monkeypatch.setattr(
        "src.env.five_station_env.inject_corridor_closure",
        lambda *a, **k: {"A_to_B": (0.0, 43.0)})
    env = FiveStationEnv(cfg=cfg, seed=0, enable_vt=False,
                         enable_initial_delay=False,
                         enable_corridor_incident=False,
                         enable_corridor_closure=True)
    env.reset()
    env._corridor_closures = {"A_to_B": (0.0, 43.0)}
    plan = ENTRY_CHOICES.index("plan")
    plus20 = ENTRY_CHOICES.index("plan+20")
    # G230 (P2, dep 31) yields; G1380 (P1, dep 53) departs on plan.
    _, _, _, _, info = env.step(
        {"G230": (0, plus20), "G1380": (0, plan)}, lambda_team=0.3)
    events = [e for e in info["overtake_events"]
              if e["is_overtake"] and e["closure_related"]]
    assert events, "forced yield produced no closure overtake"
    e = events[0]
    assert e["overtaken"] == "G230" and e["overtaker"] == "G1380"
    assert e["order_policy_driven"] is True
    assert e["req_dep_overtaken"] > e["req_dep_overtaker"]
    assert e["dep_overtaken"] > e["dep_overtaker"]
    # Closure stratification fields on the realized transit.
    g230 = [ev for ev in info["l3_transit_events"] if ev["train_id"] == "G230"]
    assert g230 and g230[0]["closure_affected"] is True
    assert g230[0]["closure_affected_plan"] is True
    assert g230[0]["closure_hold_sec"] > 0.0
    assert g230[0]["affected"] is True


def test_relevant_plus_is_a_superset(cfg):
    base = FiveStationEnv(cfg=cfg, seed=960011, enable_vt=False,
                          enable_initial_delay=False,
                          enable_corridor_incident=False,
                          enable_corridor_closure=True,
                          control_coverage="base")
    plus = FiveStationEnv(cfg=cfg, seed=960011, enable_vt=False,
                          enable_initial_delay=False,
                          enable_corridor_incident=False,
                          enable_corridor_closure=True,
                          control_coverage="plus")
    _, ib = base.reset()
    _, ip = plus.reset()
    rb = set(ib["control_relevant_ids"])
    rp = set(ip["control_relevant_ids"])
    assert rb <= rp
    assert len(rp) > len(rb), "plus coverage should expose more decisions"
    assert len(rp) >= 0.5 * len(ip["active_train_ids"])


def test_closure_episode_is_safe(cfg, monkeypatch):
    monkeypatch.setattr(
        "src.env.five_station_env.inject_corridor_closure",
        lambda *a, **k: {"A_to_B": (0.0, 85.0)})
    env = FiveStationEnv(cfg=cfg, seed=5, enable_vt=False,
                         enable_initial_delay=False,
                         enable_corridor_incident=False,
                         enable_corridor_closure=True)
    obs, info = env.reset()
    held_seen = 0
    for _ in range(400):
        _, _, term, trunc, info = env.step({}, lambda_team=0.3)
        held_seen += len(info.get("closure_held", {}))
        if term or trunc:
            break
    assert info["episode_status"] == "success"
    assert info["mask_stats"].get("postcheck_reject", 0) == 0
    assert info["hard_guard_violations"] == 0
    assert held_seen > 0
