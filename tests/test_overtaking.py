"""Tests for per-station pass-through, yield ordering, and externality pricing.

Run:
    python -m pytest tests/test_overtaking.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.control.action_mask import ENTRY_CHOICES, derive_schedule, heuristic_action
from src.control.l3_sim import L3Simulator, calibrate_corridor_sections, load_l3_physics
from src.control.resource_manager import GrantResult
from src.env.five_station_env import FiveStationEnv
from src.env.observation_builder import get_obs_dim
from src.env.reward import (
    EXTERN_ALPHA,
    L2_PRIORITY_WEIGHT,
    YIELD_TAIL_COEF,
    compute_layered_rewards,
)
from src.experiment.run_train_hierarchical import policy_control_prob
from src.experiment.scenario_loader import is_station_pass, load_scenario


@pytest.fixture(scope="module")
def cfg():
    return load_scenario(scenario_name="five_station_15t")


def test_per_station_pass_detection(cfg):
    assert cfg.trains["G230"].pass_stations == {"C"}
    assert cfg.trains["G238"].pass_stations == {"B", "C"}
    assert cfg.trains["G1487"].pass_stations == {"B"}
    # off-route placeholders must not become pass stations
    assert cfg.trains["G1185"].pass_stations == set()
    assert is_station_pass(cfg.trains["G238"], "B")
    assert not is_station_pass(cfg.trains["G238"], "A")


def test_platform_roles(cfg):
    assert cfg.trains["G238"].candidate_platforms["B"] == ["B_p1"]
    assert "B_p1" not in cfg.trains["G1380_s1"].candidate_platforms["B"]
    assert set(cfg.trains["G1380_s1"].candidate_platforms["D"]) == {
        "D_p1", "D_p2"}
    for train in cfg.trains.values():
        for st, plats in train.candidate_platforms.items():
            if st in cfg.stations:
                assert plats, f"{train.id} has no candidate at {st}"


def test_pass_schedule_at_station(cfg):
    train = cfg.trains["G238"]
    times = derive_schedule(
        train, "B", ("B_p1", "plan", 0), t_now=100.0,
        travel_time_to_throat=0.0)
    _, _, t_plat_start, t_plat_end, _, _ = times
    assert t_plat_end - t_plat_start == pytest.approx(5.0)

    stop = cfg.trains["G1380_s1"]
    sb = stop.bounds["B"]
    stimes = derive_schedule(
        stop, "B", ("B_p2", "plan", 0), t_now=100.0,
        travel_time_to_throat=0.0)
    assert stimes[3] - stimes[2] >= sb.dwell_min - 1e-6


def _heuristic_action_dict(env):
    ad = {}
    for tid in env._pending_trains(lookahead=env.batch_window_sec):
        train = env.cfg.trains[tid]
        st = env._journeys[tid].current_station
        plats = train.candidate_platforms.get(st, [])
        if not plats:
            continue
        try:
            b = env._get_effective_bounds(tid, st, env._t_now)
            plat, entry, _ = heuristic_action(
                train, st, env._t_now, 0.0, env.cfg, bounds_override=b)
            if plat in plats:
                ad[tid] = (plats.index(plat), ENTRY_CHOICES.index(entry))
        except ValueError:
            continue
    return ad


def test_forced_yield_reorders_departure(cfg):
    """G5038 (P2) delays at A; a P1 train departs first -> yield + overtake.

    This is the empirically useful origin case (see coding_log 2026-11-06):
    plan+20 improves system return, plan+50 over-delays.
    """
    env = FiveStationEnv(cfg=cfg, seed=0, enable_vt=False,
                         enable_initial_delay=False,
                         enable_corridor_incident=False)
    env.reset()
    events = []
    for _ in range(6):
        ad = _heuristic_action_dict(env)
        plats = cfg.trains["G5038"].candidate_platforms["A"]
        ad["G5038"] = (plats.index("A_p2"),
                       ENTRY_CHOICES.index("plan+20"))
        _, _, term, trunc, info = env.step(ad, lambda_team=0.3)
        events.extend(info["overtake_events"])
        if term or trunc:
            break
    a_events = [e for e in events if e["station"] == "A"]
    assert a_events, "expected a departure re-sequencing event at A"
    assert any(e["overtaken"] == "G5038" and e["is_yield"]
               and e["prio_overtaker"] < e["prio_overtaken"]
               for e in a_events)


def test_externality_charge_math(cfg):
    # P2 front: charge = alpha * ledger / w = 0.05 * 16 / 1.0 = 0.8
    grant = GrantResult(train_id="G230", platform="A_p2",
                        W_in=(26.0, 26.0), W_plat=(26.0, 31.0),
                        W_out=(31.0, 31.0), winner=True)
    out = compute_layered_rewards(
        cfg, [grant], [], active_train_ids=["G230"],
        station_map={"G230": "A"}, externality_charges={"G230": 16.0})
    assert out["G230"].externality == pytest.approx(
        EXTERN_ALPHA * 16.0 / L2_PRIORITY_WEIGHT[2])

    # P1 front with the same ledger pays half.
    grant_p1 = GrantResult(train_id="G1380", platform="A_p2",
                           W_in=(48.0, 48.0), W_plat=(48.0, 53.0),
                           W_out=(53.0, 53.0), winner=True)
    out_p1 = compute_layered_rewards(
        cfg, [grant_p1], [], active_train_ids=["G1380"],
        station_map={"G1380": "A"}, externality_charges={"G1380": 16.0})
    assert out_p1["G1380"].externality == pytest.approx(
        EXTERN_ALPHA * 16.0 / L2_PRIORITY_WEIGHT[1])


def test_yield_tail_charge(cfg):
    grant = GrantResult(train_id="G230", platform="A_p2",
                        W_in=(26.0, 26.0), W_plat=(26.0, 31.0),
                        W_out=(31.0, 31.0), winner=True)
    out = compute_layered_rewards(
        cfg, [grant], [], active_train_ids=["G230"],
        station_map={"G230": "A"}, yield_tail_charges={"G230": 30.0})
    assert out["G230"].yield_tail == pytest.approx(YIELD_TAIL_COEF * 30.0)


def test_front_train_id():
    cfg = load_scenario(scenario_name="five_station")
    physics = load_l3_physics(cfg.project_root / "configs" / "five_station")
    sections = calibrate_corridor_sections(cfg.trains, physics)
    sim = L3Simulator(physics, {}, {}, sections=sections)
    assert sim.front_train_id("A_to_B") is None
    n = len(sections["A_to_B"])
    sim._corridors.record_occupancy_abs(
        "A_to_B", "X", [0.0] * n, [10.0] * n)
    assert sim.front_train_id("A_to_B") == "X"


def test_policy_action_set_covers_batch_window(cfg):
    """The policy must act on every batch-window train, not only arrived ones.

    Regression guard for the 2026-11-06 finding that only ~14% of grants were
    policy-controlled (the rest were heuristic-scheduled 120 s ahead).
    """
    env = FiveStationEnv(cfg=cfg, seed=0, enable_vt=False,
                         enable_initial_delay=False,
                         enable_corridor_incident=False)
    _, info = env.reset()
    expect = set(env._pending_trains(lookahead=env.batch_window_sec))
    assert set(info["action_masks"]) == expect
    assert info["active_train_ids"] == info["policy_train_ids"]
    # Origin trains within the lookahead must be policy-controlled.
    assert "G5038" in expect


def test_policy_control_ramp():
    assert policy_control_prob(0, 2000) == pytest.approx(0.1)
    assert policy_control_prob(600, 2000) == pytest.approx(0.55)
    assert policy_control_prob(1200, 2000) == pytest.approx(1.0)
    assert policy_control_prob(1999, 2000) == pytest.approx(1.0)
    assert policy_control_prob(10, 0) == pytest.approx(1.0)


def test_control_relevant_ids(cfg):
    env = FiveStationEnv(cfg=cfg, seed=0, enable_vt=False,
                         enable_initial_delay=False,
                         enable_corridor_incident=False)
    _, info = env.reset()
    rel = set(info["control_relevant_ids"])
    assert rel <= set(info["action_masks"])
    # P2 trains ahead of a P1 follower at A are yield-relevant.
    assert "G230" in rel and "G238" in rel and "G5038" in rel
    # A P1 train has no higher-priority follower -> heuristic handles it.
    assert "G1380" not in rel


def test_yield_entry_cap(cfg):
    env = FiveStationEnv(cfg=cfg, seed=0, enable_vt=False,
                         enable_initial_delay=False,
                         enable_corridor_incident=False,
                         yield_entry_cap=True)
    _, info = env.reset()
    em = info["action_masks"]["G230"][1]
    assert em[4] is False and em[5] is False  # plan+30 / plan+50 capped
    assert any(em[:4])

    env2 = FiveStationEnv(cfg=cfg, seed=0, enable_vt=False,
                          enable_initial_delay=False,
                          enable_corridor_incident=False,
                          yield_entry_cap=False)
    _, info2 = env2.reset()
    em2 = info2["action_masks"]["G230"][1]
    assert any(em2[4:])


def test_obs_dim_and_clean_episode(cfg):
    assert get_obs_dim(cfg) == 37
    env = FiveStationEnv(cfg=cfg, seed=3, enable_vt=False,
                         enable_initial_delay=False,
                         enable_corridor_incident=False)
    obs, info = env.reset()
    assert all(len(v) == 37 for v in obs.values())
    granted = []
    for _ in range(400):
        _, _, term, trunc, info = env.step({}, lambda_team=0.3)
        granted.extend(info.get("granted_windows", []))
        if term or trunc:
            break
    assert info["episode_status"] == "success"
    assert info["mask_stats"].get("postcheck_reject", 0) == 0
    assert info["hard_guard_violations"] == 0
    # Pass trains only use through-allowed platforms.
    for g in granted:
        train = cfg.trains[g.train_id]
        for st, plats in train.candidate_platforms.items():
            if g.platform in plats and is_station_pass(train, st):
                station = cfg.stations[st]
                allowed = {p.id for p in station.platforms if p.through_allowed}
                assert g.platform in allowed
