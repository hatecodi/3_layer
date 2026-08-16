"""L3 v2 physics tests: calibration + moving-block headway (sections as speed zones).

Run:
    python -m pytest tests/test_l3_physics.py -v
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.control.l3_sim import (
    L3Simulator,
    L3Section,
    SpeedMode,
    calibrate_corridor_sections,
    compute_corridor_headway,
    load_l3_physics,
    _reference_transit,
)
from src.env.disturbance import _corridor_transit_windows, inject_corridor_incident
from src.experiment.scenario_loader import load_scenario


def _physics_and_sections():
    cfg = load_scenario(scenario_name="five_station")
    physics = load_l3_physics(cfg.project_root / "configs" / "five_station")
    sections = calibrate_corridor_sections(cfg.trains, physics)
    return cfg, physics, sections


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_sections_loaded_from_config(self):
        cfg, physics, sections = _physics_and_sections()
        assert physics.corridor_sections, "corridor_sections must load from yaml"
        assert "A_to_B" in sections
        # all configured corridors calibrated (4 sections each)
        for cid, secs in sections.items():
            assert len(secs) == 4, f"{cid}: expected 4 sections"

    def test_cruise_reproduces_reference_transit(self):
        """On-time departure + CRUISE -> arrival exactly at the timetable
        reference transit (fixes the systematic early-arrival / S2)."""
        cfg, physics, sections = _physics_and_sections()
        hw = compute_corridor_headway(cfg.trains, physics)
        overhead = physics.corridor_overhead_sec
        speeds = physics.get_speeds(2)
        for cid, secs in sections.items():
            ref = _reference_transit(cid, cfg.trains, 2)
            assert ref is not None, f"no reference transit for {cid}"
            # CRUISE run must respect per-section speed limits (approach zone) —
            # the calibration compensates by lengthening the free sections.
            cruise_run = sum(
                s.length_m / min(speeds.cruise_mps, s.limit_mps)
                if s.limit_mps else s.length_m / speeds.cruise_mps
                for s in secs)
            assert abs((cruise_run + overhead) - ref) < 1e-3, (
                f"{cid}: CRUISE transit {cruise_run + overhead:.2f} != ref {ref:.2f}")

    def test_reference_transit_not_polluted_by_placeholder_legs(self):
        """Trains with placeholder planned stations outside their route must
        not contribute fake legs (G1487 used to give D_to_C a 29s transit)."""
        cfg, _, _ = _physics_and_sections()
        ref = _reference_transit("D_to_C", cfg.trains, 2)
        assert ref is not None and ref > 100.0, f"D_to_C ref suspiciously small: {ref}"


# ---------------------------------------------------------------------------
# moving-block headway
# ---------------------------------------------------------------------------

class TestMovingBlockHeadway:
    def _sim(self):
        cfg, physics, sections = _physics_and_sections()
        hw = compute_corridor_headway(cfg.trains, physics)
        return L3Simulator(physics, hw, {}, sections=sections), sections

    def test_same_speed_following_respects_safety_distance(self):
        """Moving-block: a same-speed follower may enter once the front train is
        a safety distance ahead (NOT after clearing a whole section).  The entry
        gap is much tighter than fixed-block, even with the section approach
        limit (which can make the corridor-exit constraint bind and delay the
        entry a little beyond the pure safety-distance bound)."""
        sim, _ = self._sim()
        speeds = sim.p.get_speeds(2)
        safety_time = sim.p.moving_block_safety_m / speeds.cruise_mps
        # T1 enters corridor at 100+3=103; it is a safety distance ahead at
        # 103 + safety_time; T2 may enter no earlier than then + margin.
        sim.transit("T1", "A", "B", "A_to_B", 76, 100.0, SpeedMode.CRUISE, None, 2)
        earliest_safety = 103.0 + safety_time + sim.p.moving_block_margin_sec
        r2 = sim.transit("T2", "A", "B", "A_to_B", 76, 100.0, SpeedMode.CRUISE, None, 2)
        entry2 = 100.0 + 3.0 + r2.headway_delay_sec
        # cannot enter before the front is safety ahead of the entry point
        assert entry2 >= earliest_safety - 1e-6, (
            f"entry2 {entry2:.3f} < earliest_safety {earliest_safety:.3f}")
        # much tighter than fixed-block's whole-section0 wait (~16.75s + margin)
        assert r2.headway_delay_sec < 14.0

    def test_fast_follower_held_mid_section(self):
        """P2 regression: section_progress enforces the moving-block
        CONTINUOUSLY within a section — a FAST follower is held until the front
        clears safety beyond the section END, not just the section start."""
        from src.control.l3_sim import L3Physics, TrainClassSpeed
        physics = L3Physics()
        physics.moving_block_safety_m = 500.0
        physics.moving_block_margin_sec = 3.0
        physics.train_speeds = {"2": TrainClassSpeed(83.33, 100.0)}
        sections = {"T": [L3Section(1000.0) for _ in range(4)]}
        sim = L3Simulator(physics, {"T": 15.0}, {}, sections=sections)
        bounds = sim._sec_bounds["T"]
        # slow front: 5 m/s (200 s per 1000 m section)
        sim._corridors.record_occupancy_abs("T", "front",
                                            [0.0, 200.0, 400.0, 600.0],
                                            [200.0, 400.0, 600.0, 800.0])
        sec0 = sections["T"][0]
        wait, run = sim.section_progress("T", 0.0, bounds[0], sec0,
                                         SpeedMode.FAST, 2)
        # continuous: the follower may not exit section 0 before the front is
        # safety ahead of the section END (1000 + 500 = 1500 m) -> front_at 300
        # + margin 3 = 303; exit = wait + run must be >= 303.
        front_at_end = sim._corridors.front_time_at("T", 1500.0, bounds)
        min_exit = front_at_end + physics.moving_block_margin_sec
        assert 0.0 + wait + run >= min_exit - 1e-6, (
            f"follower exits {wait+run:.1f}s before front clears section end "
            f"{min_exit:.1f}s")
        # a start-only check (the old bug) would give only ~103s (front at 500m
        # = 100s + margin); the continuous constraint holds it to ~293s.
        assert wait > 200.0, f"wait {wait:.1f} not enforcing continuous constraint"

    def test_fast_behind_cruise_held_more(self):
        """A FAST follower behind a CRUISE train is held more than a same-speed
        follower (it would catch up mid-corridor)."""
        # both followers face the SAME single front train; only mode differs
        sim_fast, _ = self._sim()
        sim_fast.transit("T1", "A", "B", "A_to_B", 76, 100.0, SpeedMode.CRUISE, None, 2)
        r_fast = sim_fast.transit("T3", "A", "B", "A_to_B", 76, 110.0, SpeedMode.FAST, None, 2)
        sim_cruise, _ = self._sim()
        sim_cruise.transit("T1", "A", "B", "A_to_B", 76, 100.0, SpeedMode.CRUISE, None, 2)
        r_cruise = sim_cruise.transit("T4", "A", "B", "A_to_B", 76, 110.0, SpeedMode.CRUISE, None, 2)
        assert r_fast.headway_delay_sec > r_cruise.headway_delay_sec, (
            f"FAST delay {r_fast.headway_delay_sec:.2f} should exceed "
            f"CRUISE delay {r_cruise.headway_delay_sec:.2f}")

    def test_backward_compat_no_sections(self):
        """Without sections, transit falls back to the distance-based path."""
        cfg, physics, _ = _physics_and_sections()
        sim = L3Simulator(physics, {}, dict(physics.corridor_distances_m))
        r = sim.transit("T1", "A", "B", "A_to_B", 76, 100.0, SpeedMode.CRUISE, None, 2)
        # distance path: 5600m / 83.33 + 9s overhead
        assert abs(r.actual_transit_sec - (5600 / 83.33 + 9.0)) < 0.5
        # no occupancy recorded without sections
        assert sim._corridors._occupants.get("A_to_B", []) == []


# ---------------------------------------------------------------------------
# corridor speed-restriction incident
# ---------------------------------------------------------------------------

class TestCorridorIncident:
    def _sim(self):
        cfg, physics, sections = _physics_and_sections()
        hw = compute_corridor_headway(cfg.trains, physics)
        return L3Simulator(physics, hw, {}, sections=sections), sections

    def test_speed_factor_scales_run_time(self):
        """A restricted corridor is slower; factor 1.0 is a no-op."""
        sim, sections = self._sim()
        sec = sections["A_to_B"][0]
        base = sim.section_run_time(sec, SpeedMode.CRUISE, 2)
        assert sim.section_run_time(sec, SpeedMode.CRUISE, 2, speed_factor=1.0) == base
        half = sim.section_run_time(sec, SpeedMode.CRUISE, 2, speed_factor=0.5)
        assert abs(half - 2.0 * base) < 1e-6
        # min with section limit still applies
        assert half > base

    def test_safety_factor_grows_headway(self):
        """Wet-rail braking (safety_factor > 1) forces a LARGER following gap."""
        sim, sections = self._sim()
        sim.transit("T1", "A", "B", "A_to_B", 76, 100.0, SpeedMode.CRUISE, None, 2)
        sec = sections["A_to_B"][0]
        t = 103.0  # follower earliest corridor entry (throat-out + headway)
        w1, _ = sim.section_progress("A_to_B", t, 0.0, sec, SpeedMode.CRUISE, 2,
                                     safety_factor=1.0)
        w2, _ = sim.section_progress("A_to_B", t, 0.0, sec, SpeedMode.CRUISE, 2,
                                     safety_factor=1.5)
        assert w2 > w1, f"safety_factor should widen gap: {w1:.2f} -> {w2:.2f}"

    def test_transit_honors_speed_factor(self):
        """B1: the rule-based CRUISE baseline faces the same restriction —
        transit() with speed_factor < 1 is slower; factor 1.0 is unchanged."""
        sim, _ = self._sim()
        r1 = sim.transit("T1", "A", "B", "A_to_B", 76, 100.0, SpeedMode.CRUISE, None, 2)
        r2 = sim.transit("T2", "A", "B", "A_to_B", 76, 200.0, SpeedMode.CRUISE, None, 2,
                         speed_factor=0.5)
        r3 = sim.transit("T3", "A", "B", "A_to_B", 76, 300.0, SpeedMode.CRUISE, None, 2,
                         speed_factor=1.0)
        assert r2.actual_transit_sec > r1.actual_transit_sec
        assert abs(r3.actual_transit_sec - r1.actual_transit_sec) < 1e-6

    def test_transit_safety_factor_widens_entry_gap(self):
        """B1: wet-rail safety_factor also widens the CRUISE baseline's entry gap."""
        sim_a, _ = self._sim()
        sim_a.transit("T1", "A", "B", "A_to_B", 76, 100.0, SpeedMode.CRUISE, None, 2)
        r_a = sim_a.transit("T2", "A", "B", "A_to_B", 76, 100.0, SpeedMode.CRUISE, None, 2,
                            safety_factor=1.0)
        sim_b, _ = self._sim()
        sim_b.transit("T1", "A", "B", "A_to_B", 76, 100.0, SpeedMode.CRUISE, None, 2)
        r_b = sim_b.transit("T3", "A", "B", "A_to_B", 76, 100.0, SpeedMode.CRUISE, None, 2,
                            safety_factor=1.5)
        assert r_b.headway_delay_sec > r_a.headway_delay_sec

    def test_sa_execution_honors_speed_factor(self):
        """Regression: the env's SA EXECUTION loop must apply speed_factor.  It
        used to plan WITH the restriction (sa_optimize_profile) but execute at
        full speed -> SA arrived artificially EARLY on affected transits."""
        from src.env.five_station_env import FiveStationEnv
        cfg = load_scenario(scenario_name="five_station_15t")
        env = FiveStationEnv(cfg=cfg, seed=7, enable_vt=False,
                             enable_initial_delay=False,
                             enable_corridor_incident=False,
                             l3_worker=None, l3_sa=True)
        env.reset()
        # force an incident over the whole episode on the long C_D corridor
        env._corridor_incidents = {"C_to_D": (0.0, 10000.0, 0.4, 1.0)}
        affected = []
        for _ in range(20):
            obs, _, term, trunc, info = env.step({}, lambda_team=0.3)
            affected.extend(ev["arr_delay"] for ev in info.get("l3_transit_events", [])
                            if ev["affected"] and ev["corridor"] == "C_to_D")
            if term or trunc:
                break
        assert affected, "no affected C_to_D transit observed"
        # factor 0.4 over 13640m => even FAST adds >100s; a cheating SA (full
        # speed) would be ~0 or negative here
        assert min(affected) > 50, f"SA not slowed by incident: {affected[:5]}"

    def test_sa_online_honors_speed_factor(self):
        """Partial-observation SA baseline (per-section greedy, current-state
        only) must also honor speed_factor in execution — a cheating baseline
        that ignored the restriction would arrive artificially early."""
        from src.env.five_station_env import FiveStationEnv
        cfg = load_scenario(scenario_name="five_station_15t")
        env = FiveStationEnv(cfg=cfg, seed=7, enable_vt=False,
                             enable_initial_delay=False,
                             enable_corridor_incident=False,
                             l3_worker=None, l3_sa_online=True)
        env.reset()
        env._corridor_incidents = {"C_to_D": (0.0, 10000.0, 0.4, 1.0)}
        affected = []
        for _ in range(20):
            obs, _, term, trunc, info = env.step({}, lambda_team=0.3)
            affected.extend(ev["arr_delay"] for ev in info.get("l3_transit_events", [])
                            if ev["affected"] and ev["corridor"] == "C_to_D")
            if term or trunc:
                break
        assert affected, "no affected C_to_D transit observed"
        # same physics as the offline SA: factor 0.4 over 13640m means FAST still
        # adds >100s, so a baseline ignoring the restriction would be ~0 here.
        assert min(affected) > 50, f"sa_online not slowed by incident: {affected[:5]}"

    def test_incident_anchored_to_traffic(self):
        """Every generated incident overlaps >= 1 real transit (fixes the old
        50.8%-zero-train problem) and the factor stays in [lo, 1.0]."""
        cfg = load_scenario(scenario_name="five_station_15t")
        block = cfg.time_generation["corridor_incident"]
        f_lo, f_hi = block["factor_range"]
        rng = random.Random(11)
        hits = {0: 0, 1: 0, 2: 0}
        for _ in range(500):
            inc = inject_corridor_incident(cfg, rng)
            if not inc:
                continue
            for cid, (s, e, f, sf) in inc.items():
                n = sum(1 for (a, b) in _corridor_transit_windows(cfg)[cid]
                        if not (b <= s or a >= e))
                hits[min(n, 2)] += 1
                assert f_lo - 1e-9 <= f <= 1.0
                assert e > s
        assert hits[0] == 0, f"found zero-train incidents: {hits}"
        assert hits[2] > 0, f"expected multi-train incidents: {hits}"

    def test_incident_calibrated_recoverable(self):
        """The calibrated worst-case added delay stays within max_added_delay_sec,
        so the incident is recoverable by L3 speed control (not like bunching)."""
        cfg = load_scenario(scenario_name="five_station_15t")
        block = cfg.time_generation["corridor_incident"]
        max_added = block["max_added_delay_sec"]
        wins = _corridor_transit_windows(cfg)
        rng = random.Random(5)
        checked = 0
        for _ in range(500):
            inc = inject_corridor_incident(cfg, rng)
            if not inc:
                continue
            for cid, (s, e, f, _sf) in inc.items():
                transit = min(b - a for a, b in wins[cid])
                dur = e - s
                est = (transit * (1.0 / f - 1.0) if transit <= dur
                       else dur * (1.0 - f))
                assert est <= max_added + 1e-6, (
                    f"{cid}: est delay {est:.1f}s > max_added {max_added}s (f={f:.2f})")
                checked += 1
        assert checked > 0, "no incidents generated"


if __name__ == "__main__":
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"])
