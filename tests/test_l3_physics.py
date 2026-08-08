"""L3 v2 physics tests: calibration + moving-block headway (sections as speed zones).

Run:
    python -m pytest tests/test_l3_physics.py -v
"""
from __future__ import annotations

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
            cruise_run = sum(s.length_m / speeds.cruise_mps for s in secs)
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
        a safety distance ahead of the entry point (NOT after clearing a whole
        section).  Entry gap = safety_m/v + margin, much tighter than fixed-block."""
        sim, _ = self._sim()
        speeds = sim.p.get_speeds(2)
        safety_time = sim.p.moving_block_safety_m / speeds.cruise_mps
        # T1 enters corridor at 100+3=103; it is a safety distance ahead at
        # 103 + safety_time; T2 may enter then, plus the reaction margin.
        sim.transit("T1", "A", "B", "A_to_B", 76, 100.0, SpeedMode.CRUISE, None, 2)
        earliest = 103.0 + safety_time + sim.p.moving_block_margin_sec
        r2 = sim.transit("T2", "A", "B", "A_to_B", 76, 100.0, SpeedMode.CRUISE, None, 2)
        entry2 = 100.0 + 3.0 + r2.headway_delay_sec
        assert abs(entry2 - earliest) < 1e-6, (
            f"entry2 {entry2:.3f} != earliest {earliest:.3f}")
        # much tighter than fixed-block's whole-section0 wait (~16.75s + margin)
        assert r2.headway_delay_sec < 10.0

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


if __name__ == "__main__":
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"])
