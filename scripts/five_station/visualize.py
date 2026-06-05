#!/usr/bin/env python3
"""Text-based visualization of the 5-station multi-station railway network."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.five_station.run_heuristic import load_multistation_cfg
from src.control.dispatch_bounds import DispatchBounds


def main():
    cfg = load_multistation_cfg()
    l1 = DispatchBounds(cfg)

    # ==================================================================
    print(f"\n{'='*70}")
    print("  FIVE-STATION RAILWAY NETWORK")
    print(f"{'='*70}")

    # Network topology
    print(f"""
    A ──1200m/90s── B ──2400m/180s── C ──1200m/90s── D ──1200m/90s── E
    [3 platforms]  [3 platforms]  [4 platforms]  [2 platforms]  [3 platforms]
    dwell=60s      dwell=30s      dwell=60s      dwell=15s      dwell=30s
                                        ^^^^^^
                                        BOTTLENECK
""")

    # ==================================================================
    print(f"{'='*70}")
    print("  TRAIN ROUTES")
    print(f"{'='*70}")

    up_trains = [t for t in cfg.trains.values() if t.direction == "up"]
    down_trains = [t for t in cfg.trains.values() if t.direction == "down"]

    print(f"\n  Up (A→B→C→D→E): {len(up_trains)} trains")
    print(f"  {'Train':<8} {'Class':<6} {'Prio':<5} {'Type':<10} {'Depart A':>8} {'Arrive E':>8}")
    print(f"  {'-'*50}")
    for t in sorted(up_trains, key=lambda x: l1.for_train(x.id, 'A').t_dep_plan):
        b_a = l1.for_train(t.id, 'A')
        b_e = l1.for_train(t.id, 'E') if 'E' in [s for s in ['A','B','C','D','E'] if True] else None
        ptype = "pass" if t.pass_through else f"dwell={b_a.dwell_min:.0f}s"
        print(f"  {t.id:<8} {t.train_class:<6} {t.priority:<5} {ptype:<10} {b_a.t_dep_plan:>8.0f} {b_e.t_dep_plan:>8.0f}")

    print(f"\n  Down (E→D→C→B→A): {len(down_trains)} trains")
    print(f"  {'Train':<8} {'Class':<6} {'Prio':<5} {'Type':<10} {'Depart E':>8} {'Arrive A':>8}")
    print(f"  {'-'*50}")
    for t in sorted(down_trains, key=lambda x: l1.for_train(x.id, 'E').t_dep_plan):
        b_e = l1.for_train(t.id, 'E')
        b_a = l1.for_train(t.id, 'A')
        ptype = "pass" if t.pass_through else f"dwell={b_e.dwell_min:.0f}s"
        print(f"  {t.id:<8} {t.train_class:<6} {t.priority:<5} {ptype:<10} {b_e.t_dep_plan:>8.0f} {b_a.t_dep_plan:>8.0f}")

    # ==================================================================
    print(f"\n{'='*70}")
    print("  STATION D — BOTTLENECK ANALYSIS")
    print(f"{'='*70}")

    # Collect all trains at D
    print(f"\n  Trains passing through D (2 platforms, dwell=15s):")
    print(f"  {'Train':<8} {'Dir':<5} {'Prio':<4} {'arr_plan':>8} {'dep_plan':>8} {'Window':>15}")
    print(f"  {'-'*55}")

    d_entries = []
    for tid, t in cfg.trains.items():
        b = l1.for_train(tid, 'D')
        if b.t_dep_plan == 0 and b.t_arr_plan == 0:
            continue
        d_entries.append((tid, t.direction, b.prio_i, b.t_arr_plan, b.t_dep_plan))

    for tid, direction, prio, arr, dep in sorted(d_entries, key=lambda x: x[3]):
        window = f"[{arr:.0f}, {dep:.0f}]"
        direction_symbol = "↑" if direction == "up" else "↓"
        print(f"  {tid:<8} {direction_symbol:<5} {prio:<4} {arr:>8.0f} {dep:>8.0f} {window:>15}")

    # Show conflicts
    print(f"\n  Conflicts (overlapping windows):")
    conflicts = []
    for i, (tid1, _d1, _p1, arr1, dep1) in enumerate(d_entries):
        for j, (tid2, _d2, _p2, arr2, dep2) in enumerate(d_entries):
            if i >= j:
                continue
            if arr1 < dep2 and arr2 < dep1:
                overlap = min(dep1, dep2) - max(arr1, arr2)
                conflicts.append((tid1, tid2, overlap, arr1, dep1, arr2, dep2))

    if conflicts:
        for tid1, tid2, overlap, arr1, dep1, arr2, dep2 in sorted(conflicts, key=lambda x: x[2], reverse=True):
            print(f"  {tid1} [{arr1:.0f}-{dep1:.0f}] <-> {tid2} [{arr2:.0f}-{dep2:.0f}] overlap={overlap:.0f}s")
    else:
        print(f"  None — timetable is conflict-free at D")

    # ==================================================================
    print(f"\n{'='*70}")
    print("  STATION D TIMELINE")
    print(f"{'='*70}")

    # Gantt chart for D
    max_time = max(d[4] for d in d_entries) + 50 if d_entries else 600
    scale = 50  # chars for 600s

    print(f"\n  Time (s)  |")
    for tid, direction, prio, arr, dep in sorted(d_entries, key=lambda x: x[3]):
        bar_start = int(arr / max_time * scale)
        bar_end = int(dep / max_time * scale)
        bar = " " * bar_start + "█" * max(1, bar_end - bar_start)
        direction_symbol = "↑" if direction == "up" else "↓"
        print(f"  {tid:<8}   |{bar} {arr:.0f}→{dep:.0f} ({direction_symbol}, P{prio})")

    # Platform assignment visualization
    print(f"\n  Platform occupancy (greedy FCFS):")
    plat_1 = []  # list of (train, start, end)
    plat_2 = []

    for tid, direction, prio, arr, dep in sorted(d_entries, key=lambda x: x[3]):
        # Assign to first free platform
        assigned = False
        for plat, bookings in [("D_p1", plat_1), ("D_p2", plat_2)]:
            if all(not (arr < b_end and b_start < dep) for (_t, b_start, b_end) in bookings):
                bookings.append((tid, arr, dep))
                assigned = True
                break
        if not assigned:
            # Put on p1 anyway (conflict)
            plat_1.append((tid + " ⚠", arr, dep))

    for plat_name, bookings in [("D_p1", plat_1), ("D_p2", plat_2)]:
        bar = [" "] * scale
        for tid, arr, dep in bookings:
            start = int(arr / max_time * scale)
            end = int(dep / max_time * scale)
            for k in range(start, min(end, scale)):
                bar[k] = tid[0] if len(tid) < 6 else "*"
        print(f"  {plat_name}     |{''.join(bar)}")
        for tid, arr, dep in bookings:
            print(f"    {tid:<10} [{arr:.0f}-{dep:.0f}]")

    # ==================================================================
    print(f"\n{'='*70}")
    print("  FULL ROUTE — K641 (critical train, special dwell 120s at A)")
    print(f"{'='*70}")
    print(f"  {'Station':<10} {'arr_plan':>8} {'dep_plan':>8} {'dwell_min':>8} {'Platforms':>12}")
    print(f"  {'-'*55}")
    for st in ['A', 'B', 'C', 'D', 'E']:
        b = l1.for_train('K641', st)
        if b.t_arr_plan == 0 and b.t_dep_plan == 0:
            continue
        print(f"  {st:<10} {b.t_arr_plan:>8.0f} {b.t_dep_plan:>8.0f} {b.dwell_min:>8.0f} {str(b.candidate_platforms):>12}")

    print(f"\n  Total transit A→E: {l1.for_train('K641','E').t_dep_plan - l1.for_train('K641','A').t_arr_plan:.0f}s")
    print(f"  Last train arrives at: {max(d[4] for d in d_entries):.0f}s")
    print(f"  Suggested MAX_EPISODE_TIME: {max(d[4] for d in d_entries) + 120:.0f}s")
    print()


if __name__ == "__main__":
    main()
