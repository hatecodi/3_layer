#!/usr/bin/env python3
"""
Phase 2 heuristic baseline -- end-to-end without learning.

1. Load scenario config.
2. For each train, apply heuristic policy (earliest legal entry + first legal
   platform + min dwell) at its origin station.
3. Convert actions to WindowRequests.
4. Run deterministic conflict resolver (with proper resource-window rebuild).
5. Print granted windows and verify zero hard-constraint violations.

Usage:
    python -m src.experiment.run_heuristic
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .heuristic_planner import build_heuristic_plan
from .scenario_loader import ScenarioConfig, load_scenario, print_summary, ValidationError
from ..control.dispatch_bounds import DispatchBounds
from ..control.action_mask import heuristic_action, action_to_request, is_legal_action
from ..control.resource_manager import (
    WindowRequest,
    GrantResult,
    RebuildFn,
    resolve_all,
    build_conflict_graph,
)


# ---------------------------------------------------------------------------
# rebuild callback (Step 0 fix: creates proper _resource_windows per platform)
# ---------------------------------------------------------------------------

def _make_rebuild_fn(
    layer1: DispatchBounds,
    station_id: str,
    t_now: float,
    travel_time_to_throat: float,
    cfg: ScenarioConfig,
) -> RebuildFn:
    """Return a closure that rebuilds a WindowRequest for any (platform, shift)."""

    def rebuild(train_id: str, platform: str, shift: float,
                base_req: WindowRequest) -> Optional[WindowRequest]:
        train = layer1.trains[train_id]
        bounds = layer1.for_train(train_id, station_id)
        w_in = (base_req.W_in[0] + shift, base_req.W_in[1] + shift)
        w_plat = (base_req.W_plat[0] + shift, base_req.W_plat[1] + shift)
        w_out = (base_req.W_out[0] + shift, base_req.W_out[1] + shift)

        if w_in[0] < bounds.t_arr_min - 1e-6:
            return None
        if w_out[0] > bounds.t_dep_max + 1e-6:
            return None

        station = cfg.stations[station_id]
        dir_cfg = station.up if train.direction == "up" else station.down
        in_route = next((r for r in dir_cfg.in_routes
                         if r.platform == platform), None)
        out_route = next((r for r in dir_cfg.out_routes
                          if r.platform == platform), None)

        req = WindowRequest(
            train_id=train_id,
            platform=platform,
            W_in=w_in,
            W_plat=w_plat,
            W_out=w_out,
            W_corridor=(base_req.W_corridor[0] + shift,
                        base_req.W_corridor[1] + shift)
            if base_req.W_corridor else None,
            priority=base_req.priority,
            delay=base_req.delay + shift,
        )
        req._resource_windows = {
            f"{station_id}/{platform}": w_plat,
        }
        if in_route:
            req._resource_windows[f"{station_id}/{in_route.conflict_group}"] = w_in
        if out_route:
            req._resource_windows[f"{station_id}/{out_route.conflict_group}"] = w_out
        return req

    return rebuild


# ---------------------------------------------------------------------------
# single station scheduling
# ---------------------------------------------------------------------------

def schedule_station(
    layer1: DispatchBounds,
    station_id: str,
    train_ids: List[str],
    t_now: float,
    travel_time_to_throat: float,
    cfg: ScenarioConfig,
) -> Tuple[List[WindowRequest], Dict[str, int]]:
    requests: List[WindowRequest] = []
    violations: Dict[str, int] = {}

    for tid in sorted(train_ids,
                       key=lambda t: layer1.for_train(t, station_id).t_dep_plan):
        train = layer1.trains[tid]
        bounds = layer1.for_train(tid, station_id)

        try:
            action = heuristic_action(
                train, station_id, t_now, travel_time_to_throat, cfg)
        except ValueError:
            violations[tid] = 1
            continue

        if not is_legal_action(
            train, station_id, action, t_now, travel_time_to_throat, cfg):
            violations[tid] = 2
            continue

        req = action_to_request(
            train, station_id, action, t_now, travel_time_to_throat,
            current_delay=max(0.0, t_now - bounds.t_arr_plan),
            cfg=cfg,
        )
        requests.append(req)
        violations[tid] = 0

    return requests, violations


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        cfg = load_scenario()
    except ValidationError as e:
        print(f"Config error: {e}")
        raise SystemExit(1)

    print_summary(cfg)

    layer1 = DispatchBounds(cfg)

    bound_errors = layer1.validate_bounds()
    if bound_errors:
        print("Layer 1 bound errors:")
        for e in bound_errors:
            print(f"  - {e}")
    else:
        print("Layer 1 bounds valid")

    # ----- heuristic scheduling ------------------------------------------
    t_now = 0.0
    travel_to_throat = 30.0

    print(f"\n{'='*60}")
    print(f"  Heuristic Scheduling -- Origin Stations")
    print(f"{'='*60}")

    plan = build_heuristic_plan(cfg, t_now, travel_to_throat)
    all_requests = plan.requests
    all_violations = plan.violations
    grants = plan.grants
    held_ids = plan.held_train_ids

    for station_id in sorted(cfg.stations):
        station_train_ids = [
            tid for tid in layer1.train_ids
            if layer1.trains[tid].origin_station == station_id
        ]
        station_request_count = sum(
            1 for r in all_requests
            if layer1.trains[r.train_id].origin_station == station_id
        )
        station_violation_count = sum(
            1 for tid in station_train_ids
            if all_violations.get(tid, 0) > 0
        )
        print(f"\n  Station {station_id}: {station_request_count} requests, "
              f"{station_violation_count} violations")

    # ----- conflict detection --------------------------------------------
    conflict_graph = build_conflict_graph(all_requests)
    n_conflicts = sum(len(ns) for ns in conflict_graph.values()) // 2
    print(f"\n  Conflict graph: {len(all_requests)} nodes, {n_conflicts} edges")

    if n_conflicts > 0:
        for tid, neighbors in sorted(conflict_graph.items()):
            if neighbors:
                print(f"    {tid} conflicts with {sorted(neighbors)}")

    # ----- report --------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  {'Train':<8} {'Platform':<8} {'W_in':<22} {'W_plat':<22} "
          f"{'W_out':<22} {'Winner':<8}")
    print(f"{'='*70}")
    for g in grants:
        win_str = "OK" if g.winner else f"->{g.fallback_from or '?'}"
        print(f"  {g.train_id:<8} {g.platform:<8} "
              f"{g.W_in[0]:5.0f}-{g.W_in[1]:5.0f}    "
              f"{g.W_plat[0]:5.0f}-{g.W_plat[1]:5.0f}    "
              f"{g.W_out[0]:5.0f}-{g.W_out[1]:5.0f}    {win_str:<8}")
    if held_ids:
        print(f"\n  Held/replan trains: {held_ids}")

    # ----- Step 0: resource-overlap check --------------------------------
    print(f"\n  Resource-overlap check (post-resolution):")
    overlap_count = 0
    for i, g1 in enumerate(grants):
        for g2 in grants[i + 1:]:
            # Reconstruct requests for overlap checking
            r1 = WindowRequest(g1.train_id, g1.platform, g1.W_in, g1.W_plat, g1.W_out)
            r2 = WindowRequest(g2.train_id, g2.platform, g2.W_in, g2.W_plat, g2.W_out)
            # Use the grant info to infer resource windows
            train1 = layer1.trains[g1.train_id]
            train2 = layer1.trains[g2.train_id]
            station1 = train1.origin_station
            station2 = train2.origin_station
            st1 = cfg.stations[station1]
            st2 = cfg.stations[station2]
            dir1 = st1.up if train1.direction == "up" else st1.down
            dir2 = st2.up if train2.direction == "up" else st2.down
            in1 = next((rt for rt in dir1.in_routes if rt.platform == g1.platform), None)
            out1 = next((rt for rt in dir1.out_routes if rt.platform == g1.platform), None)
            in2 = next((rt for rt in dir2.in_routes if rt.platform == g2.platform), None)
            out2 = next((rt for rt in dir2.out_routes if rt.platform == g2.platform), None)

            # Check platform overlap (only if same station, same platform)
            if station1 == station2 and g1.platform == g2.platform:
                if g1.W_plat[0] < g2.W_plat[1] and g2.W_plat[0] < g1.W_plat[1]:
                    print(f"  ** OVERLAP: {g1.train_id}/{g2.train_id} on "
                          f"{station1}/{g1.platform} platform")
                    overlap_count += 1

            # Check throat overlap (same conflict group)
            for tag, rt1, rt2, w1, w2 in [
                ("throat-in", in1, in2, g1.W_in, g2.W_in),
                ("throat-out", out1, out2, g1.W_out, g2.W_out),
            ]:
                if rt1 and rt2 and rt1.conflict_group == rt2.conflict_group:
                    if w1[0] < w2[1] and w2[0] < w1[1]:
                        print(f"  ** OVERLAP: {g1.train_id}/{g2.train_id} on "
                              f"{tag} group {rt1.conflict_group}")
                        overlap_count += 1

    if overlap_count == 0:
        print(f"  OK: 0 resource overlaps")

    # ----- Step 0: fallback resource-ID check ----------------------------
    print(f"\n  Fallback resource-ID check:")
    for g in grants:
        if not g.winner and g.fallback_from:
            # Verify fallback platform != original
            if g.platform != g.fallback_from:
                print(f"  {g.train_id}: platform {g.fallback_from} -> {g.platform} "
                      f"(fallback OK)")
            else:
                print(f"  {g.train_id}: delayed on same platform {g.platform}")

    # ----- hard-bound check ----------------------------------------------
    print(f"\n  Hard-bound check:")
    n_fail = 0
    for g in grants:
        bounds = layer1.for_train(g.train_id, layer1.trains[g.train_id].origin_station)
        if g.W_in[0] < bounds.t_arr_min - 1e-6:
            print(f"  FAIL {g.train_id}: W_in start {g.W_in[0]:.0f} < "
                  f"arr_min {bounds.t_arr_min:.0f}")
            n_fail += 1
        if g.W_out[0] > bounds.t_dep_max + 1e-6:
            print(f"  FAIL {g.train_id}: W_out start {g.W_out[0]:.0f} > "
                  f"dep_max {bounds.t_dep_max:.0f}")
            n_fail += 1
    if n_fail == 0:
        print(f"  OK: All granted windows respect Layer 1 hard bounds")
    else:
        print(f"  {n_fail} hard-bound violations")

    print(f"\n  Heuristic baseline complete\n")


if __name__ == "__main__":
    main()
