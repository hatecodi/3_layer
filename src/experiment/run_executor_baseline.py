#!/usr/bin/env python3
"""Run the Phase 3 rule-based SUMO executor with heuristic grants."""
from __future__ import annotations

from .heuristic_planner import build_heuristic_plan
from .scenario_loader import load_scenario
from ..control.executor import RuleBasedExecutor


def main() -> None:
    cfg = load_scenario()
    plan = build_heuristic_plan(cfg)
    print(f"Heuristic grants: {len(plan.grants)}")
    if plan.held_train_ids:
        print(f"Held/replan trains: {plan.held_train_ids}")

    executor = RuleBasedExecutor(cfg)
    targets = executor.build_targets(plan.grants)
    result = executor.run(targets)

    print(f"Expected vehicles: {len(result.expected_vehicle_ids)}")
    print(f"Completed vehicles: {len(result.completed_vehicle_ids)}")
    print(f"Completion rate: {result.completion_rate:.3f}")
    print(f"Route file: {result.route_file}")
    print(f"Tripinfo: {result.tripinfo_file}")
    print(f"Actual events: {result.event_csv}")

    if result.completion_rate < 1.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
