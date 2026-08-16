#!/usr/bin/env python3
"""Generate scaled (more trains) timetable variants for the scalability study.

For each target count, add "shadow" trains that follow an existing template train.

Placement is DENSITY-AWARE: each shadow's origin departure is placed in the
densest window of its (origin_station, direction) group (greedy), so shadows
concentrate in already-dense departure intervals (convoys / congestion — the
point of the scalability arm).  `--min-spacing` guarantees no two trains in a
group share an origin departure within that gap, which rules out the
exact-duplicate schedules the old fixed-delta cycling could emit (e.g.
G1380_s1 == G1380_s7 when the jitter period aligned with the template cycle).

Optional per-target overrides:
  --slack         rewrite time_generation.dep_max_slack_sec (dynamic L2 slack)
  --bounds-slack  rewrite every stopping station's bounds dep_max = planned_dep + X

Config-only expansion: writes
  configs/five_station_{N}t/{timetable,railway_resources,l3_physics}.yaml

Usage:
    python scripts/generate_scaled_timetable.py --counts 20 --slack 30 --bounds-slack 60
"""
from __future__ import annotations

import argparse
import bisect
import copy
import shutil
from pathlib import Path

import yaml

BASE = Path("configs/five_station")
OUT = Path("configs")


def _origin_dep(train: dict) -> float:
    o = train["route"]["origin_station"]
    return float(train["planned"][o]["dep"])


def _shift(train: dict, new_id: str, delta: float) -> dict:
    s = copy.deepcopy(train)
    s["id"] = new_id
    for st, p in s["planned"].items():
        p["arr"] = p["arr"] + delta
        p["dep"] = p["dep"] + delta
    for st, b in s["bounds"].items():
        b["arr_min"] = b["arr_min"] + delta
        b["dep_max"] = b["dep_max"] + delta
    return s


def _dense_slot(group_deps: list, min_spacing: float, window: float,
                tpl_dep: float, horizon: float):
    """Greediest free origin-departure slot for a template in this group.

    A candidate slot must be >= min_spacing from every existing departure (no
    duplicates / over-tight packing).  Score = number of existing departures
    within `window` seconds of the slot; ties break to the LATER slot so convoys
    extend forward behind existing trains (placing a shadow before everyone at
    t=0 is not "dense" — it has no one ahead to contend with).
    """
    best, best_score = None, -1
    t = 0.0
    while t <= horizon:
        if all(abs(t - d) >= min_spacing for d in group_deps):
            score = sum(1 for d in group_deps if abs(d - t) <= window)
            if score > best_score or (score == best_score and best is not None and t > best):
                best, best_score = t, score
        t += min_spacing
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", type=int, nargs="+", default=[15, 20])
    ap.add_argument("--delta", type=float, default=22.0, help="fallback shadow offset (s)")
    ap.add_argument("--jitter", type=float, default=8.0, help="fallback delta jitter (s)")
    ap.add_argument("--min-spacing", type=float, default=12.0,
                    help="min gap between origin departures in the same group (s)")
    ap.add_argument("--window", type=float, default=60.0,
                    help="density window width (s)")
    ap.add_argument("--horizon", type=float, default=1200.0,
                    help="max origin-departure time considered for shadow placement (s)")
    ap.add_argument("--slack", type=float, default=None,
                    help="override time_generation.dep_max_slack_sec")
    ap.add_argument("--bounds-slack", type=float, default=None,
                    help="rewrite stopping-station bounds dep_max = planned_dep + X")
    ap.add_argument("--swap-cost", type=float, default=None,
                    help="override time_generation.swap_cost_sec (cross-platform fallback delay)")
    args = ap.parse_args()

    base_yaml = yaml.safe_load((BASE / "timetable.yaml").read_text(encoding="utf-8"))
    trains = base_yaml["trains"]
    n_base = len(trains)
    print(f"base timetable: {n_base} trains")

    for target in args.counts:
        if target <= n_base:
            print(f"skip {target} (<= {n_base})")
            continue
        need = target - n_base
        variants = copy.deepcopy(base_yaml)
        up_tpl = [t for t in trains if t["direction"] == "up"]
        dn_tpl = [t for t in trains if t["direction"] == "down"]

        # group existing origin departures by (origin_station, direction)
        groups: dict = {}
        for t in trains:
            groups.setdefault((t["route"]["origin_station"], t["direction"]),
                              []).append(_origin_dep(t))
        for g in groups.values():
            g.sort()

        added = []
        for i in range(need):
            pool = up_tpl if i % 2 == 0 else dn_tpl
            tpl = pool[i % len(pool)]
            key = (tpl["route"]["origin_station"], tpl["direction"])
            slot = _dense_slot(groups[key], args.min_spacing, args.window,
                               _origin_dep(tpl), args.horizon)
            if slot is None:
                slot = _origin_dep(tpl) + args.delta + (i % 3 - 1) * args.jitter
            delta = slot - _origin_dep(tpl)
            new_id = f"{tpl['id']}_s{i+1}"
            added.append(_shift(tpl, new_id, delta))
            bisect.insort(groups[key], slot)

        variants["trains"].extend(added)

        # optional slack overrides (20t only in practice)
        if args.slack is not None:
            variants["time_generation"]["dep_max_slack_sec"] = args.slack
        if args.bounds_slack is not None:
            for t in variants["trains"]:
                for st, pl in t["planned"].items():
                    b = t["bounds"].get(st)
                    if b and pl["dep"] > 0:
                        b["dep_max"] = pl["dep"] + args.bounds_slack
        if args.swap_cost is not None:
            variants["time_generation"]["swap_cost_sec"] = args.swap_cost

        out_dir = OUT / f"five_station_{target}t"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "timetable.yaml").write_text(
            yaml.safe_dump(variants, allow_unicode=True, sort_keys=False), encoding="utf-8")
        shutil.copy(BASE / "railway_resources.yaml", out_dir / "railway_resources.yaml")
        shutil.copy(BASE / "l3_physics.yaml", out_dir / "l3_physics.yaml")
        print(f"{target}t: added {need} shadows -> {out_dir / 'timetable.yaml'} "
              f"({len(variants['trains'])} trains)")

        # report placement
        o0: dict = {}
        for t in variants["trains"]:
            o0.setdefault((t["route"]["origin_station"], t["direction"]),
                          []).append((_origin_dep(t), t["id"]))
        print("  origin departures by (station, direction):")
        for k in sorted(o0):
            print(f"    {k[0]} {k[1]}: {[(d, tid) for d, tid in sorted(o0[k])]}")


if __name__ == "__main__":
    main()
