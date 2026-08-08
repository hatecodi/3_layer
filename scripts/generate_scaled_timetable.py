#!/usr/bin/env python3
"""Generate scaled (more trains) timetable variants for the scalability study.

For each target count, add "shadow" trains that follow an existing template train
at a small time offset — creating corridor convoys / congestion (the point of the
scalability arm). Config-only expansion: writes
  configs/five_station_{N}t/{timetable,railway_resources,l3_physics}.yaml

Usage:
    python scripts/generate_scaled_timetable.py --counts 15 20 --delta 22
"""
from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path

import yaml

BASE = Path("configs/five_station")
OUT = Path("configs")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", type=int, nargs="+", default=[15, 20])
    ap.add_argument("--delta", type=float, default=22.0, help="shadow offset (s)")
    ap.add_argument("--jitter", type=float, default=8.0, help="per-shadow delta jitter (s)")
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
        # interleave shadows: cycle templates, alternate up/down, jittered deltas
        up_tpl = [t for t in trains if t["direction"] == "up"]
        dn_tpl = [t for t in trains if t["direction"] == "down"]
        added = []
        for i in range(need):
            tpl = (up_tpl if i % 2 == 0 else dn_tpl)[i % max(len(up_tpl), len(dn_tpl))]
            # ensure template exists in that direction
            pool = up_tpl if tpl["direction"] == "up" else dn_tpl
            tpl = pool[i % len(pool)]
            delta = args.delta + (i % 3 - 1) * args.jitter
            new_id = f"{tpl['id']}_s{i+1}"
            added.append(_shift(tpl, new_id, delta))
        variants["trains"] = trains + added
        out_dir = OUT / f"five_station_{target}t"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "timetable.yaml").write_text(
            yaml.safe_dump(variants, allow_unicode=True, sort_keys=False), encoding="utf-8")
        shutil.copy(BASE / "railway_resources.yaml", out_dir / "railway_resources.yaml")
        shutil.copy(BASE / "l3_physics.yaml", out_dir / "l3_physics.yaml")
        print(f"{target}t: added {need} shadows -> {out_dir / 'timetable.yaml'} "
              f"({len(variants['trains'])} trains)")


if __name__ == "__main__":
    main()
