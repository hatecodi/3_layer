"""Trace L2 output: per train, per station, platform, time, dwell."""
from __future__ import annotations

import json
from collections import defaultdict

from src.env.five_station_env import FiveStationEnv
from src.experiment.scenario_loader import load_scenario
from src.control.action_mask import ENTRY_CHOICES, heuristic_action


def run_trace():
    cfg = load_scenario(scenario_name="five_station")
    env = FiveStationEnv(cfg, seed=42)

    obs, info = env.reset()
    print(f"Episode start, t_now=0, {len(env._pending_trains())} pending trains")
    print("=" * 100)

    # Collect per-station grant records
    station_records: dict[str, list] = defaultdict(list)

    step = 0
    while True:
        active = info.get("active_train_ids", [])
        if not active and env._episode_status.value != "running":
            break

        # Use heuristic actions for deterministic trace
        action_dict = {}
        for tid in active:
            train = cfg.trains[tid]
            st = env._journeys[tid].current_station
            try:
                bounds = env._get_effective_bounds(tid, st, env._t_now)
                plat, entry, dwell = heuristic_action(
                    train, st, env._t_now, 0.0, cfg, bounds_override=bounds
                )
                # Map to factored action indices
                platforms = train.candidate_platforms.get(st, [])
                plat_idx = platforms.index(plat) if plat in platforms else 0
                entry_idx = ENTRY_CHOICES.index(entry) if entry in ENTRY_CHOICES else 0
                action_dict[tid] = (plat_idx, entry_idx)
            except ValueError:
                pass

        obs, reward, terminated, truncated, info = env.step(action_dict)
        step += 1

        grants = info.get("granted_windows", [])
        requests = info.get("requests", [])
        held = info.get("held_train_ids", [])

        for g in grants:
            station_id = env._journeys[g.train_id].current_station if not env._journeys[
                g.train_id
            ].is_done else "?"
            station_records[station_id].append({
                "train_id": g.train_id,
                "station": station_id,
                "platform": g.platform,
                "W_in": [round(g.W_in[0], 1), round(g.W_in[1], 1)],
                "W_plat": [round(g.W_plat[0], 1), round(g.W_plat[1], 1)],
                "W_out": [round(g.W_out[0], 1), round(g.W_out[1], 1)],
                "dwell_sec": round(g.W_out[0] - g.W_plat[0], 1),
                "platform_occupy_sec": round(g.W_out[1] - g.W_in[0], 1),
                "winner": g.winner,
                "fallback_from": g.fallback_from,
            })

        if terminated or truncated:
            break

    # Print per-station summary
    for sid in sorted(station_records.keys()):
        records = station_records[sid]
        print(f"\n{'=' * 100}")
        print(f"Station {sid}: {len(records)} grants")
        print(f"{'train':>8} {'platform':>8} {'W_in':>14} {'W_plat':>14} {'W_out':>14} "
              f"{'dwell':>8} {'occupy':>8} {'winner':>6} {'fallback':>10}")
        print("-" * 100)
        for r in sorted(records, key=lambda x: x["W_in"][0]):
            fb = r["fallback_from"] or ""
            print(f"{r['train_id']:>8} {r['platform']:>8} "
                  f"{r['W_in'][0]:>6.0f}-{r['W_in'][1]:>5.0f} "
                  f"{r['W_plat'][0]:>6.0f}-{r['W_plat'][1]:>5.0f} "
                  f"{r['W_out'][0]:>6.0f}-{r['W_out'][1]:>5.0f} "
                  f"{r['dwell_sec']:>6.0f}s {r['platform_occupy_sec']:>6.0f}s "
                  f"{'W' if r['winner'] else 'L':>6} {fb:>10}")

    # Per-train summary
    print(f"\n{'=' * 100}")
    print("Per-train itinerary:")
    print(f"{'train':>8} {'direction':>9} {'stations':>30} {'platforms':>30} {'dwells':>25}")
    print("-" * 100)
    train_records: dict[str, list] = defaultdict(list)
    for sid_recs in station_records.values():
        for r in sid_recs:
            train_records[r["train_id"]].append(r)
    for tid in sorted(train_records.keys()):
        recs = sorted(train_records[tid], key=lambda x: x["W_in"][0])
        train = cfg.trains[tid]
        stations = " → ".join(r["station"] for r in recs)
        platforms = " → ".join(r["platform"] for r in recs)
        dwells = " → ".join(f"{r['dwell_sec']:.0f}s" for r in recs)
        print(f"{tid:>8} {train.direction:>9} {stations:>30} {platforms:>30} {dwells:>25}")

    # Count by platform per station (utilization check)
    print(f"\n{'=' * 100}")
    print("Platform usage per station:")
    for sid in sorted(station_records.keys()):
        plat_count: dict[str, int] = defaultdict(int)
        for r in station_records[sid]:
            plat_count[r["platform"]] += 1
        total = len(station_records[sid])
        total_plats = len(cfg.stations[sid].platforms)
        usage_pct = len(plat_count) / total_plats * 100
        print(f"  {sid}: {dict(plat_count)} ({len(plat_count)}/{total_plats} platforms used = {usage_pct:.0f}%, "
              f"{total} total grants)")


if __name__ == "__main__":
    run_trace()
