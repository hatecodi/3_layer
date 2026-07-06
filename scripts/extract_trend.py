"""Extract training trend data and comparison tables."""
import csv, sys
import numpy as np
from pathlib import Path

csv_path = sys.argv[1] if len(sys.argv) > 1 else None
if csv_path is None:
    csvs = sorted(Path("output/five_station").glob("*/training_metrics.csv"),
                  key=lambda p: p.stat().st_mtime)
    csv_path = str(csvs[-1])

rows = []
with open(csv_path) as f:
    for r in csv.DictReader(f):
        rows.append(r)

print("=" * 65)
print("  MAPPO 5000ep Training Trend")
print("=" * 65)
print(f"{'episodes':>12} {'reward':>8} {'ontime':>8} {'dep_dly':>8} {'arr_dly':>8} {'util':>8}")
print("-" * 55)

for i in range(0, len(rows), 500):
    chunk = rows[i:i+500]
    rw = np.mean([float(r['mean_reward']) for r in chunk])
    ot = np.mean([float(r['ontime_rate']) for r in chunk])
    dd = np.mean([float(r['mean_dep_delay']) for r in chunk])
    ad = np.mean([float(r['mean_arr_delay']) for r in chunk])
    pu = np.mean([float(r.get('platform_utilization',0)) for r in chunk])
    s = int(chunk[0]['episode'])
    e = int(chunk[-1]['episode'])
    print(f"{s:>5}-{e:<5} {rw:>8.1f} {ot*100:>7.1f}% {dd:>7.1f}s {ad:>7.1f}s {pu*100:>6.1f}%")

# Best
best = max(rows, key=lambda r: float(r['mean_reward']))
print(f"\nBest: ep={best['episode']} reward={best['mean_reward']} ontime={float(best['ontime_rate'])*100:.1f}% dep_dly={best['mean_dep_delay']}s")

# Variance
rewards = np.array([float(r['mean_reward']) for r in rows])
chunk_means = [np.mean(rewards[i:i+500]) for i in range(0, len(rewards), 500)]
print(f"\nReward std: first500={np.std(rewards[:500]):.1f} last500={np.std(rewards[-500:]):.1f}")
print(f"Reward 500ep chunk range: {min(chunk_means):.1f} to {max(chunk_means):.1f}")

# Priority breakdown
last_200 = rows[-200:]
print(f"\nLast 200ep summary:")
print(f"  Mean reward:   {np.mean([float(r['mean_reward']) for r in last_200]):.1f}")
print(f"  Mean ontime:   {np.mean([float(r['ontime_rate']) for r in last_200])*100:.1f}%")
print(f"  Mean dep_dly:  {np.mean([float(r['mean_dep_delay']) for r in last_200]):.1f}s")
print(f"  Mean util:     {np.mean([float(r.get('platform_utilization',0)) for r in last_200])*100:.1f}%")
print(f"  Deadlocks:     {sum(1 for r in last_200 if r.get('status')=='deadlock')}")
