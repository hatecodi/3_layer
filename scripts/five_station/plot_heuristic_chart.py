#!/usr/bin/env python3
"""Heuristic running chart for latest five-station config."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.env.five_station_env import FiveStationEnv
from src.experiment.scenario_loader import load_scenario
from src.control.dispatch_bounds import DispatchBounds

cfg = load_scenario(scenario_name='five_station')
l1 = DispatchBounds(cfg)

# ---- Run heuristic ----
env = FiveStationEnv(cfg=cfg, seed=42, batch_window_sec=120.0)
obs, info = env.reset()
actual = {}
step = 0
seen = set()
while True:
    obs, reward, terminated, truncated, info = env.step({})
    step += 1
    for g in info.get('granted_windows', []):
        # Extract station from platform (e.g., "A_p1" → "A")
        st = g.platform.split('_')[0]
        key = (g.train_id, st)
        if key not in seen:
            actual.setdefault(g.train_id, []).append((st, g.W_in[0], g.W_out[0]))
            seen.add(key)
    if terminated or truncated:
        break

# ---- Planned data ----
planned = {}
for tid, train in cfg.trains.items():
    path = []
    stations = ['A','B','C','D','E'] if train.direction == 'up' else ['E','D','C','B','A']
    for st in stations:
        b = l1.for_train(tid, st)
        if b.t_arr_plan > 0 or b.t_dep_plan > 0 or st == train.origin_station:
            path.append((st, b.t_arr_plan, b.t_dep_plan))
    if path:
        planned[tid] = path

# ---- Total delay ----
total_delay = 0
for tid, path in actual.items():
    for st, arr, dep in path:
        b = l1.for_train(tid, st)
        if b.t_dep_plan > 0:
            total_delay += max(0, dep - b.t_dep_plan)

# ---- Colors ----
colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00',
          '#a65628','#f781bf','#999999','#66c2a5','#d73027']
tids = sorted(actual.keys())
color_of = {tid: colors[i % len(colors)] for i, tid in enumerate(tids)}

# ---- Station positions ----
corridor_km = {'A_B': 1.2, 'B_C': 2.4, 'C_D': 1.2, 'D_E': 1.2}
station_km = {'A': 0.0}
for st, prev in [('B','A'),('C','B'),('D','C'),('E','D')]:
    station_km[st] = station_km[prev] + corridor_km.get(f'{prev}_{st}', 1.2)

# ======== FIGURE ========
fig = plt.figure(figsize=(20, 13))
fig.suptitle(
    f'Five-Station Heuristic Baseline (Seed=42)  |  '
    f'10 trains, A/B/C/E=4p D=2p  |  '
    f'Total delay: {total_delay:.0f}s  |  Steps: {step}  |  '
    f'{len(env._completed_ids)}/10 complete',
    fontsize=11, fontweight='bold', y=0.98)

# (1) Main running chart
ax1 = fig.add_subplot(2, 3, (1, 3))

for tid, path in planned.items():
    if not path: continue
    times, kms = [], []
    for st, arr, dep in path:
        km = station_km[st]
        times.extend([arr, dep]); kms.extend([km, km])
    ax1.plot(times, kms, '--', color=color_of.get(tid,'#000'), linewidth=0.8, alpha=0.3)

for tid, path in actual.items():
    if not path: continue
    times, kms = [], []
    for st, arr, dep in path:
        km = station_km[st]
        times.extend([arr, dep]); kms.extend([km, km])
    ax1.plot(times, kms, '-', color=color_of.get(tid,'#000'), linewidth=1.8, alpha=0.9, label=tid)
    if path:
        ax1.annotate(tid, (path[0][1], station_km[path[0][0]]),
                    textcoords='offset points', xytext=(5, 5),
                    fontsize=6, color=color_of[tid], fontweight='bold')

for st, km in station_km.items():
    pc = len(cfg.stations[st].platforms)
    ax1.axhline(y=km, color='gray', linewidth=0.3, linestyle=':')
    ax1.text(-100, km, f'{st} ({pc}p)', fontsize=7, va='center', color='gray')

d_km = station_km['D']
ax1.axhspan(d_km - 0.3, d_km + 0.3, facecolor='red', alpha=0.05)
ax1.annotate('BOTTLENECK\n(2 platforms)', xy=(750, d_km), fontsize=8, color='red',
            ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

ax1.set_xlabel('Time (seconds)', fontsize=10)
ax1.set_ylabel('Distance from A (km)', fontsize=10)
ax1.set_title('Train Running Chart — Planned (dashed) vs Heuristic Actual (solid)',
             fontsize=12, fontweight='bold')
ax1.set_xlim(-100, 1600)
ax1.set_ylim(-0.2, max(station_km.values()) + 0.3)
ax1.legend(fontsize=6, loc='upper left', ncol=2)

# (2) Delay bars
ax2 = fig.add_subplot(2, 3, 4)
delay_data, delay_labels, delay_colors = [], [], []
for tid in tids:
    for st, arr, dep in actual.get(tid, []):
        b = l1.for_train(tid, st)
        if b.t_dep_plan > 0:
            d = max(0, dep - b.t_dep_plan)
            if d > 5:
                delay_data.append(d)
                delay_labels.append(f'{tid}@{st}')
                delay_colors.append(color_of[tid])
if delay_data:
    ax2.barh(range(len(delay_data)), delay_data, color=delay_colors, height=0.7)
    ax2.set_yticks(range(len(delay_data)))
    ax2.set_yticklabels(delay_labels, fontsize=6)
    ax2.set_xlabel('Departure delay (s)', fontsize=9)
    ax2.set_title('Station Departure Delays (>5s)', fontsize=11, fontweight='bold')
    ax2.invert_yaxis()
else:
    ax2.text(0.5, 0.5, 'All on time!', ha='center', va='center',
             transform=ax2.transAxes, fontsize=14)

# (3) Cumulative delay
ax3 = fig.add_subplot(2, 3, 5)
for tid in tids:
    path = actual.get(tid, [])
    if not path: continue
    cum, x, y = 0, [0], [0]
    for i, (st, arr, dep) in enumerate(path):
        b = l1.for_train(tid, st)
        if b.t_dep_plan > 0:
            cum += max(0, dep - b.t_dep_plan)
        x.append(i + 1); y.append(cum)
    ax3.plot(x, y, '-o', color=color_of[tid], linewidth=1.5, markersize=4, label=tid)
ax3.set_xlabel('Station index in route', fontsize=9)
ax3.set_ylabel('Cumulative delay (s)', fontsize=9)
ax3.set_title('Cumulative Delay Along Route', fontsize=11, fontweight='bold')
ax3.legend(fontsize=6, ncol=2)
ax3.grid(True, alpha=0.3)

# (4) D station occupancy Gantt
ax4 = fig.add_subplot(2, 3, 6)
d_trains = []
for tid, path in actual.items():
    for st, arr, dep in path:
        if st == 'D':
            d_trains.append((tid, arr, dep, color_of[tid]))
d_trains.sort(key=lambda x: x[1])
for i, (tid, arr, dep, color) in enumerate(d_trains):
    ax4.barh(i, dep - arr, left=arr, color=color, height=0.6,
             edgecolor='black', linewidth=0.3)
    ax4.text(arr + (dep - arr) / 2, i, tid, ha='center', va='center',
             fontsize=7, color='white', fontweight='bold')
ax4.set_yticks(range(len(d_trains)))
ax4.set_yticklabels([t[0] for t in d_trains], fontsize=7)
ax4.set_xlabel('Time (s)', fontsize=9)
ax4.set_title(f'D Station Occupancy ({len(d_trains)} trains)', fontsize=11, fontweight='bold')
ax4.set_xlim(300, 1000)

plt.tight_layout(rect=[0, 0, 1, 0.95])
out = Path(__file__).resolve().parent.parent.parent / 'output' / 'five_station' / 'heuristic_running_chart.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
plt.close('all')

# ===== D station detail =====
fig2, ax = plt.subplots(figsize=(14, 5))
for tid, path in actual.items():
    d_seg = [(s, a, d) for s, a, d in path if s in ('C', 'D', 'E')]
    if len(d_seg) < 2: continue
    times, kms = [], []
    for s, a, d in d_seg:
        km = station_km[s]
        times.extend([a, d]); kms.extend([km, km])
    ax.plot(times, kms, '-o', color=color_of[tid], linewidth=2, markersize=5, label=tid)
for s in ['C', 'D', 'E']:
    pc = len(cfg.stations[s].platforms)
    ax.axhline(y=station_km[s], color='gray', linewidth=0.3, linestyle=':')
    ax.text(200, station_km[s], f'{s} ({pc}p)', fontsize=9, va='center')
ax.axhspan(station_km['D'] - 0.3, station_km['D'] + 0.3, facecolor='red', alpha=0.08)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Distance from A (km)')
ax.set_title('D Station Bottleneck Detail — C-D-E Segment', fontweight='bold')
ax.legend(fontsize=7, ncol=3)
ax.set_xlim(200, 1100)
ax.set_ylim(station_km['C'] - 0.3, station_km['E'] + 0.3)

detail_out = Path(__file__).resolve().parent.parent.parent / 'output' / 'five_station' / 'station_D_detail.png'
fig2.savefig(detail_out, dpi=150, bbox_inches='tight')
print(f'Saved: {detail_out}')
plt.close('all')
print('Done.')
