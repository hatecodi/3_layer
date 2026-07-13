#!/usr/bin/env python3
"""Training curves — SCI style — hierarchical vs flat vs heuristic+L3."""
import sys, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 13,
    'axes.titlesize': 14, 'legend.fontsize': 10, 'xtick.labelsize': 10,
    'ytick.labelsize': 10, 'axes.linewidth': 1.0, 'axes.unicode_minus': False,
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
})

WINDOW = 80

def smooth(data, w=WINDOW):
    return np.convolve(data, np.ones(w)/w, mode='valid')

def load(path):
    rows = list(csv.DictReader(open(path)))
    ep = np.array([int(r['episode']) for r in rows])
    dep_key = 'mean_dep_delay' if 'mean_dep_delay' in rows[0] else 'dep_delay'
    ont_key = 'ontime_rate' if 'ontime_rate' in rows[0] else 'ontime_rate'
    dep = np.array([float(r[dep_key]) for r in rows])
    ont = np.array([float(r[ont_key]) for r in rows])
    return ep, dep, ont, [r.get('phase','') for r in rows]

# ---- Hierarchical (A+B+C) ----
h_ep, h_dep, h_ont, h_ph = load('output/five_station_hierarchical/20260713_234114/training_metrics.csv')
h_ep_s = h_ep[WINDOW-1:]; h_dep_s = smooth(h_dep); h_ont_s = smooth(h_ont)

# ---- Flat MAPPO (72-action joint) ----
f_ep, f_dep, f_ont, _ = load('src/output/flat_mappo/20260713_211733/training_metrics.csv')
f_ep_s = f_ep[WINDOW-1:]; f_dep_s = smooth(f_dep); f_ont_s = smooth(f_ont)

# ---- Heuristic L2 + MAPPO L3 ----
hl_ep, hl_dep, hl_ont, _ = load('output/figures/heuristic_l2_l3_curve.csv')
hl_ep_s = hl_ep[WINDOW//4-1:]  # eval every 100ep, so smaller window
hl_dep_s = smooth(hl_dep, WINDOW//4)
hl_ont_s = smooth(hl_ont, WINDOW//4)

# Reference
H_DELAY = 21.0  # G1(350/400) + G(300/350) + K(120)

# Phase transitions for hierarchical
h_trans = []
for i in range(1, len(h_ph)):
    if h_ph[i] != h_ph[i-1]:
        h_trans.append(int(h_ep[i]))
h_all = [0] + h_trans + [max(h_ep)]

# Phase averages
h_stats = {}
for p in ['A', 'B', 'C']:
    idx = [i for i, ph in enumerate(h_ph) if ph == p]
    if idx:
        h_stats[p] = (float(np.mean(h_dep[idx])), float(np.mean(h_ont[idx])))

# === Plot ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

phase_colors = {'A': '#fff5f0', 'B': '#f0fff0', 'C': '#f0f0ff'}
phase_labels = {'A': 'A: L2 only', 'B': 'B: L3 (L2 frozen)', 'C': 'C: Joint'}

for i in range(len(h_all)-1):
    e0, e1 = h_all[i], h_all[i+1]
    pname = h_ph[min(e0, len(h_ph)-1)]
    for ax in [ax1, ax2]:
        ax.axvspan(e0, e1, alpha=0.55, color=phase_colors.get(pname, '#fff'),
                   edgecolor='#ddd', linewidth=0.5)
    mid = (e0 + e1) / 2
    ax1.text(mid, 52, phase_labels.get(pname, ''), ha='center', va='top',
             fontsize=8.5, color='#444', style='italic')

# -- Delay --
ax1.plot(h_ep_s, h_dep_s, color='#d62728', linewidth=1.6, alpha=0.9, label='Hierarchical MAPPO (A+B+C)')
ax1.plot(f_ep_s, f_dep_s, color='#1f77b4', linewidth=1.6, alpha=0.85, label='Flat MAPPO (72-action joint)')
ax1.plot(hl_ep_s, hl_dep_s, color='#2ca02c', linewidth=1.6, alpha=0.85, linestyle='--',
         label='Heuristic L2 + MAPPO L3')
ax1.axhline(y=H_DELAY, color='#555', linewidth=1.2, linestyle='--', alpha=0.6)
ax1.annotate(f'Heuristic L2 ({H_DELAY:.0f}s)', xy=(6800, H_DELAY),
             fontsize=9, color='#555', va='bottom', ha='right',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#ccc', alpha=0.9))

for p, (d, o) in h_stats.items():
    e0 = h_all[['A','B','C'].index(p)]
    e1 = h_all[['A','B','C'].index(p)+1]
    ax1.annotate(f'{d:.1f}s', xy=((e0+e1)/2, d), fontsize=9, ha='center', va='bottom',
                 color='#d62728', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='#d62728', alpha=0.85))

ax1.set_ylabel('Mean Departure Delay (s)')
ax1.set_ylim(0, 58)
ax1.legend(loc='upper right', framealpha=0.9, edgecolor='#ccc')
ax1.grid(axis='y', alpha=0.2, linewidth=0.5)
for s in ['top', 'right']: ax1.spines[s].set_visible(False)

# -- Ontime --
ax2.plot(h_ep_s, h_ont_s, color='#d62728', linewidth=1.6, alpha=0.9, label='Hierarchical MAPPO (A+B+C)')
ax2.plot(f_ep_s, f_ont_s, color='#1f77b4', linewidth=1.6, alpha=0.85, label='Flat MAPPO (72-action joint)')
ax2.plot(hl_ep_s, hl_ont_s, color='#2ca02c', linewidth=1.6, alpha=0.85, linestyle='--',
         label='Heuristic L2 + MAPPO L3')

for p, (d, o) in h_stats.items():
    e0 = h_all[['A','B','C'].index(p)]
    e1 = h_all[['A','B','C'].index(p)+1]
    ax2.annotate(f'{o:.3f}', xy=((e0+e1)/2, o), fontsize=9, ha='center', va='bottom',
                 color='#d62728', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='#d62728', alpha=0.85))

ax2.set_ylabel('Ontime Rate')
ax2.set_xlabel('Episode')
ax2.set_ylim(0.35, 1.02)
ax2.legend(loc='lower right', framealpha=0.9, edgecolor='#ccc')
ax2.grid(axis='y', alpha=0.2, linewidth=0.5)
ax2.set_xlim(-100, 7200)
for s in ['top', 'right']: ax2.spines[s].set_visible(False)

plt.tight_layout()
outdir = Path('output/figures')
outdir.mkdir(parents=True, exist_ok=True)
outpath = outdir / 'training_curves.png'
plt.savefig(outpath, dpi=200)
print(f'Saved: {outpath}')

# ---- Save all data tables ----
# 2x2 comparison matrix
with open(outdir / 'comparison_matrix.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['', 'L3 CRUISE (no L3)', 'L3 MAPPO'])
    w.writerow(['L2 Heuristic', f'{H_DELAY:.1f}s', '25.6s'])
    a_d = h_stats['A'][0]; b_d = h_stats['B'][0]
    w.writerow(['L2 MAPPO', f'{a_d:.1f}s', f'{b_d:.1f}s'])

# Phase summary
with open(outdir / 'phase_summary.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Phase', 'Delay (s)', 'Ontime', 'vs Heuristic'])
    for p in ['A', 'B', 'C']:
        if p in h_stats:
            d, o = h_stats[p]
            w.writerow([p, f'{d:.1f}', f'{o:.3f}', f'{d/H_DELAY*100-100:+.0f}%'])

print(f'Data tables saved to {outdir}/')
for f in sorted(outdir.glob('*.csv')):
    print(f'  {f.name}')
print(f'\nPhase summary:')
for p in ['A', 'B', 'C']:
    if p in h_stats:
        d, o = h_stats[p]
        print(f'  {p}: delay={d:.1f}s  ontime={o:.3f}  vs heuristic {d/H_DELAY*100-100:+.0f}%')
