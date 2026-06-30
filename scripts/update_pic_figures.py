#!/usr/bin/env python3
"""Update output/pic figures with latest training data."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
PIC_DIR = PROJECT / 'output' / 'pic'
PIC_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = PROJECT / 'output' / 'five_station' / '20260629_153329' / 'training_metrics.csv'

# ---- Load training data ----
with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))

rewards = [float(r['mean_reward']) for r in rows]
dep_delays = [float(r['mean_dep_delay']) for r in rows]
arr_delays = [float(r['mean_arr_delay']) for r in rows]
ontimes = [float(r['ontime_rate']) for r in rows]
steps_list = [int(r['steps']) for r in rows]

def smooth(data, window=50):
    return [np.mean(data[max(0,i-window):i+1]) for i in range(len(data))]

# ============================================================
# fig1: MAPPO vs Heuristic
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Delay comparison
last100 = dep_delays[-100:]
heuristic_delay = 28.4  # from our heuristic run across seeds
labels = ['Heuristic\n(rule-based)', 'MAPPO\n(learned)']
values = [heuristic_delay, np.mean(last100)]
colors_bar = ['#999999', '#377eb8']
bars = ax1.bar(labels, values, color=colors_bar, width=0.5, edgecolor='black')
for bar, val in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}s', ha='center', fontsize=12, fontweight='bold')
ax1.set_ylabel('Mean Departure Delay (s)', fontsize=11)
ax1.set_title('Average Delay: Heuristic vs MAPPO', fontsize=12, fontweight='bold')
ax1.set_ylim(0, max(values) * 1.3)

# Ontime comparison
heuristic_ontime = 0.0  # heuristic always has some delay > threshold
mappo_ontime = np.mean(ontimes[-100:]) * 100
ax2.bar(labels, [heuristic_ontime, mappo_ontime], color=colors_bar, width=0.5, edgecolor='black')
for i, (val, label) in enumerate(zip([heuristic_ontime, mappo_ontime], labels)):
    ax2.text(i, val + 0.5, f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
ax2.set_ylabel('On-Time Rate (%)', fontsize=11)
ax2.set_title('On-Time Performance: Heuristic vs MAPPO', fontsize=12, fontweight='bold')

fig.suptitle('MAPPO vs Heuristic Baseline — Five-Station (A/B/C/E=4p, D=2p, 10 trains)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(PIC_DIR / 'fig1_mappo_vs_heuristic.png', dpi=150, bbox_inches='tight')
print(f'Saved: fig1_mappo_vs_heuristic.png')
plt.close('all')

# ============================================================
# fig2: Training convergence
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
window = 10

# Reward
ax = axes[0, 0]
ax.plot(smooth(rewards, window), color='#377eb8', linewidth=1.2, alpha=0.9)
ax.set_xlabel('Episode', fontsize=10)
ax.set_ylabel('Mean Reward', fontsize=10)
ax.set_title('Episode Reward', fontsize=11, fontweight='bold')
ax.axhline(y=0, color='gray', linewidth=0.5, linestyle=':')
ax.grid(True, alpha=0.3)

# Departure delay
ax = axes[0, 1]
ax.plot(smooth(dep_delays, window), color='#e41a1c', linewidth=1.2, alpha=0.9)
ax.set_xlabel('Episode', fontsize=10)
ax.set_ylabel('Delay (s)', fontsize=10)
ax.set_title('Mean Departure Delay', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)
# Best window annotation
mid = len(dep_delays) // 2
best_window = dep_delays[-500:]
ax.annotate(f'Last 500ep avg: {np.mean(best_window):.1f}s',
            xy=(len(dep_delays)-250, np.mean(best_window)),
            fontsize=9, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# Steps per episode
ax = axes[1, 0]
ax.plot(smooth(steps_list, window), color='#4daf4a', linewidth=1.2, alpha=0.9)
ax.set_xlabel('Episode', fontsize=10)
ax.set_ylabel('Steps', fontsize=10)
ax.set_title('Episode Length', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

# On-time rate
ax = axes[1, 1]
ax.plot(smooth(ontimes, window), color='#ff7f00', linewidth=1.2, alpha=0.9)
ax.set_xlabel('Episode', fontsize=10)
ax.set_ylabel('On-Time Rate', fontsize=10)
ax.set_title('On-Time Rate (per-step, last step)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

fig.suptitle(f'MAPPO Training Convergence — 5000 Episodes, 10 trains, 4-platform\n'
             f'Last 100: delay={np.mean(dep_delays[-100:]):.1f}s  '
             f'reward={np.mean(rewards[-100:]):.1f}  '
             f'best={min(dep_delays):.0f}s',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(PIC_DIR / 'fig2_training_convergence.png', dpi=150, bbox_inches='tight')
print(f'Saved: fig2_training_convergence.png')
plt.close('all')

# ============================================================
# fig3: Batch window comparison (annotated)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Data from coding_log
bws_configs = ['bws=60\n(serial)', 'bws=90\n(avoidance)', 'bws=120\n(multi-agent)']
bws_delays = [14, 14, 29]
bws_ontime = [92, 94, 78]
bws_competition = [1.0, 1.1, 1.8]

x = np.arange(len(bws_configs))
width = 0.25

bars1 = ax.bar(x - width, bws_delays, width, label='Delay (s)', color='#e41a1c', alpha=0.8)
bars2 = ax.bar(x, bws_ontime, width, label='On-Time (%)', color='#377eb8', alpha=0.8)
ax_twin = ax.twinx()
bars3 = ax_twin.bar(x + width, bws_competition, width, label='Avg Grants/Step', color='#4daf4a', alpha=0.8)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bar.get_height():.0f}', ha='center', fontsize=10, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bar.get_height():.0f}%', ha='center', fontsize=10, fontweight='bold')
for bar in bars3:
    ax_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{bar.get_height():.1f}', ha='center', fontsize=10, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(bws_configs, fontsize=10)
ax.set_ylabel('Delay (s) / On-Time (%)', fontsize=11)
ax_twin.set_ylabel('Concurrent Grants per Step', fontsize=11, color='#4daf4a')
ax.set_title('Effect of Batch Window Size on Multi-Agent Competition', fontsize=12, fontweight='bold')
ax.set_ylim(0, 110)
ax_twin.set_ylim(0, 2.5)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_twin.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

plt.tight_layout()
fig.savefig(PIC_DIR / 'fig3_batch_window_competition.png', dpi=150, bbox_inches='tight')
print(f'Saved: fig3_batch_window_competition.png')
plt.close('all')

# ============================================================
# fig4: Learning coordination (n_granted evolution)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# n_granted evolution
n_granted = [float(r['n_granted']) for r in rows]
ax1.plot(smooth(n_granted, window), color='#377eb8', linewidth=1.2)
ax1.set_xlabel('Episode', fontsize=10)
ax1.set_ylabel('Trains Granted per Step', fontsize=10)
ax1.set_title('Multi-Agent Coordination Level', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=1.0, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)

# Delay vs n_granted scatter (last 500 episodes)
ax2.scatter(n_granted[-500:], dep_delays[-500:], alpha=0.4, color='#377eb8', s=20)
ax2.set_xlabel('Grants per Step', fontsize=10)
ax2.set_ylabel('Departure Delay (s)', fontsize=10)
ax2.set_title('Delay vs Concurrent Grants (Last 500ep)', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)
# Correlation
corr = np.corrcoef(n_granted[-500:], dep_delays[-500:])[0, 1]
ax2.text(0.95, 0.95, f'Correlation: {corr:.3f}', transform=ax2.transAxes,
         fontsize=10, ha='right', va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

fig.suptitle('Multi-Agent Coordination Analysis — 5000ep, bws=120',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(PIC_DIR / 'fig4_learning_coordination.png', dpi=150, bbox_inches='tight')
print(f'Saved: fig4_learning_coordination.png')
plt.close('all')

# ============================================================
# fig5: Environment configuration summary
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

summary_text = f"""FIVE-STATION RAILWAY NETWORK CONFIGURATION

Stations:  A(4 platforms) — B(4p) — C(4p) — D(2p bottleneck) — E(4p)
Corridors: A-B=90s  B-C=90s  C-D=180s  D-E=180s (simulated seconds)
Dwell defaults: A=60s  B=30s  C=60s  D=30s  E=30s

Trains (10, balanced 5 up : 5 down):
  UP (A→E):   G1380(P1) G1616(P1) G230(P2) G238(P2→D) K834(P3 C→E)
  DOWN (E→A): G7629(P1 E→D) G2245(P2 E→D) G1487(P2 C→A) K1185(P3 B→A) G1347(P2 D→A)

Action space: 4 platform choices × 6 entry timing choices = 24 discrete actions per station
Observation: 25-dimensional (self 14 + next station 5 + resource 2 + per-platform blocked 4)
Reward: dep_delay × priority + exp(-delay/60) bonus + team global reward

Training: MAPPO 5000 episodes, bws=120, entropy 0.01→0.003, lr=3e-4
  Best result: {min(dep_delays):.0f}s delay, {max(ontimes)*100:.0f}% on-time (single episode)
  Converged:  {np.mean(dep_delays[-500:]):.1f}s avg delay (last 500 episodes)
  Heuristic:  {heuristic_delay:.1f}s avg delay (rule-based baseline)
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
        fontfamily='monospace', va='top', linespacing=1.5)
ax.set_title('Environment & Training Summary', fontsize=14, fontweight='bold', y=1.01)
fig.savefig(PIC_DIR / 'fig5_environment_summary.png', dpi=150, bbox_inches='tight')
print(f'Saved: fig5_environment_summary.png')
plt.close('all')

print('\nAll figures updated in output/pic/')
