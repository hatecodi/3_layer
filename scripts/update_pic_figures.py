#!/usr/bin/env python3
"""Update output/pic_of_3 figures with latest training data."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
PIC_DIR = PROJECT / 'output' / 'pic_of_3'
PIC_DIR.mkdir(parents=True, exist_ok=True)

# Latest best run
CSV_PATH = PROJECT / 'output' / 'five_station' / '20260629_223519' / 'training_metrics_8k.csv'
with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))

rewards = [float(r['mean_reward']) for r in rows]
dep_delays = [float(r['mean_dep_delay']) for r in rows]
arr_delays = [float(r['mean_arr_delay']) for r in rows]
ontimes = [float(r['ontime_rate']) for r in rows]
steps_list = [int(r['steps']) for r in rows]
n_granted = [float(r['n_granted']) for r in rows]

def smooth(data, window=50):
    return [np.mean(data[max(0,i-window):i+1]) for i in range(len(data))]

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.titlesize': 14,
})

# ============================================================
# fig2: Training convergence
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
window = 50

# Reward
ax = axes[0, 0]
ax.plot(smooth(rewards, window), color='#377eb8', linewidth=1.0, alpha=0.95)
ax.axhline(y=0, color='gray', linewidth=0.5, linestyle=':')
# Annotate start / end
ax.annotate(f'Start: {np.mean(rewards[:100]):.0f}', xy=(50, np.mean(rewards[:100])),
            fontsize=9, color='#e41a1c', fontweight='bold')
mid_start = len(rewards) * 3 // 10
ax.annotate(f'End: {np.mean(rewards[-100:]):.1f}', xy=(len(rewards)-150, np.mean(rewards[-100:])),
            fontsize=9, color='#4daf4a', fontweight='bold')
ax.set_xlabel('Episode')
ax.set_ylabel('Mean Reward')
ax.set_title('Episode Reward')
ax.grid(True, alpha=0.3)

# Departure delay
ax = axes[0, 1]
ax.plot(smooth(dep_delays, window), color='#e41a1c', linewidth=1.0, alpha=0.95)
ax.set_xlabel('Episode')
ax.set_ylabel('Delay (s)')
ax.set_title('Mean Departure Delay')
ax.grid(True, alpha=0.3)
# Best window
best_window = dep_delays[-500:]
ax.annotate(f'Last 500ep: {np.mean(best_window):.1f}s',
            xy=(len(dep_delays)-250, np.mean(best_window)),
            fontsize=10, ha='center', fontweight='bold', color='#4daf4a',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
ax.annotate(f'Best: {min(dep_delays):.0f}s', xy=(np.argmin(dep_delays), min(dep_delays)),
            fontsize=8, ha='center', fontweight='bold', color='#377eb8')

# Steps per episode
ax = axes[1, 0]
ax.plot(smooth(steps_list, window), color='#4daf4a', linewidth=1.0, alpha=0.95)
ax.set_xlabel('Episode')
ax.set_ylabel('Steps')
ax.set_title('Episode Length')
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

# n_granted (coordination metric)
ax = axes[1, 1]
ax.plot(smooth(n_granted, window), color='#984ea3', linewidth=1.0, alpha=0.95)
ax.set_xlabel('Episode')
ax.set_ylabel('Grants per Step')
ax.set_title('Multi-Agent Competition Level')
ax.grid(True, alpha=0.3)
ax.axhline(y=1.0, color='gray', linewidth=0.5, linestyle=':', alpha=0.5, label='single-agent baseline')

fig.suptitle(f'MAPPO Training Convergence — 8000 Episodes\n'
             f'9 trains, A/B/E=3p C=4p D=2p, visitors@B/C/D, bws=120',
             fontweight='bold')
plt.tight_layout()
fig.savefig(PIC_DIR / 'fig2_training_convergence.png', dpi=150, bbox_inches='tight')
print(f'Saved: fig2_training_convergence.png')
plt.close('all')

# ============================================================
# fig3: MAPPO convergence — delay + ontime curves
# ============================================================
heuristic_per_step_delay = 69.6  # from 10-seed run
heuristic_ontime = 0.43
mappo_per_step_delay = np.mean(dep_delays[-500:])
mappo_per_step_ontime = np.mean(ontimes[-500:])

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
curve_window = 100

# (a) Delay bar comparison
ax = axes[0]
labels = ['Heuristic\n(rule-based)', 'MAPPO\n(learned)']
values = [heuristic_per_step_delay, mappo_per_step_delay]
colors_bar = ['#999999', '#377eb8']
bars = ax.bar(labels, values, color=colors_bar, width=0.45, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{val:.1f}s', ha='center', fontsize=14, fontweight='bold')
improvement = (1 - mappo_per_step_delay / heuristic_per_step_delay) * 100
ax.set_ylabel('Mean Departure Delay (s)', fontsize=11)
ax.set_title(f'Delay: MAPPO {improvement:.0f}% lower', fontsize=12, fontweight='bold')
ax.set_ylim(0, max(values) * 1.25)

# (b) Delay convergence curve
ax = axes[1]
delay_smooth = smooth(dep_delays, curve_window)
episodes = list(range(len(delay_smooth)))
ax.plot(episodes, delay_smooth, color='#e41a1c', linewidth=1.5, alpha=0.95)
ax.axhline(y=heuristic_per_step_delay, color='#999999', linewidth=1.2, linestyle='--', alpha=0.8)
ax.text(len(episodes) * 0.6, heuristic_per_step_delay + 3,
        f'Heuristic: {heuristic_per_step_delay:.0f}s', fontsize=9, color='#666666')
# Phase background
for x0, x1, label, c in [
    (0, len(episodes)*0.3, 'Exploration\n(high entropy)', '#f0f0f0'),
    (len(episodes)*0.6, len(episodes), 'Exploitation\n(low entropy)', '#f5f5f5'),
]:
    ax.axvspan(x0, x1, alpha=0.3, color=c)
    ax.text((x0+x1)/2, ax.get_ylim()[1]*0.55, label, ha='center', fontsize=8, color='#999999')
# Annotate start + end
ax.annotate(f'{np.mean(dep_delays[:100]):.0f}s',
            xy=(50, np.mean(dep_delays[:100])), fontsize=9, color='#e41a1c', fontweight='bold')
ax.annotate(f'{np.mean(dep_delays[-100:]):.1f}s',
            xy=(len(episodes)-150, np.mean(dep_delays[-100:])),
            fontsize=10, color='#4daf4a', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax.set_xlabel('Episode')
ax.set_ylabel('Departure Delay (s)')
ax.set_title('Delay Convergence', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, len(episodes))

# (c) Ontime convergence curve
ax = axes[2]
ontime_smooth = smooth(ontimes, curve_window)
ax.plot(episodes, [o * 100 for o in ontime_smooth], color='#4daf4a', linewidth=1.5, alpha=0.95)
ax.axhline(y=heuristic_ontime * 100, color='#999999', linewidth=1.2, linestyle='--', alpha=0.8)
ax.text(len(episodes) * 0.6, heuristic_ontime * 100 + 2,
        f'Heuristic: {heuristic_ontime*100:.0f}%', fontsize=9, color='#666666')
# Phase background
for x0, x1, _, c in [
    (0, len(episodes)*0.3, '', '#f0f0f0'),
    (len(episodes)*0.6, len(episodes), '', '#f5f5f5'),
]:
    ax.axvspan(x0, x1, alpha=0.3, color=c)
# Annotate start + end
ax.annotate(f'{np.mean(ontimes[:100])*100:.0f}%',
            xy=(50, np.mean(ontimes[:100])*100), fontsize=9, color='#e41a1c', fontweight='bold')
ax.annotate(f'{np.mean(ontimes[-100:])*100:.1f}%',
            xy=(len(episodes)-150, np.mean(ontimes[-100:])*100),
            fontsize=10, color='#4daf4a', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax.set_xlabel('Episode')
ax.set_ylabel('On-Time Rate (%)')
ax.set_title('On-Time Rate Convergence', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, len(episodes))
ax.set_ylim(0, 105)

fig.suptitle(f'MAPPO vs Heuristic — Five-Station Scheduling\n'
             f'9 trains, A/B/E=3p C=4p D=2p, visitors@B/C/D, bws=120',
             fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(PIC_DIR / 'fig3_mappo_vs_heuristic.png', dpi=150, bbox_inches='tight')
print(f'Saved: fig3_mappo_vs_heuristic.png')
plt.close('all')

# ============================================================
# fig4: Coordination analysis
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# (a) n_granted evolution with phase annotations
ax = axes[0, 0]
smoothed_ng = smooth(n_granted, window)
ax.plot(smoothed_ng, color='#984ea3', linewidth=1.0, alpha=0.95)
ax.set_xlabel('Episode')
ax.set_ylabel('Grants per Step')
ax.set_title('Multi-Agent Competition Level', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=1.0, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
# Phase annotations
phases = {
    'Exploration\n(entropy high)': (0, 1500),
    'Transition\n(entropy decay)': (1500, 3000),
    'Exploitation\n(entropy low)': (3000, 5000),
}
for label, (x0, x1) in phases.items():
    mid = (x0 + x1) / 2
    ax.axvspan(x0, x1, alpha=0.06)
    ax.text(mid, ax.get_ylim()[1] * 0.95, label, ha='center', fontsize=8, color='gray')

# (b) Delay vs n_granted scatter
ax = axes[0, 1]
sc = ax.scatter(n_granted[-500:], dep_delays[-500:], alpha=0.4, color='#377eb8', s=20,
                edgecolors='none')
# Binned average
bins = np.linspace(min(n_granted[-500:]), max(n_granted[-500:]), 8)
bin_means = []
for i in range(len(bins)-1):
    mask = (n_granted[-500:] >= bins[i]) & (n_granted[-500:] < bins[i+1])
    if sum(mask) > 0:
        bin_means.append((bins[i] + bins[i+1])/2)
ax.plot([bins[i] + (bins[1]-bins[0])/2 for i in range(len(bins)-1)],
        [np.mean([dep_delays[-500:][j] for j in range(len(dep_delays[-500:]))
                  if bins[i] <= n_granted[-500:][j] < bins[i+1]]) for i in range(len(bins)-1)],
        '-', color='#e41a1c', linewidth=2, label='binned avg')
ax.set_xlabel('Grants per Step')
ax.set_ylabel('Delay (s)')
ax.set_title('Delay vs Competition Level (Last 500ep)', fontweight='bold')
ax.grid(True, alpha=0.3)
corr = np.corrcoef(n_granted[-500:], dep_delays[-500:])[0, 1]
ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=10,
        va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# (c) Completion rate / efficiency over time
ax = axes[1, 0]
# Compute running ontime rate (smooth over 200 episodes)
ontime_smooth = smooth(ontimes, 200)
ax.plot(ontime_smooth, color='#ff7f00', linewidth=1.0, alpha=0.95)
ax.set_xlabel('Episode')
ax.set_ylabel('On-Time Rate')
ax.set_title('Per-Step On-Time Rate (smoothed)', fontweight='bold')
ax.grid(True, alpha=0.3)

# (d) Step efficiency
ax = axes[1, 1]
ax.plot(smooth(steps_list, window), color='#4daf4a', linewidth=1.0, alpha=0.95)
ax.set_xlabel('Episode')
ax.set_ylabel('Steps per Episode')
ax.set_title('Scheduling Efficiency', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)
# Heuristic baseline
ax.axhline(y=8.3, color='gray', linewidth=1, linestyle='--', alpha=0.7)
ax.text(len(steps_list) * 0.6, 8.5, 'Heuristic: 8.3 steps', fontsize=9, color='gray')

fig.suptitle(f'Multi-Agent Coordination Analysis — 8000 Episodes\n'
             f'9 trains, A/B/E=3p C=4p D=2p, visitors@B/C/D, bws=120',
             fontweight='bold')
plt.tight_layout()
fig.savefig(PIC_DIR / 'fig4_learning_coordination.png', dpi=150, bbox_inches='tight')
print(f'Saved: fig4_learning_coordination.png')
plt.close('all')

# ============================================================
# Plus: fig1 MAPPO vs Heuristic (simple bar)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Delay
labels = ['Heuristic', 'MAPPO']
delays = [heuristic_per_step_delay, mappo_per_step_delay]
colors_bar = ['#999999', '#377eb8']
bars = ax1.bar(labels, delays, color=colors_bar, width=0.45, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, delays):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}s', ha='center', fontsize=16, fontweight='bold')
ax1.set_ylabel('Mean Departure Delay (s)')
ax1.set_title(f'Average Delay', fontweight='bold')
ax1.set_ylim(0, max(delays) * 1.3)

# Steps/completion
heuristic_steps = 8.3
mappo_steps = np.mean(steps_list[-500:])
steps_vals = [heuristic_steps, mappo_steps]
bars2 = ax2.bar(labels, steps_vals, color=colors_bar, width=0.45, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars2, steps_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{val:.1f}', ha='center', fontsize=16, fontweight='bold')
ax2.set_ylabel('Steps per Episode')
ax2.set_title(f'Completion Efficiency', fontweight='bold')

fig.suptitle(f'MAPPO vs Heuristic — Five-Station Railway Scheduling\n'
             f'10 trains, A/B/C/E=4p D=2p, visitors@A/B/C/E',
             fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(PIC_DIR / 'fig1_mappo_vs_heuristic.png', dpi=150, bbox_inches='tight')
print(f'Saved: fig1_mappo_vs_heuristic.png')
plt.close('all')

print('\nAll figures updated.')
print(f'  fig1: MAPPO {mappo_per_step_delay:.1f}s vs Heuristic {heuristic_per_step_delay:.1f}s')
print(f'  fig2: Training convergence (reward/delay/steps/n_granted)')
print(f'  fig3: Delay comparison + distribution + reward-delay scatter')
print(f'  fig4: Coordination analysis (n_granted/correlation/ontime/efficiency)')
