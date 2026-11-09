#!/usr/bin/env python3
"""Train running diagram (列车运行图) — proper Chinese railway convention.

Format: 十分格 (10-min grid)
  - X-axis: simulation time (s)
  - Y-axis: stations ordered by real distance, A at top, E at bottom
  - Up trains (上行 A→E, even): slope ↘  (upper-left → lower-right)
  - Down trains (下行 E→A, odd):  slope ↗  (lower-left → upper-right)
  - Steeper slope = faster train (G > K)
  - Station spacing proportional to real distance
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import numpy as np
from collections import defaultdict

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

from src.experiment.scenario_loader import load_scenario
from src.env.five_station_env import FiveStationEnv
from src.env.observation_builder import get_obs_dim, critic_input
from src.marl.actor_critic import Layer2ActorCritic, compact_global_state
from src.marl.l3_worker import CorridorActor, CorridorCritic, L3WorkerTrainer
from src.control.l3_sim import L3Simulator, SpeedMode, load_l3_physics
from src.control.resource_manager import build_conflict_graph

# ---------------------------------------------------------------------------
# Config & checkpoint
# ---------------------------------------------------------------------------
cfg = load_scenario(scenario_name='five_station')
l3_physics = load_l3_physics(Path('configs/five_station'))
obs_dim = get_obs_dim(cfg)

ckpt = torch.load(
    'output/8_16/training/five_station_15t_hierarchical/20260816_140621/checkpoints/best.pt',
    map_location='cpu', weights_only=False)

ac = Layer2ActorCritic(obs_dim=obs_dim, k_local=1, hidden_dim=128)
ac.load_state_dict(ckpt['ac_state']); ac.eval()
l3_actor = CorridorActor(hidden_dim=128); l3_critic = CorridorCritic(hidden_dim=128)
l3_critic_g = CorridorCritic(hidden_dim=128)
l3_actor.load_state_dict(ckpt['l3_actor'])
l3_critic.load_state_dict(ckpt['l3_critic_local'])
l3_critic_g.load_state_dict(ckpt['l3_critic_global'])
l3_actor.eval(); l3_critic.eval(); l3_critic_g.eval()
l3_trainer = L3WorkerTrainer(l3_actor, l3_critic, l3_critic_g)

# ---------------------------------------------------------------------------
# Collect movement data
# ---------------------------------------------------------------------------
spd_names = {0: 'CRUISE', 1: 'FAST', 2: 'SLOW'}
corridor_events = []  # {tid, fr, to, dep, arr, spd, hw, rec, cls}
station_waits = {}     # {tid: [(st, arr_time, dep_time), ...]}

_real = L3Simulator.transit
def _record(self, tid, fr, to, cid, planned_total_sec, actual_departure,
            speed_choice=SpeedMode.CRUISE, train_class='G'):
    r = _real(self, tid, fr, to, cid, planned_total_sec, actual_departure,
              speed_choice, train_class)
    corridor_events.append(dict(tid=tid, fr=fr, to=to, dep=actual_departure,
        arr=r.actual_arrival, spd=spd_names.get(int(speed_choice),'?'),
        hw=r.headway_delay_sec, rec=r.recovery_used_sec, cls=train_class))
    return r
L3Simulator.transit = _record

env = FiveStationEnv(cfg, seed=42, batch_window_sec=120.0,
                     enable_vt=True, enable_initial_delay=True,
                     l3_worker=l3_trainer)
obs, info = env.reset()

# Record origin waiting
for tid, j in env._journeys.items():
    train = cfg.trains[tid]
    origin = train.origin_station
    if origin and not j.is_done:
        station_waits.setdefault(tid, []).append(
            (origin, j.arrival_at_current, None))  # departure filled later

done = False
while not done:
    masks = info.get('action_masks', {})
    requests = info.get('requests', [])
    conflict_graph = build_conflict_graph(requests)
    group_obs = {aid: critic_input(aid, obs, conflict_graph, k_max=1) for aid in obs}
    gs_vec = compact_global_state(info.get('metrics', {}))

    agent_actions = {}
    for aid in obs:
        m = masks.get(aid)
        if m and any(m[0]) and any(m[1]):
            obs_t = torch.tensor(obs[aid], dtype=torch.float32).unsqueeze(0)
            pm = torch.tensor(list(m[0])+[False]*(4-len(m[0])), dtype=torch.bool)
            em = torch.tensor(list(m[1]), dtype=torch.bool)
            local_t = torch.tensor(group_obs.get(aid, obs[aid]), dtype=torch.float32).unsqueeze(0)
            global_t = torch.tensor(gs_vec, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out = ac.act(obs_t, pm, em, local_t, global_t)
            agent_actions[aid] = (out.platform.item(), out.entry.item())

    action_dict = {aid: (a[0], a[1]) for aid, a in agent_actions.items()}
    obs, reward, terminated, truncated, info = env.step(action_dict, lambda_team=0.3)
    done = terminated or truncated

L3Simulator.transit = _real

# ---------------------------------------------------------------------------
# Build train trajectories from corridor events
# ---------------------------------------------------------------------------
# For each train: [S0 dep@T0, S1 arr@T1 (dwell), S1 dep@T2, S2 arr@T3, ...]
# Corridor events give: dep at S_N, arr at S_{N+1}
# Station dwell: time between arr at S_N and next dep from S_N

trajectories = {}  # {tid: [(st, time, 'arr'|'dep'), ...]}

for tid in set(e['tid'] for e in corridor_events):
    train = cfg.trains[tid]
    # Get all corridor events for this train, sorted by departure time
    evts = sorted([e for e in corridor_events if e['tid'] == tid], key=lambda x: x['dep'])

    traj = []

    # Origin station
    if evts:
        origin_st = evts[0]['fr']
        # Use planned departure as origin "arrival" (train available from this time).
        # journey.arrival_at_current is overwritten during the episode.
        planned_dep = cfg.trains[tid].planned[origin_st].dep
        planned_arr = cfg.trains[tid].planned[origin_st].arr
        origin_arr = planned_arr if planned_arr > 0 else 0
        traj.append((origin_st, origin_arr, 'arr'))
        traj.append((origin_st, evts[0]['dep'], 'dep'))

    # Intermediate stations: arrival from previous corridor, departure to next corridor
    for i, e in enumerate(evts):
        nxt_st = e['to']
        # Arrival at next station
        traj.append((nxt_st, e['arr'], 'arr'))
        # If there's a NEXT corridor event from this station, that's the departure
        if i + 1 < len(evts) and evts[i+1]['fr'] == nxt_st:
            traj.append((nxt_st, evts[i+1]['dep'], 'dep'))

    traj.sort(key=lambda x: x[1])
    trajectories[tid] = traj

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
STATIONS = ['A', 'B', 'C', 'D', 'E']
STATION_NAMES = {'A': '南昌西', 'B': '鹰潭北', 'C': '上饶', 'D': '杭州东', 'E': '南京南'}

# Station Y positions: proportional to real distance (cumulative from A)
REAL_DIST_KM = {'A_to_B': 140, 'B_to_C': 101, 'C_to_D': 341, 'D_to_E': 256}
cum_km = 0
STATION_KM = {'A': 0}
for i in range(len(STATIONS)-1):
    key = f'{STATIONS[i]}_to_{STATIONS[i+1]}'
    cum_km += REAL_DIST_KM.get(key, 100)
    STATION_KM[STATIONS[i+1]] = cum_km

# Invert Y: A (km 0) at top, E (max km) at bottom
max_km = STATION_KM['E']
STATION_Y = {s: STATION_KM[s] for s in STATIONS}

# Colors & styles
# Per-train distinct colors (high contrast, colorblind-friendly)
TRAIN_COLORS = {
    'G1380': '#e6194b',  # red
    'G230':  '#3cb44b',  # green
    'G1616': '#cc8800',  # dark gold
    'G238':  '#4363d8',  # blue
    'G5038': '#f58231',  # orange
    'G7629': '#911eb4',  # purple
    'G2245': '#008080',  # teal
    'G1487': '#e04080',  # hot pink
    'G1347': '#88aa00',  # olive
    'K1185': '#000000',  # black
    'K834':  '#666666',  # dark gray
}

def train_style(tid, cls, direction, prio):
    color = TRAIN_COLORS.get(tid, '#999999')
    lw = 2.8 if prio == 1 else 2.0
    ls = '--' if cls == 'K' else '-'
    return {'color': color, 'lw': lw, 'ls': ls, 'alpha': 0.9}

fig, ax = plt.subplots(figsize=(18, 12))

# Grid & station lines
for s in STATIONS:
    y = STATION_Y[s]
    ax.axhline(y=y, color='#555555', linewidth=1.0, alpha=0.6)

# Station labels on y-axis (name + ID, no duplicates)
ax.set_yticks([STATION_Y[s] for s in STATIONS])
ax.set_yticklabels([f'{STATION_NAMES[s]} ({s})' for s in STATIONS], fontsize=10)

# Time grid (every 60s = 1 min sim)
for t in range(0, 1500, 60):
    ax.axvline(x=t, color='#cccccc', linewidth=0.3, alpha=0.5)
for t in range(0, 1500, 300):
    ax.axvline(x=t, color='#999999', linewidth=0.6, alpha=0.7)

# Plot each train
for tid, traj in trajectories.items():
    train = cfg.trains[tid]
    style = train_style(tid, train.train_class, train.direction, train.priority)

    times = [t for _, t, _ in traj]
    stations = [st for st, _, _ in traj]
    ys = [STATION_Y[st] for st in stations]

    # Split into segments: dwell (same station) vs transit (different stations)
    i = 0
    while i < len(traj) - 1:
        st1, t1, evt1 = traj[i]
        st2, t2, evt2 = traj[i+1]

        if st1 == st2:
            # Dwell: horizontal line (arr → dep at same station)
            ax.plot([t1, t2], [STATION_Y[st1], STATION_Y[st1]],
                    color=style['color'], linewidth=style['lw']+1,
                    linestyle='-', alpha=0.4, solid_capstyle='butt')
        else:
            # Corridor transit: diagonal line
            ax.plot([t1, t2], [STATION_Y[st1], STATION_Y[st2]],
                    color=style['color'], linewidth=style['lw'],
                    linestyle=style['ls'], alpha=style['alpha'])
        i += 1

    # Label at origin (left side) — offset by train index to avoid overlap
    first_st = traj[0][0]
    first_t = traj[0][1]
    y_offset = (hash(tid) % 7 - 3) * 9  # spread labels vertically
    ax.annotate(tid, (first_t, STATION_Y[first_st]),
               textcoords="offset points", xytext=(-28, y_offset + 2),
               fontsize=7.5, color=style['color'], fontweight='bold', ha='left',
               va='center')

    # Label at destination (right side)
    last_st = traj[-1][0]
    last_t = traj[-1][1]
    ax.annotate(tid, (last_t, STATION_Y[last_st]),
               textcoords="offset points", xytext=(8, y_offset),
               fontsize=7.5, color=style['color'], fontweight='bold', ha='left',
               va='center')

# Mark D station as bottleneck
d_y = STATION_Y['D']
ax.axhspan(d_y-15, d_y+15, color='red', alpha=0.04)

# Axes
ax.set_xlabel('Simulation time (s)', fontsize=12)
ax.set_title('Train Running Diagram  列车运行图\nMAPPO L2 + L3 Worker — Phase C (ep 6229) — VT + bunched delay',
             fontsize=14, fontweight='bold')
ax.set_xlim(-10, 960)

# Legend — per train
legend_elements = []
for tid in sorted(TRAIN_COLORS.keys()):
    train = cfg.trains.get(tid)
    if train:
        cls_label = f'{tid} ({train.train_class}, P{train.priority}, {"↑" if train.direction=="up" else "↓"})'
        legend_elements.append(mpatches.Patch(color=TRAIN_COLORS[tid], label=cls_label))
ax.legend(handles=legend_elements, loc='upper right', fontsize=7.5, framealpha=0.9, ncol=2)

plt.tight_layout()
outdir = Path('output/figures')
outdir.mkdir(parents=True, exist_ok=True)
outpath = outdir / 'train_diagram.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f'Saved: {outpath}')

# Also save data
csv_path = outdir / 'train_diagram_data.csv'
with open(csv_path, 'w') as f:
    f.write('train_id,station,time_s,event,class,direction,priority\n')
    for tid, traj in trajectories.items():
        train = cfg.trains[tid]
        for st, t, evt in traj:
            f.write(f'{tid},{st},{t:.1f},{evt},{train.train_class},{train.direction},{train.priority}\n')

# Summary
print(f'Trains plotted: {len(trajectories)}')
for tid in sorted(trajectories.keys()):
    traj = trajectories[tid]
    parts = []
    for st, t, evt in traj:
        if evt == 'dep':
            parts.append(f'{st} dep={t:.0f}s')
        else:
            parts.append(f'{st} arr={t:.0f}s')
    arrow = ' -> '
    print(f'  {tid}: {arrow.join(parts)}')
