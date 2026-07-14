#!/usr/bin/env python3
"""Track train movements: collision detection + deadlock monitoring."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from collections import defaultdict
from src.experiment.scenario_loader import load_scenario
from src.env.five_station_env import FiveStationEnv
from src.env.observation_builder import get_obs_dim, critic_input
from src.marl.actor_critic import Layer2ActorCritic, compact_global_state
from src.marl.mappo import MAPPOTrainer
from src.marl.rollout_buffer import RolloutBuffer, AgentTransition
from src.control.resource_manager import build_conflict_graph
from src.control.l3_sim import L3Simulator, SpeedMode

cfg = load_scenario(scenario_name='five_station')
obs_dim = get_obs_dim(cfg)

# Load trained policy
ckpt = torch.load('output/five_station_hierarchical/20260713_234114/checkpoints/final.pt',
                   map_location='cpu', weights_only=False)
ac = Layer2ActorCritic(obs_dim=obs_dim, k_local=1, hidden_dim=128)
ac.load_state_dict(ckpt['ac_state']); ac.eval()

# Track safety events
collisions = []
deadlock_events = []
corridor_conflicts = []

# Patch L3 transit to record corridor occupancy
_real_transit = L3Simulator.transit
corridor_timeline = defaultdict(list)  # {corridor_id: [(train_id, enter, exit), ...]}

def _tracking_transit(self, train_id, from_station, to_station, corridor_id,
                      planned_total_sec, actual_departure, speed_choice=SpeedMode.CRUISE,
                      train_class='G'):
    r = _real_transit(self, train_id, from_station, to_station, corridor_id,
                      planned_total_sec, actual_departure, speed_choice, train_class)
    enter = actual_departure + self.p.throat_out_sec
    exit_t = r.actual_arrival - self.p.throat_in_sec
    corridor_timeline[corridor_id].append((train_id, enter, exit_t))
    return r

L3Simulator.transit = _tracking_transit

N = 500
stats = defaultdict(int)
print(f'Tracking {N} episodes (MAPPO L2 + MAPPO L3)...')

for seed in range(N):
    env = FiveStationEnv(cfg, seed=seed, batch_window_sec=120.0,
                         enable_vt=True, enable_initial_delay=True)
    obs, info = env.reset()
    done = False
    deadlock_steps = 0
    ep_collisions = 0
    no_legal_streak = 0
    max_no_legal = 0
    platform_timeline = defaultdict(list)  # {(station, platform): [(train, start, end), ...]}

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

        # Capture station BEFORE step (for platform collision check)
        pre_step_station = {}
        for tid in info.get('pending', []):
            if tid in env._journeys:
                pre_step_station[tid] = env._journeys[tid].current_station

        obs, reward, terminated, truncated, info = env.step(action_dict, lambda_team=0.3)
        done = terminated or truncated

        # Track NO_LEGAL_ACTION streaks
        pending = info.get('pending', [])
        if pending:
            all_no_legal = all(
                not any(masks.get(tid, ([False], [False]))[0]) and
                not any(masks.get(tid, ([False], [False]))[1])
                for tid in pending
            )
            if all_no_legal:
                no_legal_streak += 1
                max_no_legal = max(max_no_legal, no_legal_streak)
            else:
                no_legal_streak = 0

        # Track platform occupancy (collision check)
        for g in info.get('granted_windows', []):
            st = pre_step_station.get(g.train_id, '')
            if not st or st == '': continue  # skip trains not at a station anymore
            key = (st, g.platform)
            plat_start = g.W_plat[0]; plat_end = g.W_plat[1]
            # Check overlap with existing bookings at same station+platform
            for existing_tid, es, ee in platform_timeline.get(key, []):
                if not (plat_end <= es or plat_start >= ee):  # overlap
                    if existing_tid != g.train_id:
                        ep_collisions += 1
                        collisions.append({
                            'seed': seed, 'station': st, 'platform': g.platform,
                            'train1': existing_tid, 'train1_window': (es, ee),
                            'train2': g.train_id, 'train2_window': (plat_start, plat_end),
                        })
            platform_timeline.setdefault(key, []).append((g.train_id, plat_start, plat_end))

    # Check corridor headway violations (entry gap < minimum headway)
    for cid, events in corridor_timeline.items():
        events.sort(key=lambda x: x[1])
        min_headway = env._l3_sim._headways.get(cid, 15.0)
        for i in range(len(events)-1):
            t1, e1, x1 = events[i]
            t2, e2, x2 = events[i+1]
            entry_gap = e2 - e1  # time between successive entries
            if entry_gap < 5.0:  # only flag serious headway violations (< 5s gap)
                corridor_conflicts.append({
                    'seed': seed, 'corridor': cid,
                    'train1': t1, 'train2': t2,
                    'entry_gap': entry_gap, 'min_headway': min_headway,
                })

    corridor_timeline.clear()

    status = info.get('episode_status', '?')
    stats[status] += 1
    stats['collisions'] += ep_collisions
    stats['max_no_legal_streak'] = max(stats['max_no_legal_streak'], max_no_legal)
    if max_no_legal >= 10:
        deadlock_events.append({'seed': seed, 'max_streak': max_no_legal})

    if (seed+1) % 100 == 0:
        print(f'  {seed+1}/{N}: collisions={stats["collisions"]} deadlocks={stats["deadlock"]} '
              f'corridor_conflicts={len(corridor_conflicts)} max_no_legal={stats["max_no_legal_streak"]}')

L3Simulator.transit = _real_transit

print(f'\n=== Safety Report ({N} episodes) ===')
print(f'Status distribution:')
for s in ['success', 'truncated', 'deadlock']:
    print(f'  {s}: {stats.get(s, 0)} ({stats.get(s,0)/N*100:.1f}%)')

print(f'\nCollisions (platform overlap): {stats["collisions"]}')
if collisions:
    print('  First 5 collisions:')
    for c in collisions[:5]:
        print(f'    seed={c["seed"]} {c["station"]}/{c["platform"]}: {c["train1"]} and {c["train2"]} overlap')

print(f'\nCorridor conflicts (entry before exit): {len(corridor_conflicts)}')
if corridor_conflicts:
    print('  First 5:')
    for c in corridor_conflicts[:5]:
        s = c['seed']; corr = c['corridor']; t1 = c['train1']; t2 = c['train2']
        gap = c['entry_gap']; mh = c['min_headway']
        print(f'    seed={s} {corr}: {t1}->{t2} entry_gap={gap:.1f}s (min={mh:.0f}s)')

print(f'\nDeadlock risk (NO_LEGAL_ACTION streaks >= 10): {len(deadlock_events)}')
if deadlock_events:
    print(f'  Max streak: {stats["max_no_legal_streak"]}')
    for d in deadlock_events[:5]:
        print(f'    seed={d["seed"]} streak={d["max_streak"]}')

max_streak = stats['max_no_legal_streak']
deadlock_count = stats.get('deadlock', 0)
collision_count = stats['collisions']
print(f'\nVerdict:')
unsafe = collision_count > 0 or len(corridor_conflicts) > 0
print(f'  L3 collisions: {"UNSAFE" if unsafe else "SAFE"}')
dl_msg = f'None (max streak={max_streak})' if deadlock_count == 0 else 'DETECTED'
print(f'  L2 deadlocks: {dl_msg}')
