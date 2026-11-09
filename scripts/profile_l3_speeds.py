#!/usr/bin/env python3
"""Profile L3 Worker speed decisions — clean version."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from collections import defaultdict
from src.experiment.scenario_loader import load_scenario
from src.env.five_station_env import FiveStationEnv
from src.env.observation_builder import get_obs_dim
from src.marl.actor_critic import Layer2ActorCritic
from src.marl.l3_worker import CorridorActor, CorridorCritic, L3WorkerTrainer
from src.control.l3_sim import SpeedMode, load_l3_physics, L3Simulator
from src.control.action_mask import heuristic_action

cfg = load_scenario(scenario_name='five_station')
l3_physics = load_l3_physics(Path('configs/five_station'))
obs_dim = get_obs_dim(cfg)

# Load checkpoint
ckpt = torch.load(
    'output/8_16/training/five_station_15t_hierarchical/20260816_140621/checkpoints/best.pt',
    map_location='cpu', weights_only=False)
print(f'Checkpoint: ep {ckpt["episode"]}')

# Build models
ac = Layer2ActorCritic(obs_dim=obs_dim, k_local=1, hidden_dim=128)
ac.load_state_dict(ckpt['ac_state'])
ac.eval()
l3_actor = CorridorActor(hidden_dim=128)
l3_critic = CorridorCritic(hidden_dim=128)
l3_critic_g = CorridorCritic(hidden_dim=128)
l3_actor.load_state_dict(ckpt['l3_actor'])
l3_critic.load_state_dict(ckpt['l3_critic_local'])
l3_critic_g.load_state_dict(ckpt['l3_critic_global'])
l3_actor.eval()
l3_critic.eval()
l3_critic_g.eval()
l3_trainer = L3WorkerTrainer(l3_actor, l3_critic, l3_critic_g)

speed_names = {0: 'CRUISE', 1: 'FAST', 2: 'SLOW'}
transit_records = []

# Patch BEFORE creating any envs
_real_transit = L3Simulator.transit

def _collecting_transit(self, train_id, from_station, to_station, corridor_id,
                        planned_total_sec, actual_departure, speed_choice=SpeedMode.CRUISE,
                        train_class='G'):
    result = _real_transit(self, train_id, from_station, to_station, corridor_id,
                           planned_total_sec, actual_departure, speed_choice, train_class)
    transit_records.append(dict(
        train_id=train_id, corridor=corridor_id, train_class=train_class,
        speed=speed_names.get(int(speed_choice), '?'),
        departure=actual_departure, planned_transit=planned_total_sec,
        actual_transit=result.actual_transit_sec, distance_m=result.distance_m,
        headway_delay=result.headway_delay_sec, recovery_used=result.recovery_used_sec,
        arrival=result.actual_arrival))
    return result

L3Simulator.transit = _collecting_transit

# Run episodes
N_EP = 10
for seed in range(N_EP):
    env = FiveStationEnv(cfg, seed=seed, batch_window_sec=120.0,
                         enable_vt=True, enable_initial_delay=True,
                         l3_worker=l3_trainer)
    obs, info = env.reset()
    done = False
    while not done:
        masks = info.get('action_masks', {})
        action_dict = {}
        for tid in info.get('pending', []):
            train = cfg.trains[tid]
            st = env._journeys[tid].current_station
            bounds = env._get_effective_bounds(tid, st, env._t_now)
            try:
                plat, entry, _ = heuristic_action(
                    train, st, env._t_now, 0.0, cfg, bounds_override=bounds)
                platforms = train.candidate_platforms.get(st, [])
                p_idx = platforms.index(plat) if plat in platforms else 0
                e_idx = ['earliest','plan','plan+1','plan+2','plan+3','plan+4'].index(entry)
                action_dict[tid] = (p_idx, e_idx)
            except ValueError:
                pass
        obs, reward, terminated, truncated, info = env.step(action_dict)
        done = terminated or truncated

L3Simulator.transit = _real_transit  # restore

# === Analysis ===
print(f'\n=== L3 Speed Profile ({N_EP} episodes, {len(transit_records)} transits) ===\n')

# Speed distribution
speed_counts = defaultdict(int)
for r in transit_records:
    speed_counts[r['speed']] += 1
print('Speed distribution:')
for s in ['CRUISE', 'FAST', 'SLOW']:
    c = speed_counts[s]
    print(f'  {s}: {c} ({c/len(transit_records)*100:.0f}%)')

# By corridor (sorted by count)
print('\nBy corridor:')
by_corridor = defaultdict(list)
for r in transit_records:
    by_corridor[r['corridor']].append(r)
for cid in sorted(by_corridor, key=lambda c: -len(by_corridor[c])):
    recs = by_corridor[cid]
    sp = defaultdict(int)
    for r in recs: sp[r['speed']] += 1
    avg_hw = sum(r['headway_delay'] for r in recs) / len(recs)
    avg_rec = sum(r['recovery_used'] for r in recs) / len(recs)
    sp_str = ', '.join(f'{s}:{c}' for s, c in sorted(sp.items()))
    print(f'  {cid}: n={len(recs):>3}  avg_hw={avg_hw:>6.1f}s  avg_rec={avg_rec:>5.1f}s  [{sp_str}]')

# By train class
print('\nBy train class:')
by_cls = defaultdict(list)
for r in transit_records:
    by_cls[r['train_class']].append(r)
for cls in sorted(by_cls):
    recs = by_cls[cls]
    sp = defaultdict(int)
    for r in recs: sp[r['speed']] += 1
    avg_hw = sum(r['headway_delay'] for r in recs) / max(len(recs), 1)
    sp_str = ', '.join(f'{s}:{c}' for s, c in sorted(sp.items()))
    print(f'  {cls}: n={len(recs)}  avg_hw={avg_hw:.1f}s  [{sp_str}]')

# Headway distribution
print('\nHeadway delay distribution:')
hw_bins = [(0, 0.1, '0s'), (0.1, 5, '<5s'), (5, 15, '5-15s'), (15, 30, '15-30s'), (30, 999, '>30s')]
for lo, hi, label in hw_bins:
    c = sum(1 for r in transit_records if lo <= r['headway_delay'] < hi)
    print(f'  {label}: {c} ({c/len(transit_records)*100:.0f}%)')

# Speed vs scenario
print('\nSpeed vs headway context:')
for s in ['FAST', 'SLOW', 'CRUISE']:
    recs = [r for r in transit_records if r['speed'] == s]
    if recs:
        hw = sum(r['headway_delay'] for r in recs) / len(recs)
        rec = sum(r['recovery_used'] for r in recs) / len(recs)
        print(f'  {s} (n={len(recs)}): avg_hw_dly={hw:.1f}s  avg_rec_used={rec:.1f}s')

# Sample log (one per corridor per speed)
print('\n=== Sample transits ===')
print(f'{"train":<8} {"corridor":<10} {"cls":<3} {"speed":<7} {"dep":>6} {"plan":>5} {"actual":>6} {"hw":>5} {"rec":>5}')
seen = set()
for r in transit_records:
    key = (r['corridor'], r['speed'])
    if key not in seen:
        seen.add(key)
        print(f'{r["train_id"]:<8} {r["corridor"]:<10} {r["train_class"]:<3} {r["speed"]:<7} '
              f'{r["departure"]:>6.0f} {r["planned_transit"]:>5.0f} {r["actual_transit"]:>6.1f} '
              f'{r["headway_delay"]:>5.1f} {r["recovery_used"]:>5.1f}')
