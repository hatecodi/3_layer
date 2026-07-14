#!/usr/bin/env python3
"""Robustness test: evaluate clean-trained L2+L3 under VT+delay disturbance."""
import sys, torch; from pathlib import Path; from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env.five_station_env import FiveStationEnv
from src.env.observation_builder import get_obs_dim, critic_input
from src.marl.actor_critic import Layer2ActorCritic, compact_global_state
from src.marl.mappo import MAPPOTrainer
from src.marl.l3_worker import CorridorActor, CorridorCritic, L3WorkerTrainer, L3_OBS_DIM
from src.experiment.scenario_loader import load_scenario
from src.control.resource_manager import build_conflict_graph

HP = {"obs_dim": 29, "action_dim": 45, "k_local": 1, "hidden_dim": 128, "l3_hidden_dim": 64,
      "lr": 3e-4, "l3_lr": 1e-3, "gamma": 0.99, "gae_lambda": 0.95, "clip_ratio": 0.2,
      "ppo_epochs": 5, "minibatch_size": 256, "entropy_coef_start": 0.01,
      "entropy_coef_end": 0.005, "l3_entropy_coef": 0.01, "value_coef": 0.5,
      "max_grad_norm": 1.0, "rollout_steps": 512}

def _bc(obs, reqs):
    g = build_conflict_graph(reqs)
    return {aid: critic_input(aid, obs, g, k_max=1) for aid in obs}

cfg = load_scenario(scenario_name='five_station')
obs_dim = get_obs_dim(cfg)
device = torch.device('cpu')

# L2
l2_ckpt = torch.load('output/five_station_hierarchical/20260714_185043/checkpoints/best.pt', map_location='cpu', weights_only=False)
ac = Layer2ActorCritic(obs_dim=obs_dim, k_local=1, hidden_dim=128)
ac.load_state_dict(l2_ckpt['ac_state']); ac.eval()
for p in ac.parameters(): p.requires_grad = False
l2 = MAPPOTrainer(ac, lr=1e-3, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2,
                  ppo_epochs=5, minibatch_size=256, entropy_coef=0.01,
                  value_coef=0.5, max_grad_norm=1.0, total_episodes=1, device=device)

# L3
l3_ckpt = torch.load('output/five_station_l3_only/20260714_202432/checkpoints/best.pt', map_location='cpu', weights_only=False)
l3_actor = CorridorActor(obs_dim=L3_OBS_DIM, hidden_dim=64)
l3_critic = CorridorCritic(obs_dim=L3_OBS_DIM, hidden_dim=64)
l3_actor.load_state_dict(l3_ckpt['l3_actor']); l3_actor.eval()
l3_critic.load_state_dict(l3_ckpt['l3_critic']); l3_critic.eval()
for p in list(l3_actor.parameters()) + list(l3_critic.parameters()):
    p.requires_grad = False
l3 = L3WorkerTrainer(l3_actor, l3_critic, lr=1e-3, gamma=0.99, gae_lambda=0.95,
                     clip_ratio=0.2, ppo_epochs=5, entropy_coef=0.01,
                     value_coef=0.5, max_grad_norm=1.0, device=device)

print("Testing robustness: clean L2+L3 under VT+delay")
print(f"{'ep':>4} {'dly':>6} {'ont':>5} {'CR%':>5} {'FA%':>5} {'held':>5} {'viol'}")

results = []
for ep in range(50):
    l2.advance_episode()
    env = FiveStationEnv(cfg=cfg, seed=ep+99000, batch_window_sec=120.0,
                         enable_vt=True, enable_initial_delay=True, l3_worker=l3)
    obs, info = env.reset()
    ep_done = False
    ep_dly, ep_ont, ep_held, ep_granted = 0.0, 0, 0, 0
    l3_c, l3_f, l3_t = 0, 0, 0
    all_events = []

    while not ep_done:
        masks = info.get('action_masks', {})
        reqs = info.get('requests', [])
        group_obs = _bc(obs, reqs)
        gs_vec = compact_global_state(info.get('metrics', {}))
        acts = l2.collect_step(obs, masks, group_obs, gs_vec, device=device)
        action_dict = {aid: (a.platform, a.entry) for aid, a in acts.items()}

        for g in info.get('granted_windows', []):
            all_events.append({"station": env._journeys[g.train_id].current_station,
                               "platform": g.platform, "train": g.train_id,
                               "t_arr": g.W_plat[0], "t_dep": g.W_plat[1]})

        next_obs, _, terminated, truncated, info = env.step(action_dict, lambda_team=l2.lam_team)
        ep_done = terminated or truncated
        metrics = info.get('metrics', {})
        ep_granted += metrics.get('n_granted', 0)
        ep_ont += metrics.get('n_ontime_dep', 0)
        ep_dly += metrics.get('total_dep_delay', 0.0)
        ep_held += metrics.get('n_held', 0)
        l3c = info.get('l3_speed_choices', {})
        for _, s in l3c.items():
            l3_t += 1
            if s == 0: l3_c += 1
            elif s == 1: l3_f += 1
        obs = next_obs

    # Check violations
    bookings = defaultdict(list)
    for e in all_events:
        bookings[(e["station"], e["platform"])].append(e)
    viols = 0
    for evts in bookings.values():
        evts.sort(key=lambda e: e["t_arr"])
        for i in range(len(evts)):
            for j in range(i+1, len(evts)):
                a, b = evts[i], evts[j]
                if a["train"] == b["train"]: continue
                if b["t_arr"] - a["t_dep"] < 5.0: viols += 1

    mean_dly = ep_dly / max(ep_granted, 1)
    ont = ep_ont / max(ep_granted, 1)
    cr_pct = 100 * l3_c / max(l3_t, 1)
    fa_pct = 100 * l3_f / max(l3_t, 1)
    results.append((mean_dly, ont, cr_pct, fa_pct, ep_held, viols))
    if (ep+1) % 10 == 0 or ep < 5:
        print(f'{ep:>4} {mean_dly:>5.1f}s {ont:>4.2f} {cr_pct:>4.0f}% {fa_pct:>4.0f}% {ep_held:>5} {viols:>4}')

# Summary
avg_dly = sum(r[0] for r in results) / len(results)
avg_ont = sum(r[1] for r in results) / len(results)
avg_cr = sum(r[2] for r in results) / len(results)
avg_fa = sum(r[3] for r in results) / len(results)
avg_held = sum(r[4] for r in results) / len(results)
total_viols = sum(r[5] for r in results)
print(f'\n{"="*50}')
print(f'Average across 50 episodes:')
print(f'  delay={avg_dly:.1f}s  ontime={avg_ont:.2f}  held={avg_held:.1f}')
print(f'  L3: cruise={avg_cr:.0f}%  fast={avg_fa:.0f}%')
print(f'  Platform violations: {total_viols}')
