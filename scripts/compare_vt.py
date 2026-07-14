#!/usr/bin/env python3
"""Compare MAPPO vs heuristic under different VT intensities."""
import sys, torch; from pathlib import Path; from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env.five_station_env import FiveStationEnv
from src.env.observation_builder import get_obs_dim, critic_input
from src.marl.actor_critic import Layer2ActorCritic, compact_global_state
from src.marl.mappo import MAPPOTrainer
from src.experiment.scenario_loader import load_scenario, ScenarioConfig
from src.control.resource_manager import build_conflict_graph, resolve_all
from src.control.dispatch_bounds import DispatchBounds
from src.control.action_mask import action_to_request, heuristic_action, is_legal_action
from src.experiment.heuristic_planner import make_rebuild_fn, schedule_station
from src.env.reward import ONTIME_THRESHOLD_SEC

HP = {"obs_dim": 29, "action_dim": 45, "k_local": 1, "hidden_dim": 128,
      "lr": 3e-4, "gamma": 0.99, "gae_lambda": 0.95, "clip_ratio": 0.2,
      "ppo_epochs": 5, "minibatch_size": 256, "entropy_coef_start": 0.01,
      "entropy_coef_end": 0.005, "value_coef": 0.5, "max_grad_norm": 1.0, "rollout_steps": 512}

def _bc(obs, reqs):
    g = build_conflict_graph(reqs)
    return {aid: critic_input(aid, obs, g, k_max=1) for aid in obs}

def run_mappo_episode(env, l2, l3):
    obs, info = env.reset()
    ep_done = False
    held_total, granted_total, ontime_total, dep_dly_total = 0, 0, 0, 0.0
    all_events = []
    while not ep_done:
        masks = info.get('action_masks', {})
        reqs = info.get('requests', [])
        group_obs = _bc(obs, reqs)
        gs_vec = compact_global_state(info.get('metrics', {}))
        acts = l2.collect_step(obs, masks, group_obs, gs_vec, device='cpu')
        action_dict = {aid: (a.platform, a.entry) for aid, a in acts.items()}
        held_total += len(info.get('held_train_ids', []))
        for g in info.get('granted_windows', []):
            all_events.append({"station": env._journeys[g.train_id].current_station,
                               "platform": g.platform, "train": g.train_id,
                               "t_arr": g.W_plat[0], "t_dep": g.W_plat[1]})
        next_obs, _, terminated, truncated, info = env.step(action_dict, lambda_team=l2.lam_team)
        ep_done = terminated or truncated
        metrics = info.get('metrics', {})
        granted_total += metrics.get('n_granted', 0)
        ontime_total += metrics.get('n_ontime_dep', 0)
        dep_dly_total += metrics.get('total_dep_delay', 0.0)
        obs = next_obs
    # Count violations
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
    return held_total, granted_total, ontime_total, dep_dly_total, viols

def run_heuristic_episode(cfg, vt_prob, seed):
    """Run heuristic with fresh station-by-station scheduling at each step.
    Uses the same env time-advance logic, but scheduling decisions are heuristic."""
    env = FiveStationEnv(cfg=cfg, seed=seed, batch_window_sec=120.0,
                         enable_vt=vt_prob > 0, enable_initial_delay=True, l3_worker=None)
    obs, info = env.reset()
    ep_done = False
    held_total, granted_total, ontime_total, dep_dly_total = 0, 0, 0, 0.0
    all_events = []

    while not ep_done:
        # Override actions: use heuristic for each pending train
        pending = info.get('pending', [])
        pending_ids = info.get('active_train_ids', pending)
        held_ids = info.get('held_train_ids', [])
        held_total += len(held_ids)

        # Actually we can't easily override the policy inside env.step().
        # Instead we call env.step() with actions from heuristic.
        # Build heuristic action dict from pending trains.
        action_dict = {}
        layer1 = DispatchBounds(cfg)
        TRAVEL_TIME = 0.0
        for tid in pending_ids:
            st = env._journeys[tid].current_station
            bounds = env._get_effective_bounds(tid, st, env._t_now)
            train = cfg.trains[tid]
            try:
                action = heuristic_action(train, st, env._t_now, TRAVEL_TIME, cfg, bounds_override=bounds)
            except ValueError:
                continue
            platforms = train.candidate_platforms.get(st, [])
            from src.control.action_mask import ENTRY_CHOICES
            if action[0] in platforms and action[1] in ENTRY_CHOICES:
                action_dict[tid] = (platforms.index(action[0]), ENTRY_CHOICES.index(action[1]))

        # Record grants before step
        for g in info.get('granted_windows', []):
            station = env._journeys[g.train_id].current_station
            all_events.append({"station": station, "platform": g.platform,
                               "train": g.train_id, "t_arr": g.W_plat[0], "t_dep": g.W_plat[1]})

        next_obs, _, terminated, truncated, info = env.step(action_dict, lambda_team=0.3)
        ep_done = terminated or truncated
        metrics = info.get('metrics', {})
        granted_total += metrics.get('n_granted', 0)
        ontime_total += metrics.get('n_ontime_dep', 0)
        dep_dly_total += metrics.get('total_dep_delay', 0.0)
        obs = next_obs

    # Count violations
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
    return held_total, granted_total, ontime_total, dep_dly_total, viols


def main():
    cfg = load_scenario(scenario_name='five_station')
    obs_dim = get_obs_dim(cfg)

    # Load MAPPO L2
    l2_ckpt = torch.load('output/five_station_hierarchical/20260714_185043/checkpoints/best.pt', map_location='cpu', weights_only=False)
    ac = Layer2ActorCritic(obs_dim=obs_dim, k_local=1, hidden_dim=128)
    ac.load_state_dict(l2_ckpt['ac_state']); ac.eval()
    for p in ac.parameters(): p.requires_grad = False
    l2 = MAPPOTrainer(ac, lr=1e-3, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2,
                      ppo_epochs=5, minibatch_size=256, entropy_coef=0.01,
                      value_coef=0.5, max_grad_norm=1.0, total_episodes=1, device='cpu')

    vt_levels = [(0.0, "无VT"), (0.25, "VT-低"), (0.5, "VT-高")]
    n_ep = 50

    print(f"{'Method':<10} {'VT':<8} {'held/ep':>8} {'delay':>8} {'ontime':>8} {'viol/ep':>8}")
    print("-" * 58)

    for vt_prob, label in vt_levels:
        # MAPPO
        mappo_stats = []
        for ep in range(n_ep):
            l2.advance_episode()
            # Monkey-patch VT prob
            old_enabled = cfg.time_generation.get('visitors', {}).get('enabled', True)
            old_prob = cfg.time_generation.get('visitors', {}).get('prob_per_slot', 0.5)
            cfg.time_generation.setdefault('visitors', {})['prob_per_slot'] = vt_prob
            cfg.time_generation['visitors']['enabled'] = vt_prob > 0

            env = FiveStationEnv(cfg=cfg, seed=ep+99000, batch_window_sec=120.0,
                                 enable_vt=vt_prob > 0, enable_initial_delay=True, l3_worker=None)
            h, g, o, d, v = run_mappo_episode(env, l2, None)
            mappo_stats.append((h, g, o, d, v))

            cfg.time_generation['visitors']['enabled'] = old_enabled
            cfg.time_generation['visitors']['prob_per_slot'] = old_prob

        avg_h = sum(s[0] for s in mappo_stats) / n_ep
        avg_g = sum(s[1] for s in mappo_stats) / n_ep
        avg_d = sum(s[3] for s in mappo_stats) / max(sum(s[1] for s in mappo_stats), 1)
        avg_o = sum(s[2] for s in mappo_stats) / max(sum(s[1] for s in mappo_stats), 1)
        avg_v = sum(s[4] for s in mappo_stats) / n_ep
        print(f"{'MAPPO':<10} {label:<8} {avg_h:>8.1f} {avg_d:>7.1f}s {avg_o:>7.2f} {avg_v:>8.1f}")

        # Heuristic
        heur_stats = []
        for ep in range(n_ep):
            cfg.time_generation.setdefault('visitors', {})['prob_per_slot'] = vt_prob
            cfg.time_generation['visitors']['enabled'] = vt_prob > 0
            h, g, o, d, v = run_heuristic_episode(cfg, vt_prob, ep + 99000)
            heur_stats.append((h, g, o, d, v))
            cfg.time_generation['visitors']['enabled'] = False
            cfg.time_generation['visitors']['prob_per_slot'] = 0.5

        avg_h = sum(s[0] for s in heur_stats) / n_ep
        avg_g = sum(s[1] for s in heur_stats) / n_ep
        avg_d = sum(s[3] for s in heur_stats) / max(sum(s[1] for s in heur_stats), 1)
        avg_o = sum(s[2] for s in heur_stats) / max(sum(s[1] for s in heur_stats), 1)
        avg_v = sum(s[4] for s in heur_stats) / n_ep
        print(f"{'Heuristic':<10} {label:<8} {avg_h:>8.1f} {avg_d:>7.1f}s {avg_o:>7.2f} {avg_v:>8.1f}")
        print()


if __name__ == "__main__":
    main()
