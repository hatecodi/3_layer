"""Debug: run trained L2 to verify held count."""
import sys, torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.env.five_station_env import FiveStationEnv
from src.env.observation_builder import get_obs_dim, critic_input
from src.marl.actor_critic import Layer2ActorCritic, compact_global_state
from src.marl.mappo import MAPPOTrainer
from src.experiment.scenario_loader import load_scenario
from src.control.resource_manager import build_conflict_graph, ResourceCalendar
from collections import defaultdict

HP = {"obs_dim": 29, "action_dim": 45, "k_local": 1, "hidden_dim": 128,
      "lr": 3e-4, "gamma": 0.99, "gae_lambda": 0.95, "clip_ratio": 0.2,
      "ppo_epochs": 5, "minibatch_size": 256, "entropy_coef_start": 0.01,
      "entropy_coef_end": 0.005, "value_coef": 0.5, "max_grad_norm": 1.0,
      "rollout_steps": 512}

def _build_critic_inputs(obs, requests):
    conflict_graph = build_conflict_graph(requests)
    group_obs = {}
    for agent_id in obs:
        group_obs[agent_id] = critic_input(agent_id, obs, conflict_graph, k_max=HP["k_local"])
    return group_obs

cfg = load_scenario(scenario_name="five_station")
obs_dim = get_obs_dim(cfg)
device = torch.device("cpu")

# Load trained L2
ckpt_path = "output/five_station_hierarchical/20260713_174053/checkpoints/best.pt"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
ac = Layer2ActorCritic(obs_dim=obs_dim, k_local=HP["k_local"], hidden_dim=HP["hidden_dim"])
ac.load_state_dict(ckpt["ac_state"])
ac.eval()
for p in ac.parameters():
    p.requires_grad = False
l2_trainer = MAPPOTrainer(
    ac, lr=HP["lr"], gamma=HP["gamma"], gae_lambda=HP["gae_lambda"],
    clip_ratio=HP["clip_ratio"], ppo_epochs=HP["ppo_epochs"],
    minibatch_size=HP["minibatch_size"], entropy_coef=HP["entropy_coef_start"],
    value_coef=HP["value_coef"], max_grad_norm=HP["max_grad_norm"],
    total_episodes=1, device=device)

# Check for platform overlaps
def check_overlaps(events):
    violations = []
    bookings = defaultdict(list)
    for e in events:
        bookings[(e["station"], e["platform"])].append(e)
    for (st, plat), evts in bookings.items():
        evts.sort(key=lambda e: e["t_arr"])
        for i in range(len(evts)):
            for j in range(i + 1, len(evts)):
                a, b = evts[i], evts[j]
                if a["train"] == b["train"]:
                    continue
                gap = b["t_arr"] - a["t_dep"]
                if gap < 5.0:
                    violations.append(
                        f"OVERLAP: {st}/{plat} {a['train']}[{a['t_arr']:.0f},{a['t_dep']:.0f}] "
                        f"vs {b['train']}[{b['t_arr']:.0f},{b['t_dep']:.0f}] gap={gap:.1f}s")
    return violations

print(f"Running 10 validation episodes with trained L2 (from: {ckpt_path})")
print(f"Checkpoint episode: {ckpt.get('episode', '?')}")
print()

total_violations = 0
total_held = 0
for ep in range(10):
    l2_trainer.advance_episode()
    env = FiveStationEnv(
        cfg=cfg, seed=1000+ep, batch_window_sec=120.0,
        enable_vt=False, enable_initial_delay=False, l3_worker=None)
    obs, info = env.reset()
    ep_done = False
    ep_held = 0
    all_events = []

    while not ep_done:
        masks = info.get("action_masks", {})
        requests = info.get("requests", [])
        group_obs = _build_critic_inputs(obs, requests)
        gs_vec = compact_global_state(info.get("metrics", {}))

        agent_actions = l2_trainer.collect_step(
            obs, masks, group_obs, gs_vec, device=device)
        action_dict = {aid: (a.platform, a.entry) for aid, a in agent_actions.items()}

        for g in info.get("granted_windows", []):
            station = env._journeys[g.train_id].current_station
            all_events.append({
                "train": g.train_id, "station": station,
                "platform": g.platform,
                "t_arr": g.W_in[0], "t_dep": g.W_out[0],
            })
        ep_held += len(info.get("held_train_ids", []))

        next_obs, reward, terminated, truncated, info = env.step(
            action_dict, lambda_team=l2_trainer.lam_team)
        ep_done = terminated or truncated
        obs = next_obs

    violations = check_overlaps(all_events)
    total_violations += len(violations)
    total_held += ep_held
    metrics = info.get("metrics", {})
    print(f"  ep {ep}: held={ep_held:>2}  violations={len(violations):>1}  "
          f"dep_dly={metrics.get('mean_dep_delay',0):.1f}s  "
          f"ontime={metrics.get('ontime_rate',0):.2f}")

print(f"\nTotal: held={total_held}  violations={total_violations}")
if total_violations == 0:
    print("ALL CLEAN — no platform overlaps")
