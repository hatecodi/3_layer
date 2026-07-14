"""Debug: trace one episode to find why trains get held."""
import sys, torch
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.env.five_station_env import FiveStationEnv
from src.env.observation_builder import get_obs_dim, critic_input
from src.marl.actor_critic import Layer2ActorCritic, compact_global_state
from src.marl.mappo import MAPPOTrainer
from src.marl.rollout_buffer import RolloutBuffer, AgentTransition
from src.experiment.scenario_loader import load_scenario
from src.control.resource_manager import build_conflict_graph
from src.control.action_mask import ENTRY_CHOICES

HP = {
    "obs_dim": 29, "action_dim": 45, "k_local": 1, "hidden_dim": 128,
    "lr": 3e-4, "gamma": 0.99, "gae_lambda": 0.95, "clip_ratio": 0.2,
    "ppo_epochs": 5, "minibatch_size": 256,
    "entropy_coef_start": 0.01, "entropy_coef_end": 0.005,
    "value_coef": 0.5, "max_grad_norm": 1.0, "rollout_steps": 512,
}


def _build_critic_inputs(obs, requests):
    conflict_graph = build_conflict_graph(requests)
    group_obs = {}
    for agent_id in obs:
        group_obs[agent_id] = critic_input(agent_id, obs, conflict_graph, k_max=HP["k_local"])
    return group_obs


def main():
    cfg = load_scenario(scenario_name="five_station")
    obs_dim = get_obs_dim(cfg)
    device = torch.device("cpu")

    ac = Layer2ActorCritic(obs_dim=obs_dim, k_local=HP["k_local"], hidden_dim=HP["hidden_dim"])
    l2_trainer = MAPPOTrainer(
        ac, lr=HP["lr"], gamma=HP["gamma"], gae_lambda=HP["gae_lambda"],
        clip_ratio=HP["clip_ratio"], ppo_epochs=HP["ppo_epochs"],
        minibatch_size=HP["minibatch_size"], entropy_coef=HP["entropy_coef_start"],
        value_coef=HP["value_coef"], max_grad_norm=HP["max_grad_norm"],
        total_episodes=1, device=device)

    torch.manual_seed(42)
    l2_trainer.advance_episode()
    env = FiveStationEnv(
        cfg=cfg, seed=0, batch_window_sec=120.0,
        enable_vt=False, enable_initial_delay=False, l3_worker=None)
    obs, info = env.reset()
    ep_done = False
    step_count = 0
    total_held = 0
    held_reasons = defaultdict(int)

    while not ep_done:
        masks = info.get("action_masks", {})
        requests = info.get("requests", [])
        pending = info.get("pending", [])
        held_ids = info.get("held_train_ids", [])
        total_held += len(held_ids)
        group_obs = _build_critic_inputs(obs, requests)
        gs_vec = compact_global_state(info.get("metrics", {}))

        agent_actions = l2_trainer.collect_step(
            obs, masks, group_obs, gs_vec, device=device)
        action_dict = {aid: (a.platform, a.entry) for aid, a in agent_actions.items()}

        # Print persistent calendar state
        cal = env._persistent_cal
        active_bookings = []
        for rid, windows in cal._windows.items():
            for w in windows:
                if w.end > env._t_now:
                    active_bookings.append(f"  {rid}: {w.train_id} [{w.start:.0f},{w.end:.0f}]")

        if step_count < 10:
            print(f"\n--- Step {step_count} (t={env._t_now:.0f}) ---")
            print(f"  Pending: {pending}")
            if held_ids:
                print(f"  Held: {held_ids}")
            print(f"  Requests built: {[r.train_id for r in requests]}")
            for r in requests:
                print(f"    {r.train_id}: plat={r.platform} W_in=({r.W_in[0]:.0f},{r.W_in[1]:.0f}) "
                      f"W_plat=({r.W_plat[0]:.0f},{r.W_plat[1]:.0f}) "
                      f"W_out=({r.W_out[0]:.0f},{r.W_out[1]:.0f})")
            if active_bookings:
                print(f"  Persistent bookings (future):")
                for b in active_bookings:
                    print(b)
            print(f"  Action mask samples:")
            for tid in pending[:3]:
                m = masks.get(tid)
                if m:
                    p_mask = [i for i, v in enumerate(m[0]) if v]
                    e_mask = [i for i, v in enumerate(m[1]) if v]
                    print(f"    {tid}: platforms={p_mask}, entries={e_mask}")

        next_obs, reward, terminated, truncated, info = env.step(
            action_dict, lambda_team=l2_trainer.lam_team)
        ep_done = terminated or truncated
        obs = next_obs
        step_count += 1
        if step_count > 100:
            print("  (truncated at 100 steps)")
            break

    print(f"\n=== Summary: {step_count} steps, {total_held} held, status={info.get('episode_status')} ===")


if __name__ == "__main__":
    main()
