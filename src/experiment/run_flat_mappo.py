#!/usr/bin/env python3
"""Flat MAPPO — joint scheduling+control (72 actions, no hierarchy)."""
from __future__ import annotations
import sys, csv, json, argparse, math
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # src/experiment -> repo root
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment.scenario_loader import load_scenario
from src.env.five_station_env import FiveStationEnv
from src.env.observation_builder import get_obs_dim, critic_input, build_l3_observation
from src.marl.actor_critic import compact_global_state, _mlp
from src.control.resource_manager import build_conflict_graph
from src.control.l3_sim import SpeedMode

# ---------------------------------------------------------------------------
# Flat joint actor: platform(4) × entry(6) × speed(3) = 72 actions
# ---------------------------------------------------------------------------
PLATFORM_DIM = 4
ENTRY_DIM = 6
SPEED_DIM = 3
FLAT_ACTION_DIM = PLATFORM_DIM * ENTRY_DIM * SPEED_DIM  # 72

def flat_action_index(platform, entry, speed):
    return (platform * ENTRY_DIM + entry) * SPEED_DIM + speed

def unpack_flat_action(idx):
    speed = idx % SPEED_DIM
    idx //= SPEED_DIM
    entry = idx % ENTRY_DIM
    platform = idx // ENTRY_DIM
    return platform, entry, speed

class FlatJointActor(nn.Module):
    """Single 72-way categorical head for platform+entry+speed."""
    def __init__(self, obs_dim=31, action_dim=FLAT_ACTION_DIM, hidden_dim=128):
        super().__init__()
        self.action_dim = action_dim
        self.norm = nn.LayerNorm(obs_dim)
        self.net = _mlp(obs_dim, hidden_dim, action_dim)

    def logits(self, obs, mask=None):
        h = self.norm(obs)
        logits = self.net(h)
        if mask is not None:
            logits = logits.masked_fill(~mask, float('-inf'))
        return logits

    def dist(self, obs, mask=None):
        return Categorical(logits=self.logits(obs, mask))

class FlatCritic(nn.Module):
    def __init__(self, obs_dim=31, hidden_dim=128):
        super().__init__()
        self.norm = nn.LayerNorm(obs_dim)
        self.net = _mlp(obs_dim, hidden_dim, 1)

    def forward(self, obs):
        return self.net(self.norm(obs)).squeeze(-1)

# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------
class FlatTransition:
    def __init__(self, obs, action, logprob, value, reward, done, mask):
        self.obs = obs
        self.action = action
        self.logprob = logprob
        self.value = value
        self.reward = reward
        self.done = done
        self.mask = mask  # 72-dim boolean

class FlatBuffer:
    def __init__(self, gamma=0.99, gae_lambda=0.95):
        self.gamma = gamma; self.gae_lambda = gae_lambda
        self._transitions: List[FlatTransition] = []

    def add(self, t: FlatTransition): self._transitions.append(t)
    def clear(self): self._transitions.clear()
    def __len__(self): return len(self._transitions)

    def compute_gae(self, final_value=0.0):
        T = len(self._transitions)
        adv = torch.zeros(T); ret = torch.zeros(T)
        gae = 0.0; nv = final_value
        for t in reversed(range(T)):
            tr = self._transitions[t]
            delta = tr.reward + self.gamma * nv * (1 - int(tr.done)) - tr.value
            gae = delta + self.gamma * self.gae_lambda * (1 - int(tr.done)) * gae
            adv[t] = gae; ret[t] = gae + tr.value; nv = tr.value
        return adv, ret

# ---------------------------------------------------------------------------
# Flat MAPPO trainer
# ---------------------------------------------------------------------------
class FlatMAPPOTrainer:
    def __init__(self, actor, critic, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_ratio=0.2, ppo_epochs=5, entropy_coef=0.01,
                 value_coef=0.5, max_grad_norm=1.0, device=None):
        self.actor = actor; self.critic = critic
        self.device = device or torch.device('cpu')
        self.actor.to(self.device); self.critic.to(self.device)
        self.optimizer = Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
        self.gamma = gamma; self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio; self.ppo_epochs = ppo_epochs
        self.entropy_coef = entropy_coef; self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.buffer = FlatBuffer(gamma, gae_lambda)

    def update(self):
        buffer = self.buffer
        if len(buffer) == 0: return {}
        advantages, returns = buffer.compute_gae(0.0)

        obs = torch.stack([t.obs for t in buffer._transitions]).to(self.device)
        old_logprobs = torch.tensor([t.logprob for t in buffer._transitions], device=self.device)
        old_actions = torch.tensor([t.action for t in buffer._transitions], device=self.device)
        masks = torch.stack([t.mask for t in buffer._transitions]).to(self.device)
        advantages = advantages.to(self.device); returns = returns.to(self.device)

        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        stats = {}
        for _ in range(self.ppo_epochs):
            dist = self.actor.dist(obs, masks)
            new_logprobs = dist.log_prob(old_actions)
            entropy = dist.entropy().mean()
            values = self.critic(obs)

            ratio = torch.exp(new_logprobs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            loss_policy = -torch.min(surr1, surr2).mean()
            loss_value = F.mse_loss(values, returns)
            loss = loss_policy + self.value_coef * loss_value - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.max_grad_norm)
            self.optimizer.step()

            stats = {'loss_policy': loss_policy.item(), 'loss_value': loss_value.item(), 'entropy': entropy.item()}

        buffer.clear()
        return stats

# ---------------------------------------------------------------------------
# Build flat action mask: which (platform, entry, speed) combos are legal
# ---------------------------------------------------------------------------
def build_flat_mask(p_mask, e_mask):
    """Combine factored (platform_mask, entry_mask) into 72-dim flat mask.
    All 3 speed modes are always legal (speed choice doesn't affect station feasibility)."""
    flat = torch.zeros(FLAT_ACTION_DIM, dtype=torch.bool)
    n_plat = min(len(p_mask), PLATFORM_DIM)
    n_entry = len(e_mask)
    for p in range(n_plat):
        if not p_mask[p]: continue
        for e in range(n_entry):
            if not e_mask[e]: continue
            for s in range(SPEED_DIM):
                flat[flat_action_index(p, e, s)] = True
    return flat

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
HP = {
    'obs_dim': 31, 'hidden_dim': 128, 'lr': 3e-4,
    'gamma': 0.99, 'gae_lambda': 0.95, 'clip_ratio': 0.2,
    'ppo_epochs': 5, 'entropy_coef_start': 0.01, 'entropy_coef_end': 0.005,
    'value_coef': 0.5, 'max_grad_norm': 1.0, 'rollout_steps': 512,
}

def eval_flat_heldout(actor, cfg, seeds, device, incident=True):
    """Run held-out incident episodes with the current flat actor (no learning).
    Returns {arr_ontime, dep_ontime, n_arr, succ, deadlock, trunc}."""
    n_ontime_arr = 0; n_arr = 0; n_ontime_dep = 0; n_granted = 0
    succ = 0; deadlock = 0; trunc = 0
    for s in seeds:
        env = FiveStationEnv(cfg, seed=s, batch_window_sec=120.0, enable_vt=False,
                             enable_initial_delay=False,
                             enable_corridor_incident=incident, l3_worker=None)
        obs, info = env.reset()
        done = False
        speed_choices: Dict[str, int] = {}
        while not done:
            masks = info.get('action_masks', {})
            action_dict = {}
            for tid in info.get('pending', []):
                m = masks.get(tid)
                if not m or not any(m[0]) or not any(m[1]):
                    continue
                obs_t = torch.tensor(obs[tid], dtype=torch.float32, device=device).unsqueeze(0)
                flat_mask = build_flat_mask(m[0], m[1]).to(device)
                with torch.no_grad():
                    a = actor.dist(obs_t, flat_mask.unsqueeze(0)).sample()
                p, e, sp = unpack_flat_action(a.item())
                action_dict[tid] = (p, e)
                speed_choices[tid] = sp
            env._flat_speed_choices = speed_choices
            obs, _, term, trunc_, info = env.step(action_dict, lambda_team=0.3)
            done = term or trunc_
            mm = info.get('metrics', {})
            n_granted += mm.get('n_granted', 0)
            n_ontime_dep += mm.get('n_ontime_dep', 0)
            n_ontime_arr += mm.get('n_ontime_arr', 0)
            n_arr += mm.get('n_arrivals', 0)
        status = info.get('episode_status', '?')
        succ += status == 'success'; deadlock += status == 'deadlock'
        trunc += status == 'truncated'
    return {'arr_ontime': n_ontime_arr / max(n_arr, 1),
            'dep_ontime': n_ontime_dep / max(n_granted, 1),
            'n_arr': n_arr, 'succ': succ, 'deadlock': deadlock, 'trunc': trunc}


def train_flat_mappo(episodes=7000, scenario="five_station_15t", incident=False,
                     out_root="output", use_cpu=False,
                     eval_every=0, eval_n=12, eval_seed_base=900000):
    device = torch.device("cuda" if torch.cuda.is_available() and not use_cpu else "cpu")
    cfg = load_scenario(scenario_name=scenario)
    obs_dim = get_obs_dim(cfg)

    actor = FlatJointActor(obs_dim=obs_dim, hidden_dim=HP['hidden_dim'])
    critic = FlatCritic(obs_dim=obs_dim, hidden_dim=HP['hidden_dim'])
    trainer = FlatMAPPOTrainer(actor, critic, lr=HP['lr'], gamma=HP['gamma'],
        gae_lambda=HP['gae_lambda'], clip_ratio=HP['clip_ratio'],
        ppo_epochs=HP['ppo_epochs'], entropy_coef=HP['entropy_coef_start'],
        value_coef=HP['value_coef'], max_grad_norm=HP['max_grad_norm'], device=device)

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = PROJECT_ROOT / out_root / 'flat_mappo' / run_id
    ckpt_dir = run_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    episode_metrics = []
    total_ep = 0
    best_reward = -float('inf')
    # Store speed choices per train (set during step, read in _advance_trains)
    speed_choices: Dict[str, int] = {}
    # Rolling window of arrival ontime (realised L3 punctuality) for the
    # episodes-to-converge comparison vs hierarchical (flat's departure ontime
    # converges to ~88% but arrival stays ~40%, so arrival is the honest metric).
    win_arr_ontime = deque(maxlen=50)
    eval_rows = []   # held-out eval at checkpoints (eval_every > 0)

    for ep in range(episodes):
        # Entropy decay
        progress = min(ep / max(episodes * 0.3, 1), 1.0)
        trainer.entropy_coef = HP['entropy_coef_start'] + (HP['entropy_coef_end'] - HP['entropy_coef_start']) * progress

        env = FiveStationEnv(cfg, seed=ep, batch_window_sec=120.0,
                             enable_vt=False,
                             enable_initial_delay=not incident,
                             enable_corridor_incident=incident,
                             l3_worker=None)
        obs, info = env.reset()
        done = False

        ep_reward = defaultdict(float)
        ep_granted = 0; ep_dep = 0.0; ep_ontime = 0
        ep_ontime_arr = 0; ep_arrivals = 0   # realised (L3) arrival punctuality

        while not done:
            masks = info.get('action_masks', {})
            requests = info.get('requests', [])
            gs_vec = compact_global_state(info.get('metrics', {}))

            action_dict = {}
            step_transitions = []

            for tid in info.get('pending', []):
                m = masks.get(tid)
                if not m or not any(m[0]) or not any(m[1]):
                    continue

                obs_t = torch.tensor(obs[tid], dtype=torch.float32, device=device).unsqueeze(0)
                flat_mask = build_flat_mask(m[0], m[1]).to(device)

                with torch.no_grad():
                    dist = actor.dist(obs_t, flat_mask.unsqueeze(0))
                    action = dist.sample()
                    logprob = dist.log_prob(action)
                    value = critic(obs_t)
                    entropy_val = dist.entropy()

                p_idx, e_idx, s_idx = unpack_flat_action(action.item())
                action_dict[tid] = (p_idx, e_idx)
                speed_choices[tid] = s_idx

                step_transitions.append(FlatTransition(
                    obs=obs_t.squeeze(0).cpu(),
                    action=action.item(),
                    logprob=logprob.item(),
                    value=value.item(),
                    reward=0.0,  # filled after step
                    done=False,
                    mask=flat_mask.cpu(),
                ))

            # Inject speed_choices into env for L3 transit
            env._flat_speed_choices = speed_choices

            obs, reward, terminated, truncated, info = env.step(action_dict, lambda_team=0.3)
            done = terminated or truncated

            m = info.get('metrics', {})
            ep_granted += m.get('n_granted', 0)
            ep_dep += m.get('total_dep_delay', 0)
            ep_ontime += m.get('n_ontime_dep', 0)
            ep_ontime_arr += m.get('n_ontime_arr', 0)
            ep_arrivals += m.get('n_arrivals', 0)

            reward_breakdown = info.get('reward_breakdown', {})
            for i, tr in enumerate(step_transitions):
                tid = list(action_dict.keys())[i] if i < len(action_dict) else ''
                r_local = reward_breakdown.get(tid, {}).get('local', 0.0)
                r_global = reward_breakdown.get(tid, {}).get('global_', 0.0)
                tr.reward = r_local + r_global
                tr.done = done
                trainer.buffer.add(tr)

            for tid, r in reward.items():
                ep_reward[tid] += r

            if len(trainer.buffer) >= HP['rollout_steps']:
                trainer.update()

        # End of episode
        total_r = sum(ep_reward.values())
        mean_r = total_r / max(len(ep_reward), 1)
        mean_dep = ep_dep / max(ep_granted, 1)
        ontime = ep_ontime / max(ep_granted, 1)
        arr_ontime = ep_ontime_arr / max(ep_arrivals, 1)

        episode_metrics.append({
            'episode': ep, 'mean_reward': mean_r, 'mean_dep_delay': mean_dep,
            'ontime_rate': ontime, 'arr_ontime': arr_ontime,
            'status': info.get('episode_status', '?'),
        })
        win_arr_ontime.append(arr_ontime)

        if (ep+1) % max(1, episodes//20) == 0 or ep < 5:
            print(f'  ep {ep:>5} | reward={mean_r:>7.2f} | dep_dly={mean_dep:>6.1f}s | dep_ontime={ontime:.2f} | '
                  f'arr_ontime={sum(win_arr_ontime)/len(win_arr_ontime):.2f} | {info.get("episode_status","?")}')

        # ---- held-out eval hook (fixed seeds, no learning) ----
        if eval_every > 0 and (ep + 1) % eval_every == 0:
            seeds = [eval_seed_base + i * 7919 for i in range(eval_n)]
            ev = eval_flat_heldout(actor, cfg, seeds, device, incident=incident)
            eval_rows.append({'episode': ep + 1, 'arr_ontime': round(ev['arr_ontime'], 4),
                              'dep_ontime': round(ev['dep_ontime'], 4),
                              'n_arr': ev['n_arr'], 'succ': ev['succ'],
                              'deadlock': ev['deadlock'], 'trunc': ev['trunc']})
            print(f'    [heldout @ ep {ep+1}] arr_ontime={ev["arr_ontime"]:.3f} '
                  f'(succ {ev["succ"]}/{len(seeds)})')

        if mean_r > best_reward:
            best_reward = mean_r
            torch.save({'actor': actor.state_dict(), 'critic': critic.state_dict(),
                       'episode': ep, 'hp': HP}, ckpt_dir / 'best.pt')

        total_ep += 1

    # Save
    torch.save({'actor': actor.state_dict(), 'critic': critic.state_dict(),
               'hp': HP, 'episode_metrics': episode_metrics}, ckpt_dir / 'final.pt')

    csv_path = run_dir / 'training_metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=episode_metrics[0].keys())
        w.writeheader(); w.writerows(episode_metrics)

    if eval_rows:
        ev_path = run_dir / 'eval_ontime.csv'
        with open(ev_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys()))
            w.writeheader(); w.writerows(eval_rows)
        print(f'held-out eval CSV: {ev_path}')

    last = episode_metrics[-300:]
    avg_dep = sum(e['mean_dep_delay'] for e in last) / len(last)
    print(f'\nFlat MAPPO complete: last300 dep={avg_dep:.1f}s')
    print(f'CSV: {csv_path}')
    return avg_dep

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='five_station_15t')
    parser.add_argument('--incident', action='store_true',
                        help='corridor speed-restriction incident as the training '
                             'disturbance (disables origin-bunching)')
    parser.add_argument('--episodes', type=int, default=7000)
    parser.add_argument('--lr', type=float, default=HP['lr'])
    parser.add_argument('--out-root', type=str, default='output',
                        help="root dir under PROJECT_ROOT for this run, e.g. "
                             "output/8_16/training (default 'output')")
    parser.add_argument('--cpu', action='store_true',
                        help='force CPU (default: auto CUDA)')
    parser.add_argument('--eval-every', type=int, default=0,
                        help='run a held-out incident eval every N training eps (0=off)')
    parser.add_argument('--eval-n', type=int, default=12,
                        help='number of held-out episodes per eval checkpoint')
    parser.add_argument('--eval-seed-base', type=int, default=900000,
                        help='base seed for the deterministic held-out eval set')
    args = parser.parse_args()
    HP['lr'] = args.lr
    train_flat_mappo(episodes=args.episodes, scenario=args.scenario,
                     incident=args.incident, out_root=args.out_root,
                     use_cpu=args.cpu, eval_every=args.eval_every,
                     eval_n=args.eval_n, eval_seed_base=args.eval_seed_base)
