#!/usr/bin/env python3
"""Difficulty-tier evaluation.
Generate fixed visitor seeds, grade by heuristic delay, evaluate MAPPO per tier.
"""
import sys
from pathlib import Path
import json
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment.scenario_loader import load_scenario
from src.env.five_station_env import FiveStationEnv, TRAVEL_TIME_TO_THROAT
from src.control.action_mask import ENTRY_CHOICES, compute_joint_action_mask
from src.marl.actor_critic import Layer2ActorCritic


# ── Heuristic episode (empty action_dict triggers env's built-in heuristic) ─
def run_heuristic_episode(env: FiveStationEnv) -> dict:
    obs, info = env.reset()
    ep_done = False
    while not ep_done:
        # Empty dict → env uses heuristic_action for all trains
        obs, reward, terminated, truncated, info = env.step({})
        ep_done = terminated or truncated
    m = info.get("metrics", {})
    return {
        "mean_dep_delay": m.get("mean_dep_delay", 999),
        "steps": env._step_idx,
        "status": info.get("episode_status", "?"),
        "ontime_rate": m.get("ontime_rate", 0),
    }


# ── MAPPO evaluation ──────────────────────────────────────────────────────
def evaluate_mappo_on_seeds(
    ac: Layer2ActorCritic,
    cfg,
    seed_list: list,
    device: str = "cpu",
) -> list:
    results = []
    for seed in seed_list:
        env = FiveStationEnv(cfg=cfg, seed=seed, batch_window_sec=120.0)
        obs, info = env.reset()
        ep_done = False

        while not ep_done:
            active_ids = info.get("active_train_ids", [])
            masks = info.get("action_masks", {})

            action_dict = {}
            for tid in active_ids:
                if tid not in masks:
                    continue
                pm, em = masks[tid]
                o = torch.tensor(obs[tid], dtype=torch.float32, device=device).unsqueeze(0)
                pm_t = torch.tensor(pm + [False]*(4-len(pm)), dtype=torch.bool, device=device).unsqueeze(0)
                em_t = torch.tensor(em, dtype=torch.bool, device=device).unsqueeze(0)
                with torch.no_grad():
                    p_logits, e_logits = ac.actor.logits(o, pm_t, em_t)
                plat_idx = int(torch.argmax(p_logits[0]).item())
                entry_idx = int(torch.argmax(e_logits[0]).item())
                action_dict[tid] = (plat_idx, entry_idx)

            obs, reward, terminated, truncated, info = env.step(action_dict)
            ep_done = terminated or truncated

        m = info.get("metrics", {})
        results.append({
            "seed": seed,
            "mean_dep_delay": m.get("mean_dep_delay", 999),
            "steps": env._step_idx,
            "status": info.get("episode_status", "?"),
            "ontime_rate": m.get("ontime_rate", 0),
        })
    return results


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to MAPPO checkpoint (best.pt or final.pt)")
    parser.add_argument("--seeds", type=int, default=60, help="Number of seeds to scan")
    parser.add_argument("--per-tier", type=int, default=10, help="Seeds per difficulty tier")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_scenario(scenario_name="five_station")

    # ── Phase 1: Grade seeds by heuristic ────────────────────────────────
    print(f"{'='*60}")
    print(f"Phase 1: Grading {args.seeds} seeds by heuristic delay")
    print(f"{'='*60}")

    seed_scores = []
    for seed in range(args.seeds):
        env = FiveStationEnv(cfg=cfg, seed=seed, batch_window_sec=120.0)
        result = run_heuristic_episode(env)
        seed_scores.append((seed, result["mean_dep_delay"], result))
        if (seed + 1) % 20 == 0:
            print(f"  ... {seed+1}/{args.seeds} seeds scanned")

    seed_scores.sort(key=lambda x: x[1])  # sort by delay ascending

    # Tier boundaries: fixed thresholds aligned to natural distribution gaps.
    # Heuristic delay is bimodal: ~0-48s (no visitor conflict), ~56s (normal),
    # 60-736s (platform blocking at D or similar bottleneck).
    T_EASY, T_HARD = 45, 80
    print(f"\nTier boundaries (fixed): Easy < {T_EASY}s, Medium {T_EASY}-{T_HARD}s, Hard > {T_HARD}s")

    tiers = {"Easy": [], "Medium": [], "Hard": []}
    for seed, delay, result in seed_scores:
        if delay < T_EASY:
            tiers["Easy"].append((seed, delay, result))
        elif delay <= T_HARD:
            tiers["Medium"].append((seed, delay, result))
        else:
            tiers["Hard"].append((seed, delay, result))

    print(f"\nSeed distribution by tier:")
    for name in ["Easy", "Medium", "Hard"]:
        entries = tiers[name]
        delays = [e[1] for e in entries]
        print(f"  {name}: {len(entries)} seeds, delay [{min(delays):.0f}-{max(delays):.0f}s]"
              if entries else f"  {name}: 0 seeds")

    # Select evenly from each tier
    selected = {}
    for name in ["Easy", "Medium", "Hard"]:
        entries = tiers[name]
        if len(entries) >= args.per_tier:
            indices = np.linspace(0, len(entries)-1, args.per_tier, dtype=int)
            selected[name] = [entries[i] for i in indices]
        else:
            selected[name] = entries
        delays = [s[1] for s in selected[name]]
        print(f"  {name} selected: {len(selected[name])} seeds, "
              f"delay range [{min(delays):.0f}-{max(delays):.0f}s], mean={np.mean(delays):.1f}s")

    # ── Phase 2: Load MAPPO model ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 2: Loading MAPPO checkpoint")
    print(f"{'='*60}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    hp = ckpt.get("hp", {})
    obs_dim = hp.get("obs_dim", 25)
    k_local = hp.get("k_local", 1)
    hidden_dim = hp.get("hidden_dim", 128)

    ac = Layer2ActorCritic(obs_dim=obs_dim, k_local=k_local, hidden_dim=hidden_dim)
    ac.load_state_dict(ckpt["ac_state"])
    ac.to(device)
    ac.eval()
    print(f"  Loaded checkpoint (episode {ckpt.get('episode', '?')}, obs_dim={obs_dim})")

    # ── Phase 3: Evaluate MAPPO per tier ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 3: Evaluating MAPPO per tier")
    print(f"{'='*60}")

    tier_results = {}
    for tier_name in ["Easy", "Medium", "Hard"]:
        entries = selected[tier_name]
        seeds = [e[0] for e in entries]
        results = evaluate_mappo_on_seeds(ac, cfg, seeds, device=device)

        h_dly = [e[1] for e in entries]
        m_dly = [r["mean_dep_delay"] for r in results]
        h_ot = [e[2]["ontime_rate"] for e in entries]
        m_ot = [r["ontime_rate"] for r in results]

        td = {
            "heuristic_delays": h_dly, "mappo_delays": m_dly,
            "heuristic_ontimes": h_ot, "mappo_ontimes": m_ot,
            # Mean ± std (for reporting)
            "h_d_mean": np.mean(h_dly), "m_d_mean": np.mean(m_dly),
            "h_d_std": np.std(h_dly), "m_d_std": np.std(m_dly),
            "h_ot_mean": np.mean(h_ot), "m_ot_mean": np.mean(m_ot),
            # Median ± IQR (robust to outliers)
            "h_d_med": np.median(h_dly), "m_d_med": np.median(m_dly),
            "h_ot_med": np.median(h_ot), "m_ot_med": np.median(m_ot),
        }
        tier_results[tier_name] = td

        print(f"\n  {tier_name} ({len(seeds)} seeds):")
        print(f"    Heuristic:  delay (mean)={td['h_d_mean']:.1f}s ± {td['h_d_std']:.1f},  "
              f"(median)={td['h_d_med']:.1f}s,  ontime={td['h_ot_mean']*100:.0f}%")
        print(f"    MAPPO:      delay (mean)={td['m_d_mean']:.1f}s ± {td['m_d_std']:.1f},  "
              f"(median)={td['m_d_med']:.1f}s,  ontime={td['m_ot_mean']*100:.0f}%")
        impr_d_med = (1 - td['m_d_med']/max(td['h_d_med'], 1))
        impr_o = td['m_ot_mean'] - td['h_ot_mean']
        print(f"    Improvement: delay (median) {impr_d_med*100:.0f}%, ontime +{impr_o*100:.0f}pt")

    # ── Phase 4: Plot ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 4: Generating difficulty-tier figure")
    print(f"{'='*60}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    tier_names = ["Easy", "Medium", "Hard"]
    colors = ["#4daf4a", "#ff7f00", "#e41a1c"]
    x = np.arange(len(tier_names))
    width = 0.32
    gap = 0.04

    # Delay bar chart
    ax = axes[0]
    for i, name in enumerate(tier_names):
        td = tier_results[name]
        h_q1, h_q3 = np.percentile(td["heuristic_delays"], [25, 75])
        m_q1, m_q3 = np.percentile(td["mappo_delays"], [25, 75])

        h_bar = ax.bar(i - width/2 - gap, td["h_d_med"], width,
                       yerr=[[td["h_d_med"]-h_q1], [h_q3-td["h_d_med"]]],
                       color='#cccccc', edgecolor='white', linewidth=0.8,
                       error_kw={'linewidth': 1.2, 'ecolor': '#888888', 'capsize': 3},
                       label='Heuristic' if i==0 else '')
        m_bar = ax.bar(i + width/2 + gap, td["m_d_med"], width,
                       yerr=[[td["m_d_med"]-m_q1], [m_q3-td["m_d_med"]]],
                       color=colors[i], edgecolor='white', linewidth=0.8,
                       error_kw={'linewidth': 1.2, 'ecolor': colors[i], 'capsize': 3},
                       label='MAPPO' if i==0 else '')

        # Improvement label above MAPPO bar
        impr = (1 - td["m_d_med"]/max(td["h_d_med"], 1))*100
        bar_top = td["m_d_med"] + (m_q3 - td["m_d_med"])
        ax.text(i + width/2 + gap, bar_top + max(4, bar_top*0.08),
                f'{impr:+.0f}%', ha='center', fontsize=10, fontweight='bold',
                color=colors[i] if impr > 0 else '#999999')

    ax.set_xticks(x)
    ax.set_xticklabels(tier_names, fontsize=11)
    ax.set_ylabel('Departure Delay (s)', fontsize=11)
    ax.set_title('Delay by Difficulty Tier', fontweight='bold', fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    # Ensure y=0 is visible
    ax.set_ylim(bottom=-5)

    # Ontime bar chart
    ax = axes[1]
    for i, name in enumerate(tier_names):
        td = tier_results[name]
        h_ot_all = [v*100 for v in td["heuristic_ontimes"]]
        m_ot_all = [v*100 for v in td["mappo_ontimes"]]
        h_med = td["h_ot_med"] * 100
        m_med = td["m_ot_med"] * 100
        h_q1, h_q3 = np.percentile(h_ot_all, [25, 75])
        m_q1, m_q3 = np.percentile(m_ot_all, [25, 75])

        ax.bar(i - width/2 - gap, h_med, width,
               yerr=[[h_med-h_q1], [h_q3-h_med]],
               color='#cccccc', edgecolor='white', linewidth=0.8,
               error_kw={'linewidth': 1.2, 'ecolor': '#888888', 'capsize': 3})
        ax.bar(i + width/2 + gap, m_med, width,
               yerr=[[m_med-m_q1], [m_q3-m_med]],
               color=colors[i], edgecolor='white', linewidth=0.8,
               error_kw={'linewidth': 1.2, 'ecolor': colors[i], 'capsize': 3})

        delta = m_med - h_med
        bar_top = m_med + (m_q3 - m_med)
        ax.text(i + width/2 + gap, min(103, bar_top + 3),
                f'{delta:+.0f}pt', ha='center', fontsize=10, fontweight='bold',
                color=colors[i] if delta > 0 else '#999999')
        # Also annotate the heuristic value on the gray bar
        ax.text(i - width/2 - gap, h_med / 2, f'{h_med:.0f}%',
                ha='center', fontsize=7, color='#666666')

    ax.set_xticks(x)
    ax.set_xticklabels(tier_names, fontsize=11)
    ax.set_ylabel('On-Time Rate (%)', fontsize=11)
    ax.set_title('On-Time Rate by Difficulty Tier', fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(-2, 108)

    fig.suptitle('MAPPO vs Heuristic — Difficulty-Tier Evaluation\n'
                 f'{args.seeds} seeds graded by heuristic delay, {args.per_tier} seeds/tier, '
                 'bars = median, error = IQR',
                 fontweight='bold', y=1.02)
    plt.tight_layout()

    pic_dir = PROJECT_ROOT / "output" / "pic_of_3"
    fig_path = pic_dir / "fig5_difficulty_tiers.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig_path}")
    plt.close('all')

    # Save data
    data_path = pic_dir / "difficulty_tiers.json"
    save_data = {
        name: {
            "seeds": [int(s[0]) for s in selected[name]],
            "heuristic_delays": [float(x) for x in tier_results[name]["heuristic_delays"]],
            "mappo_delays": [float(x) for x in tier_results[name]["mappo_delays"]],
            "heuristic_ontimes": [float(x) for x in tier_results[name]["heuristic_ontimes"]],
            "mappo_ontimes": [float(x) for x in tier_results[name]["mappo_ontimes"]],
            "h_d_mean": float(tier_results[name]["h_d_mean"]),
            "m_d_mean": float(tier_results[name]["m_d_mean"]),
            "h_d_med": float(tier_results[name]["h_d_med"]),
            "m_d_med": float(tier_results[name]["m_d_med"]),
        }
        for name in tier_names
    }
    with open(data_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Data: {data_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
