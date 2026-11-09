#!/usr/bin/env python3
"""Standard verification protocol: fixed held-out eval set, stratified by the
ACTIVE disturbance (bunched origin-delay OR corridor speed-restriction incident),
full delay distribution (realised ARRIVAL delay from corridor transit events),
learned/SA/CRUISE.  The CRUISE baseline runs the SAME per-corridor physics as
worker/SA (incl. the incident restriction), so comparisons are apples-to-apples.

Incident arm (--incident): instead of an absolute >60s rate (meaningless when
delay is inevitable), reports the PAIRED excess vs the CRUISE baseline on the
incident-affected subset — the coordination value is the delay reduction relative
to the no-coordination floor, measured on the trains whose corridor transit
overlapped the restriction window.

Usage:
    # origin-bunching (legacy stratification)
    python scripts/eval_protocol.py --scenario five_station_15t --l2 <l2.pt> --l3 <l3.pt> --n 60

    # corridor speed-restriction incident (incident-aware + realised arrival)
    python scripts/eval_protocol.py --scenario five_station_15t --l2 <l2.pt> --l3 <l3.pt> \
        --incident --n 60 --blocks 3
    python scripts/eval_protocol.py --scenario five_station_15t --sa-only    ... (SA)
    python scripts/eval_protocol.py --scenario five_station_15t --sa-online-only ... (partial-obs SA)
    python scripts/eval_protocol.py --scenario five_station_15t --sa-nofc-only ... (SA w/o forecast)
    python scripts/eval_protocol.py --scenario five_station_15t --cruise-only ...   (CRUISE)
    python scripts/eval_protocol.py --scenario five_station_15t --fcfs-only ...     (rule FCFS)

Modes (default set: learned, sa, sa-online, sa-nofc, fcfs, cruise [+flat]):
    learned / sa / sa-online / sa-nofc  — learned L2, differing L3 execution layers
    fcfs   — RULE L2 (heuristic_action, planned-departure order) + rule CRUISE L3.
             The only genuinely non-learning baseline; contrast with `cruise`, which
             still uses the LEARNED L2 and so isolates L3 alone.
    cruise — learned L2 + rule CRUISE L3 (L3-only ablation, NOT a non-learning baseline)
    Caveat: conflict arbitration (resource_manager.resolve_all) stays priority-ordered
    for every mode, so `fcfs` is "FCFS action rule + shared priority resolver", not
    textbook strict FCFS.
    # no-disturbance arm: --no-delay disables origin bunching (clean timetable);
    # combine with the incident ckpts (trained under no-bunching) for an
    # in-distribution learned comparison.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment.run_flat_mappo import (
    FlatJointActor, build_flat_mask, unpack_flat_action, FLAT_ACTION_DIM)

from src.env.five_station_env import FiveStationEnv
from src.env.observation_builder import get_obs_dim, critic_input
from src.env.reward import L2_PRIORITY_WEIGHT, ONTIME_THRESHOLD_SEC

# 严重晚点 threshold: > 60 sim-s (~25 real min) — the TAIL metric.  Captures
# "尾端恢复": how SEVERE the worst-delayed trains are, i.e. whether the system
# bounds the long tail or leaves some trains badly late.
SEVERE_DELAY_SEC = 60.0

# Layer-1 action-mask rejection reasons (see action_mask.illegal_reason) — the
# mask diagnostics show WHICH bound was binding, not just that masking happened.
MASK_REASONS = [
    "arr_min", "dep_max", "dwell_min",            # timetable envelope
    "platform_not_candidate", "unknown_platform",  # route/platform compatibility
    "platform_not_through", "passthrough_dwell",   # through-track restriction
    "no_station",
]


def _tail_severity(delays):
    """(n, rate%, mean, max) of severely-delayed arrivals (arr_delay > SEVERE)."""
    sev = [d for d in delays if d > SEVERE_DELAY_SEC]
    n = len(sev)
    if n == 0:
        return (0, 0.0, float("nan"), float("nan"))
    return (n, n / max(len(delays), 1) * 100, sum(sev) / n, max(sev))
from src.marl.actor_critic import Layer2ActorCritic, compact_global_state
from src.marl.mappo import MAPPOTrainer
from src.marl.l3_worker import CorridorActor, CorridorCritic, L3WorkerTrainer, L3_OBS_DIM
from src.experiment.scenario_loader import load_scenario
from src.control.resource_manager import build_conflict_graph
from src.experiment.recovery_metrics import (
    DDT_THETA_SEC, DDT_K, recovery_stats, recovery_grid, floor_index, with_floor)


def _load_l2(cfg, l2_ckpt, device):
    obs_dim = get_obs_dim(cfg)
    ckpt = torch.load(l2_ckpt, map_location=device, weights_only=False)
    ac = Layer2ActorCritic(obs_dim=obs_dim, k_local=1, hidden_dim=128)
    try:
        ac.load_state_dict(ckpt["ac_state"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"L2 checkpoint incompatible with obs_dim={obs_dim} "
            "(obs changes 31->34->37: overtaking + corridor closure); "
            "retrain L2 on this branch"
        ) from exc
    ac.eval()
    for p in ac.parameters():
        p.requires_grad = False
    tr = MAPPOTrainer(ac, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2, ppo_epochs=5,
                      minibatch_size=256, entropy_coef=0.01, value_coef=0.5, max_grad_norm=1.0,
                      total_episodes=1, device=device)
    return tr


def _load_l3(l3_ckpt, device):
    ckpt = torch.load(l3_ckpt, map_location=device, weights_only=False)
    a = CorridorActor(L3_OBS_DIM, hidden_dim=128); a.load_state_dict(ckpt["l3_actor"]); a.eval()
    c_local = CorridorCritic(L3_OBS_DIM, hidden_dim=128)
    c_global = CorridorCritic(L3_OBS_DIM, hidden_dim=128)
    if "l3_critic_local" in ckpt:
        c_local.load_state_dict(ckpt["l3_critic_local"])
        c_global.load_state_dict(ckpt["l3_critic_global"])
    else:
        # LEGACY 1-critic checkpoint (pre-c3ff7ba; keys: ac_state/l3_actor/l3_critic).
        # This fallback is EVAL-SAFE and must NOT be used for training:
        #   L3WorkerTrainer.select_action samples the action from `actor` ONLY and
        #   merely REPORTS the two critic values, which land in the transition and
        #   are never consumed when no update runs.  So putting the single
        #   `l3_critic` into critic_local and leaving critic_global at init has
        #   ZERO effect on the rollout — it only exists so historical policies can
        #   be evaluated on the current code.
        c_local.load_state_dict(ckpt["l3_critic"])
        print("    [l3] legacy 1-critic ckpt — critics are inert at eval; "
              "l3_critic -> critic_local")
    c_local.eval(); c_global.eval()
    for p in list(a.parameters()) + list(c_local.parameters()) + list(c_global.parameters()):
        p.requires_grad = False
    return L3WorkerTrainer(a, c_local, c_global, lr=1e-3, device=device)


def run_episode(cfg, tr, l3, seed, mode, dev, incident_on, vt, no_delay=False,
                extra_train=False, severity=1.0, flat_actor=None, sa_iters=300,
                relevant_l2=False, yield_cap=False, closure_on=False,
                coverage="base"):
    env = FiveStationEnv(cfg=cfg, seed=seed, enable_vt=vt,
                         enable_initial_delay=(not incident_on and not no_delay
                                               and not closure_on),
                         enable_corridor_incident=incident_on,
                         enable_corridor_closure=closure_on,
                         control_coverage=coverage,
                         l3_worker=(l3 if mode == "learned" else None),
                         l3_sa=(mode in ("sa", "heur-sa")),
                         l3_sa_iters=sa_iters,
                         yield_entry_cap=yield_cap,
                         l3_sa_online=(mode == "sa-online"),
                         l3_sa_nofc=(mode == "sa-nofc"),
                         enable_extra_train=extra_train,
                         incident_scale=severity)
    obs, info = env.reset()
    incident = dict(env._corridor_incidents) if incident_on else {}
    closures = dict(env._corridor_closures) if closure_on else {}
    # legacy bunching classification (only meaningful without --incident)
    bunched = 0
    released = []          # actual release time of every bunched train
    for tid, j in env._journeys.items():
        t = cfg.trains[tid]
        if j.current_station == t.origin_station:
            p = t.planned.get(t.origin_station)
            if p and j.arrival_at_current > p.dep + 1:
                bunched += 1
                released.append(j.arrival_at_current)
    # ---- delay-dissipation anchor (recovery_metrics.py) ----
    # The disturbance END, never its onset: a train inside the injection window
    # is delayed by physics regardless of policy, so anchoring at onset would
    # fold the disturbance DURATION into the metric.
    #   incident arm -> end of the restriction window
    #   bunching arm -> release time of the LAST bunched train
    #   >=2 must have been released together to count as a bunch (inject_bunched_delay
    #   releases the whole window to one new_dep, so a genuine bunch is >=2).
    if incident_on or closure_on:
        ends = [v[1] for v in incident.values()]
        ends += [v[1] for v in closures.values()]
        t_rec_start = max(ends) if ends else None
    else:
        t_rec_start = max(released) if len(released) >= 2 else None
    g = 0; o = 0; arr_events = []; status = "?"
    ep_overtakes = ep_yields = ep_prio_ot = ep_pass_ot = ep_dy = 0
    ep_closure_ot = 0
    ep_closure_policy_ot = 0
    ep_closure_hold_sec = 0.0
    mask_stats = defaultdict(int)
    for _ in range(600):
        masks = info.get("action_masks", {})
        if mode == "flat" and flat_actor is not None:
            # flat MAPPO: joint 72-action (platform x entry x speed), the speed
            # applied as a single per-transit mode via env._flat_speed_choices.
            ad, flat_speeds = {}, {}
            for tid in info.get("active_train_ids", info.get("pending", [])):
                m = masks.get(tid)
                if not m or not any(m[0]) or not any(m[1]):
                    continue
                obs_t = torch.tensor(obs[tid], dtype=torch.float32).unsqueeze(0).to(dev)
                flat_mask = build_flat_mask(m[0], m[1]).to(dev)
                with torch.no_grad():
                    act = flat_actor.dist(obs_t, flat_mask.unsqueeze(0)).sample().item()
                p, e, s = unpack_flat_action(act)
                ad[tid] = (p, e)
                flat_speeds[tid] = s
            env._flat_speed_choices = flat_speeds
        elif mode in ("fcfs", "heur-sa"):
            # rule-based L2, no learned policy.  An EMPTY action_dict makes the
            # env fall back to heuristic_action for every policy-active train.
            # `fcfs` uses CRUISE L3; `heur-sa` uses SA L3 — the clean L2
            # comparison against `sa` (same L3, learned vs heuristic L2).
            ad = {}
        else:
            g0 = build_conflict_graph(info.get("requests", []))
            go = {aid: critic_input(aid, obs, g0, k_max=1) for aid in obs}
            gs = compact_global_state(info.get("metrics", {}))
            aa = tr.collect_step(obs, masks, go, gs, device=dev)
            ad = {aid: (a.platform, a.entry) for aid, a in aa.items()}
            if relevant_l2:
                allowed = set(info.get("control_relevant_ids", []))
                ad = {aid: act for aid, act in ad.items() if aid in allowed}
        obs, rew, term, trunc, info = env.step(ad, lambda_team=0.3)
        # mask_stats is CUMULATIVE within the episode (env resets it in reset()),
        # so the last step's dict IS the episode total — replace, do not sum.
        if info.get("mask_stats"):
            mask_stats = dict(info["mask_stats"])
        m = info.get("metrics", {})
        g += m.get("n_arrivals", 0); o += m.get("n_ontime_arr", 0)
        ep_overtakes += m.get("overtake_count", 0)
        ep_yields += m.get("yield_count", 0)
        ep_prio_ot += m.get("priority_overtake_count", 0)
        ep_pass_ot += m.get("pass_overtake_count", 0)
        ep_dy += m.get("deliberate_yield_count", 0)
        ep_closure_ot += sum(
            1 for e in info.get("overtake_events", [])
            if e.get("closure_related"))
        ep_closure_policy_ot += sum(
            1 for e in info.get("overtake_events", [])
            if e.get("closure_related") and e.get("order_policy_driven"))
        ep_closure_hold_sec += sum(info.get("closure_held", {}).values())
        for ev in info.get("l3_transit_events", []):
            ev.setdefault("priority", cfg.trains[ev["train_id"]].priority)
            arr_events.append(ev)
        if term or trunc:
            status = info["episode_status"]; break
    return {"incident": incident, "closure": closures,
            "bunched": bunched >= 1,
            "seed": seed, "t_rec_start": t_rec_start,
            "g": g, "o": o, "arr_events": arr_events, "status": status,
            "n_sections": len(env._l3_behavior),
            "overtakes": ep_overtakes, "yields": ep_yields,
            "deliberate_yields": ep_dy,
            "priority_overtakes": ep_prio_ot, "pass_overtakes": ep_pass_ot,
            "closure_overtakes": ep_closure_ot,
            "closure_policy_overtakes": ep_closure_policy_ot,
            "closure_hold_sec": ep_closure_hold_sec,
            "mask_stats": dict(mask_stats)}


def _tail10(vals):
    """Mean of the WORST 10% (the tail the paper's bar chart uses).

    A quantile point (p90) is noisy at this sample size; the mean of the worst
    decile is the same notion with far less sampling variance.  Count is
    max(1, round(0.1n)) so a single-transit episode still has a tail.
    """
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return float("nan")
    k = max(1, int(round(0.1 * n)))
    return sum(v[-k:]) / k


def _tail10_abs(vals):
    """Worst 10% by |delay| — the guard against early running.

    `arr_delay` is SIGNED, so a policy that runs everything early gets a small
    (or negative) tail10 while its punctuality collapses; measured on 2026-09-21,
    flat's signed tail mean was +12.9s against hier's +81.0s while flat's
    on-time rate was 37.6% vs hier's 65.0%.  Reporting both makes the
    cancellation visible instead of hiding it.
    """
    return _tail10([abs(v) for v in vals])


def _ep_stats(r):
    """Per-EPISODE stats — the iid unit for the bootstrap error bars.

    Episode-level, not block-level: a block is 60 episodes pooled, so only 3
    values exist per (mode, scale) and their std has df=2.  Requiring
    per-episode rows is what makes the CI cheap — no extra training seeds needed.
    """
    evs = r["arr_events"]
    ds = [e["arr_delay"] for e in evs]
    aff = [e["arr_delay"] for e in evs if e.get("affected")]
    aff_plan = [e["arr_delay"] for e in evs if e.get("affected_plan")]
    n = len(ds)
    pw_num = pw_den = 0.0
    for e in evs:
        w = L2_PRIORITY_WEIGHT.get(e.get("priority", 2), 1.0)
        pw_num += w * max(0.0, e["arr_delay"]); pw_den += w
    return {
        "n": n,
        "mean": (sum(ds) / n) if n else float("nan"),
        "tail10": _tail10(ds), "tail10abs": _tail10_abs(ds),
        "p90": sorted(ds)[int(n * 0.9)] if n else float("nan"),
        "max": max(ds) if n else float("nan"),
        "ontime": (100.0 * sum(1 for d in ds if abs(d) <= ONTIME_THRESHOLD_SEC) / n)
                  if n else float("nan"),
        "aff_n": len(aff), "aff_tail10": _tail10(aff), "aff_max": max(aff) if aff else float("nan"),
        # policy-independent subset (planned-window overlap) — see _ep_stats usage
        "aff_plan_n": len(aff_plan), "aff_plan_tail10": _tail10(aff_plan),
        "pw_arr_delay": pw_num / pw_den if pw_den else float("nan"),
        "overtakes": r.get("overtakes", 0),
        "yields": r.get("yields", 0),
        "deliberate_yields": r.get("deliberate_yields", 0),
        "closure_overtakes": r.get("closure_overtakes", 0),
        "closure_policy_overtakes": r.get("closure_policy_overtakes", 0),
        "closure_hold_sec": r.get("closure_hold_sec", 0.0),
    }


def _block_stats(eps):
    """SILENT per-block summary — key metrics only, no printing.

    Exists so the per-block dispersion is recoverable: `report_*` aggregates over
    ALL blocks (``all_eps``) and flattens the block structure, which destroys the
    only source of an error bar in this setup.  Modes share the same eval seeds
    per block (``base = seeds + b*1e6``), so these per-block values are PAIRED and
    their differences can carry a paired error band.
    """
    ds = [ev["arr_delay"] for r in eps for ev in r["arr_events"]]
    aff = [ev["arr_delay"] for r in eps for ev in r["arr_events"] if ev.get("affected")]
    unaff = [ev["arr_delay"] for r in eps for ev in r["arr_events"] if not ev.get("affected")]
    g = sum(r["g"] for r in eps)
    o = sum(r["o"] for r in eps)
    mean = (lambda v: sum(v) / len(v) if v else float("nan"))
    pw_num = pw_den = 0.0
    for r in eps:
        for e in r["arr_events"]:
            w = L2_PRIORITY_WEIGHT.get(e.get("priority", 2), 1.0)
            pw_num += w * max(0.0, e["arr_delay"]); pw_den += w
    return {
        "n": len(ds), "mean": mean(ds),
        "aff_n": len(aff), "aff_mean": mean(aff),
        "unaff_n": len(unaff), "unaff_mean": mean(unaff),
        "ontime": 100.0 * o / g if g else float("nan"),
        # TAIL columns (pooled within this block) — the bar-chart quantity.
        # Added 2026-09-21; consumers build fieldnames from the header, so old
        # readers (plot_conv_scalability reads ontime.csv) are unaffected.
        "tail10": _tail10(ds), "tail10abs": _tail10_abs(ds),
        "p90": sorted(ds)[int(len(ds) * 0.9)] if ds else float("nan"),
        "max": max(ds) if ds else float("nan"),
        "aff_tail10": _tail10(aff), "aff_max": max(aff) if aff else float("nan"),
        "overtakes": sum(r.get("overtakes", 0) for r in eps),
        "yields": sum(r.get("yields", 0) for r in eps),
        "pw_arr_delay": pw_num / pw_den if pw_den else float("nan"),
    }


def _dist(vals):
    """(n, mean, p50, p90, p95, max) of arrival delays."""
    vals = sorted(vals)
    n = len(vals)
    if n == 0:
        return (0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    return (n, sum(vals) / n, vals[n // 2], vals[int(n * 0.9)], vals[int(n * 0.95)], vals[-1])


def _fmt_dist(d):
    n, mean, p50, p90, p95, mx = d
    if n == 0:
        return "no events"
    return (f"n={n} mean={mean:6.1f} p50={p50:6.1f} p90={p90:6.1f} "
            f"p95={p95:6.1f} max={mx:6.1f}")


def _report_episode_level(eps):
    c = Counter(r["status"] for r in eps)
    n = len(eps)
    return (f"succ={c['success']}/{n} deadlock={c['deadlock']} "
            f"trunc={c['truncated']} (n_ep={n})")


def report_legacy(label, ep_results):
    """Origin-bunching stratification (backward-compatible with the old output).

    Returns a stats dict (for per-metric CSV export)."""
    g = sum(r["g"] for r in ep_results); o = sum(r["o"] for r in ep_results)
    ds = sorted([ev["arr_delay"] for r in ep_results for ev in r["arr_events"]])
    st = Counter(r["status"] for r in ep_results)
    if not ds:
        print(f"{label}: no transit events")
        return {"ontime": 0.0, "n": 0, "succ": st.get("success", 0),
                "deadlock": st.get("deadlock", 0), "trunc": st.get("truncated", 0)}
    n = len(ds)
    sn, sr, sm, smx = _tail_severity(ds)
    print(f"{label}: ontime={o/max(g,1)*100:.1f}% "
          f"arr_mean={sum(ds)/n:.1f} p50={ds[n//2]:.1f} p90={ds[int(n*0.9)]:.1f} "
          f"p95={ds[int(n*0.95)]:.1f} max={ds[-1]:.1f} "
          f"| severe(>{SEVERE_DELAY_SEC:.0f}s): n={sn} ({sr:.1f}%) mean={sm:.1f} max={smx:.1f} "
          f"| {_report_episode_level(ep_results)}")
    return {
        "ontime": o / max(g, 1) * 100,
        "n": n, "mean": sum(ds) / n, "p50": ds[n // 2], "p90": ds[int(n * 0.9)],
        "p95": ds[int(n * 0.95)], "max": ds[-1],
        "sev_rate": sr, "sev_mean": sm, "sev_max": smx,
        "succ": st.get("success", 0), "deadlock": st.get("deadlock", 0),
        "trunc": st.get("truncated", 0),
        "overtakes": sum(r.get("overtakes", 0) for r in ep_results),
        "yields": sum(r.get("yields", 0) for r in ep_results),
        "deliberate_yields": sum(
            r.get("deliberate_yields", 0) for r in ep_results),
        "priority_overtakes": sum(
            r.get("priority_overtakes", 0) for r in ep_results),
        "pass_overtakes": sum(
            r.get("pass_overtakes", 0) for r in ep_results),
    }


# --- Measured per-unit L3 decision costs (ms, CPU) ---------------------------
# Source: scripts/bench_l3_decision_cost.py -> output/9.20/l3_decision_cost_scaling.csv
# (15t ckpt transferred to every scale per section 6.3; 5 episodes x 10/15/20/25/30t).
# These REPLACE the earlier hardcoded estimates (3.48 ms/transit, 0.36 ms/section):
# the old SA constant was ~2x low, which made the SA/learned ratio read 2.4x where
# the measured value is ~4.4x.  The per-unit costs are ~scale-invariant (that is
# itself the finding), so a single constant per method is the right model.
MS_PER_SECTION_LEARNED = 0.4533
MS_PER_TRANSIT_SA = 7.4820
MS_PER_TRANSIT_FLAT = 0.0539


def _decision_ms(mode, n_transits, n_sections):
    """Estimated L3 decision cost (ms) for a whole eval, from MEASURED unit costs.

    None means the compute is deliberately not reported for that mode.
    """
    if mode in ("sa", "sa-nofc", "heur-sa"):
        return n_transits * MS_PER_TRANSIT_SA
    if mode == "learned":
        return n_sections * MS_PER_SECTION_LEARNED
    if mode == "flat":
        return n_transits * MS_PER_TRANSIT_FLAT
    if mode == "sa-online":
        return None          # compute not reported (user decision 2026-09-20)
    return 0.0               # cruise / fcfs: rule-based, no learned forward


def report_clean(eps, mode="learned"):
    """No-disturbance (clean timetable) arm: realised-arrival ontime stratified by
    priority, full arrival-delay distribution + tail severity, and the SAME E1
    re-solve compute accounting as the incident arm — so the clean and incident
    scalability tables share a schema (minus the incident-only affected subset)."""
    byp = defaultdict(lambda: [0, 0])
    for r in eps:
        for ev in r["arr_events"]:
            p = ev.get("priority")
            if p is None:
                continue
            byp[p][1] += 1
            byp[p][0] += int(abs(ev["arr_delay"]) <= ONTIME_THRESHOLD_SEC)
    print("  priority arrival ontime:")
    for p in (1, 2, 3):
        a = byp[p]
        s = f"{a[0] / a[1] * 100:.1f}% ({a[0]}/{a[1]})" if a[1] else "n/a"
        print(f"    P{p}: {s}")
    all_on = sum(byp[p][0] for p in (1, 2, 3))
    all_n = sum(byp[p][1] for p in (1, 2, 3))
    ds = sorted(ev["arr_delay"] for r in eps for ev in r["arr_events"])
    dall = _dist(ds)
    s_all = _tail_severity(ds)
    print(f"  arrival delay: {_fmt_dist(dall)}")
    print(f"  tail severe(>{SEVERE_DELAY_SEC:.0f}s): n={s_all[0]} ({s_all[1]:.1f}%) "
          f"mean={s_all[2]:.1f} max={s_all[3]:.1f}")
    print(f"  {_report_episode_level(eps)}")
    ot = sum(r.get("overtakes", 0) for r in eps)
    yd = sum(r.get("yields", 0) for r in eps)
    pot = sum(r.get("priority_overtakes", 0) for r in eps)
    p_ot = sum(r.get("pass_overtakes", 0) for r in eps)
    dy = sum(r.get("deliberate_yields", 0) for r in eps)
    cot = sum(r.get("closure_overtakes", 0) for r in eps)
    cpot = sum(r.get("closure_policy_overtakes", 0) for r in eps)
    cheld = sum(r.get("closure_hold_sec", 0.0) for r in eps)
    print(f"  re-sequencing: overtakes={ot} yields={yd} "
          f"deliberate_yields={dy} priority_overtakes={pot} "
          f"pass_overtakes={p_ot} closure_overtakes={cot} "
          f"(policy-driven {cpot}) closure_hold={cheld:.0f}s")
    n_transits = sum(len(r["arr_events"]) for r in eps)
    n_sections = sum(r.get("n_sections", 0) for r in eps)
    method_ms = _decision_ms(mode, n_transits, n_sections)
    sa_ms = n_transits * MS_PER_TRANSIT_SA
    lr_ms = n_sections * MS_PER_SECTION_LEARNED
    if method_ms is None:
        print(f"  E1 re-solve compute [{mode}]: transits={n_transits} "
              f"sections={n_sections} | method compute not reported")
    else:
        print(f"  E1 re-solve compute [{mode}]: transits={n_transits} sections={n_sections} | "
              f"method ~{method_ms:.0f}ms | SA re-solve ~{sa_ms:.0f}ms vs learned ~{lr_ms:.0f}ms "
              f"({sa_ms / max(lr_ms, 1e-9):.1f}x) [measured unit costs]")
    st = Counter(r["status"] for r in eps)
    return {
        "ontime": all_on / max(all_n, 1) * 100,
        "p1": byp[1][0] / max(byp[1][1], 1) * 100,
        "p2": byp[2][0] / max(byp[2][1], 1) * 100,
        "p3": byp[3][0] / max(byp[3][1], 1) * 100,
        "n": dall[0], "mean": dall[1], "p50": dall[2], "p90": dall[3],
        "p95": dall[4], "max": dall[5],
        "n_transits": n_transits, "n_sections": n_sections,
        "sa_resolve_ms": round(sa_ms, 1), "learned_ms": round(lr_ms, 1),
        # _decision_ms returns None for sa-online (compute deliberately not
        # reported, user decision 2026-09-20) — round() would crash.
        "method_ms": (round(method_ms, 1) if method_ms is not None else None),
        "resolve_ratio": round(sa_ms / max(lr_ms, 1e-9), 1),
        "sev_rate": s_all[1], "sev_mean": s_all[2], "sev_max": s_all[3],
        "succ": st.get("success", 0), "deadlock": st.get("deadlock", 0),
        "trunc": st.get("truncated", 0),
        "overtakes": ot, "yields": yd, "deliberate_yields": dy,
        "priority_overtakes": pot, "pass_overtakes": p_ot,
        "closure_overtakes": cot, "closure_policy_overtakes": cpot,
        "closure_hold_sec": round(cheld, 1),
    }


def report_incident(eps, mode="learned"):
    """Disturbance-aware: realised arrival delay, affected (transit in the
    restriction window OR held by a closure) vs unaffected, and disturbed vs
    non-disturbed episodes.  Covers both the speed-restriction and closure arms.
    """
    aff = [ev["arr_delay"] for r in eps for ev in r["arr_events"] if ev["affected"]]
    unaff = [ev["arr_delay"] for r in eps for ev in r["arr_events"] if not ev["affected"]]
    inc_eps = [r for r in eps if r.get("incident") or r.get("closure")]
    non_eps = [r for r in eps if not (r.get("incident") or r.get("closure"))]
    n_aff_ep = sum(1 for r in eps if any(ev["affected"] for ev in r["arr_events"]))
    print(f"  affected   : {_fmt_dist(_dist(aff))}  (transits in window / held)")
    print(f"  unaffected : {_fmt_dist(_dist(unaff))}")
    print(f"  disturbed episodes: {len(inc_eps)}/{len(eps)}, "
          f"with >=1 affected transit: {n_aff_ep}")
    print(f"  -- episodes WITH disturbance ({len(inc_eps)}) --")
    print(f"     {_report_episode_level(inc_eps)}")
    if non_eps:
        print(f"  -- episodes WITHOUT disturbance ({len(non_eps)}) --")
        print(f"     {_report_episode_level(non_eps)}")
    # priority-stratified realised-arrival ontime (railway KPI by class)
    byp_all = defaultdict(lambda: [0, 0])
    byp_aff = defaultdict(lambda: [0, 0])
    for r in eps:
        for ev in r["arr_events"]:
            p = ev.get("priority")
            if p is None:
                continue
            on = abs(ev["arr_delay"]) <= ONTIME_THRESHOLD_SEC
            byp_all[p][1] += 1
            byp_all[p][0] += int(on)
            if ev.get("affected"):
                byp_aff[p][1] += 1
                byp_aff[p][0] += int(on)
    print("  priority arrival ontime (all / affected):")
    for p in (1, 2, 3):
        a = byp_all[p]
        f = byp_aff[p]
        a_s = f"{a[0]/max(a[1],1)*100:.1f}% ({a[0]}/{a[1]})" if a[1] else "n/a"
        f_s = f"{f[0]/max(f[1],1)*100:.1f}% ({f[0]}/{f[1]})" if f[1] else "n/a"
        print(f"    P{p}: all={a_s}  affected={f_s}")
    all_on = sum(byp_all[p][0] for p in (1, 2, 3))
    all_n = sum(byp_all[p][1] for p in (1, 2, 3))
    da = _dist(aff)
    du = _dist(unaff)
    dall = _dist([ev["arr_delay"] for r in eps for ev in r["arr_events"]])
    st = Counter(r["status"] for r in eps)
    # TAIL severity (尾端恢复): how severe the worst-delayed trains are.
    s_all = _tail_severity([ev["arr_delay"] for r in eps for ev in r["arr_events"]])
    s_aff = _tail_severity(aff)
    print(f"  tail severe(>{SEVERE_DELAY_SEC:.0f}s): all n={s_all[0]} ({s_all[1]:.1f}%) "
          f"mean={s_all[2]:.1f} max={s_all[3]:.1f} | affected n={s_aff[0]} "
          f"mean={s_aff[2]:.1f} max={s_aff[3]:.1f}")
    ot = sum(r.get("overtakes", 0) for r in eps)
    yd = sum(r.get("yields", 0) for r in eps)
    pot = sum(r.get("priority_overtakes", 0) for r in eps)
    p_ot = sum(r.get("pass_overtakes", 0) for r in eps)
    dy = sum(r.get("deliberate_yields", 0) for r in eps)
    cot = sum(r.get("closure_overtakes", 0) for r in eps)
    cpot = sum(r.get("closure_policy_overtakes", 0) for r in eps)
    cheld = sum(r.get("closure_hold_sec", 0.0) for r in eps)
    print(f"  re-sequencing: overtakes={ot} yields={yd} "
          f"deliberate_yields={dy} priority_overtakes={pot} "
          f"pass_overtakes={p_ot} closure_overtakes={cot} "
          f"(policy-driven {cpot}) closure_hold={cheld:.0f}s")
    # E1 re-solve compute (per MODE), from MEASURED unit costs (see _decision_ms):
    # SA re-solves once per corridor transit (~7.5 ms, the 300-iteration anneal);
    # learned reacts per section (~0.45 ms, one fixed-dim forward); flat spends
    # ~0.054 ms/transit.  "SA 重排代价 vs learned 即时响应成本".
    n_transits = sum(len(r["arr_events"]) for r in eps)
    n_sections = sum(r.get("n_sections", 0) for r in eps)
    method_ms = _decision_ms(mode, n_transits, n_sections)
    sa_ms = n_transits * MS_PER_TRANSIT_SA
    lr_ms = n_sections * MS_PER_SECTION_LEARNED
    if method_ms is None:
        print(f"  E1 re-solve compute [{mode}]: transits={n_transits} "
              f"sections={n_sections} | method compute not reported")
    else:
        print(f"  E1 re-solve compute [{mode}]: transits={n_transits} sections={n_sections} | "
              f"method ~{method_ms:.0f}ms | SA 重排 ~{sa_ms:.0f}ms vs learned 即时 ~{lr_ms:.0f}ms "
              f"({sa_ms/max(lr_ms,1e-9):.1f}x) [measured unit costs]")
    return {
        "ontime": all_on / max(all_n, 1) * 100,
        "p1": byp_all[1][0] / max(byp_all[1][1], 1) * 100,
        "p2": byp_all[2][0] / max(byp_all[2][1], 1) * 100,
        "p3": byp_all[3][0] / max(byp_all[3][1], 1) * 100,
        "n": dall[0], "mean": dall[1], "p50": dall[2], "p90": dall[3],
        "p95": dall[4], "max": dall[5],
        "n_transits": n_transits, "n_sections": n_sections,
        "sa_resolve_ms": round(sa_ms, 1), "learned_ms": round(lr_ms, 1),
        # _decision_ms returns None for sa-online (compute deliberately not
        # reported, user decision 2026-09-20) — round() would crash.
        "method_ms": (round(method_ms, 1) if method_ms is not None else None),
        "resolve_ratio": round(sa_ms / max(lr_ms, 1e-9), 1),
        "aff_n": da[0], "aff_mean": da[1], "aff_p90": da[3], "aff_max": da[5],
        "unaff_n": du[0], "unaff_mean": du[1], "unaff_max": du[5],
        "sev_rate": s_all[1], "sev_mean": s_all[2], "sev_max": s_all[3],
        "aff_sev_mean": s_aff[2], "aff_sev_max": s_aff[3],
        "succ": st.get("success", 0), "deadlock": st.get("deadlock", 0),
        "trunc": st.get("truncated", 0),
        "inc_eps": len(inc_eps), "aff_eps": n_aff_ep,
        "overtakes": ot, "yields": yd, "deliberate_yields": dy,
        "priority_overtakes": pot, "pass_overtakes": p_ot,
        "closure_overtakes": cot, "closure_policy_overtakes": cpot,
        "closure_hold_sec": round(cheld, 1),
    }


def report_recovery(eps, mode="learned", floor_eps=None):
    """Delay-dissipation time (DDT): how fast the system comes BACK, not how bad
    it got.  Full definition + the rationale for each design choice lives in
    src/experiment/recovery_metrics.py — read that docstring before changing the
    threshold, the anchor, or the sign convention.

    Pre-registered cell: theta=30 sim-s, K=3 consecutive arrivals within theta.
    A time-to-event statistic is heavy-tailed and right-censored, so this reports
    median + IQR + the censoring rate and NEVER a mean.  Because theta and K are
    chosen rather than derived, the grid below is printed too: the paper claim is
    RANK STABILITY of the mode ordering across the grid, not one absolute cell.

    `floor_eps` (incident arm, = CRUISE) switches to the PAIRED excess over a
    reference execution layer, because the restriction adds delay that L3 cannot
    fully recover — measured absolutely, the incident arm is censored by
    construction and reports nothing about coordination.

    Returns ``(stats, grid_rows)`` for the per-metric CSVs — ``({}, [])`` when
    nothing was disturbed (clean arm).
    """
    label = "excess vs CRUISE" if floor_eps is not None else "absolute"
    coverage = float("nan")
    if floor_eps is not None:
        # The reference is the SAME run, so pair by (seed, train_id, corridor).
        eps, coverage = with_floor(eps, floor_index(floor_eps))
    value_of = (lambda e: e["_excess"]) if floor_eps is not None else None
    st = recovery_stats(eps, DDT_THETA_SEC, DDT_K, value_of=value_of)
    if st["n_eps_dist"] == 0:
        print("  delay dissipation: no disturbed episodes — skipped")
        return {}, []
    cov = "" if floor_eps is None else f" paired={coverage:.1f}%"
    print(f"  delay dissipation [{label}{cov}] (theta={DDT_THETA_SEC:.0f}s, "
          f"K={DDT_K} consecutive arrivals within theta):")
    print(f"    DDT median={st['med']:.0f}s p25={st['p25']:.0f} p75={st['p75']:.0f} "
          f"| recovered {st['n_rec']}/{st['n_eps_dist']} "
          f"(censored {st['censor_pct']:.1f}%, of which structurally short "
          f"{st['short_pct']:.1f}%)")
    rows = recovery_grid(eps, value_of=value_of)
    print("    sensitivity (median s, censored%):")
    print("           " + "".join(f"{f'K={k}':>16}" for k in sorted({r['k'] for r in rows})))
    for theta in sorted({r["theta"] for r in rows}):
        cells = ""
        for r in sorted((x for x in rows if x["theta"] == theta), key=lambda x: x["k"]):
            cells += (f"{r['med']:>11.0f},{r['censor_pct']:>4.0f}"
                      if r["med"] == r["med"] else f"{'n/a':>16}")
        print(f"    th={theta:>2.0f}  {cells}")
    out = {"n_eps_dist": st["n_eps_dist"], "n_rec": st["n_rec"],
           "ddt_med": st["med"], "ddt_p25": st["p25"], "ddt_p75": st["p75"],
           "ddt_censor_pct": st["censor_pct"], "ddt_short_pct": st["short_pct"],
           "ddt_pair_pct": coverage}
    out.update({f"grid_t{int(r['theta'])}_k{r['k']}": r["med"] for r in rows})
    return out, rows


def per_seed_affected(eps, key):
    """Per-episode affected arrival-delay aggregate (None if no affected transit)."""
    out = []
    for r in eps:
        aff = sorted(ev["arr_delay"] for ev in r["arr_events"] if ev["affected"])
        if not aff:
            out.append(None)
        elif key == "mean":
            out.append(sum(aff) / len(aff))
        elif key == "p90":
            out.append(aff[int(len(aff) * 0.9)])
        else:  # max
            out.append(aff[-1])
    return out


def paired_excess(base_eps, ref_eps, key):
    """mean ± std of (base - ref) per episode over episodes where BOTH have
    affected transits — the coordination value relative to the reference
    baseline (e.g. learned vs CRUISE) on the incident-affected subset."""
    a = per_seed_affected(base_eps, key)
    b = per_seed_affected(ref_eps, key)
    pairs = [(x - y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = sum(pairs) / n
    var = sum((p - mean) ** 2 for p in pairs) / max(n - 1, 1)
    return mean, var ** 0.5, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="five_station_15t")
    ap.add_argument("--l2", required=True)
    ap.add_argument("--l3", default=None)
    ap.add_argument("--flat-ckpt", type=Path, default=None,
                    help="flat MAPPO checkpoint (actor) for the 'flat' mode — the "
                         "joint platformxentryxspeed baseline (adds flat to the "
                         "default mode set)")
    ap.add_argument("--flat-only", action="store_true",
                    help="only the flat MAPPO baseline")
    ap.add_argument("--sa-only", action="store_true")
    ap.add_argument("--heur-sa-only", action="store_true",
                    help="heuristic L2 + SA L3 (clean L2 comparison vs --sa-only)")
    ap.add_argument("--relevant-l2", action="store_true",
                    help="apply relevance gating: policy acts only on "
                         "higher-priority-follower / pass-opportunity decisions")
    ap.add_argument("--relevant-plus-l2", action="store_true",
                    help="wider relevance gating: base + closure + any "
                         "co-pending same-direction partner (~85%% coverage)")
    ap.add_argument("--yield-cap", action="store_true",
                    help="cap yield entry choices to plan/+10/+20 (match training)")
    ap.add_argument("--sa-online-only", action="store_true",
                    help="partial-observation SA baseline (per-section greedy, "
                         "current state only — the fair match for the learned worker)")
    ap.add_argument("--sa-nofc-only", action="store_true",
                    help="SA with NO incident forecast: plans as if clear but "
                         "executes the real restriction (the cost of needing a "
                         "forecast vs MAPPO's reactive adaptation)")
    ap.add_argument("--cruise-only", action="store_true")
    ap.add_argument("--fcfs-only", action="store_true",
                    help="only the rule-based FCFS mode (non-learning L2: "
                         "heuristic_action / planned-departure order + rule CRUISE L3)")
    ap.add_argument("--learned-only", action="store_true",
                    help="only the learned mode (SA/CRUISE are checkpoint-independent)")
    ap.add_argument("--no-delay", action="store_true",
                    help="no-disturbance arm: disable origin bunching AND incident "
                         "(completely clean timetable)")
    ap.add_argument("--extra-train", action="store_true",
                    help="random mid-operation extra P2 train (spawns at a random "
                         "station, transits to D, stops at D) — tests new-instance "
                         "adaptation without retraining")
    ap.add_argument("--severity", type=float, default=1.0,
                    help="incident strength multiplier (1.0 = base config; >1 e.g. 2.0 "
                         "opens the L2 re-dispatch window by pushing affected delays "
                         "past dep_max_slack) — WITHOUT editing the base config")
    ap.add_argument("--seeds", type=int, default=800000)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--incident", action="store_true",
                    help="corridor speed-restriction incident as the ONLY disturbance "
                         "(disables origin-bunching); incident-aware stratification")
    ap.add_argument("--closure", action="store_true",
                    help="corridor closure (区间中断): no new entries during a "
                         "window designed to hold a lower-priority leader behind a "
                         "higher-priority follower (disables origin-bunching)")
    ap.add_argument("--vt", action="store_true", help="enable visitor trains")
    ap.add_argument("--blocks", type=int, default=3,
                    help="independent seed blocks (seed sets are {seeds+b*1e6..}); "
                         "variance across blocks addresses seed-set sensitivity")
    ap.add_argument("--sa-iters", type=int, default=300,
                    help="SA annealing iterations per transit (default 300, the paper "
                         "value; lower values yield a WEAKER local-experiment baseline)")
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="write per-mode summary (ontime, priority, affected delay) to CSV")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write ONE CSV per metric (ontime/delay/affected/tail/episode) "
                         "into this directory")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_scenario(scenario_name=args.scenario)
    tr = _load_l2(cfg, args.l2, dev)
    l3 = _load_l3(args.l3, dev) if (args.l3 and not args.sa_only) else None
    flat_actor = None
    if args.flat_ckpt:
        fk = torch.load(args.flat_ckpt, map_location=dev, weights_only=False)
        flat_actor = FlatJointActor(obs_dim=get_obs_dim(cfg), hidden_dim=128)
        flat_actor.load_state_dict(fk["actor"])
        flat_actor.to(dev).eval()
        for p in flat_actor.parameters():
            p.requires_grad = False

    modes = []
    if args.cruise_only:
        modes = ["cruise"]
    elif args.fcfs_only:
        modes = ["fcfs"]
    elif args.sa_only:
        modes = ["sa"]
    elif args.heur_sa_only:
        modes = ["heur-sa"]
    elif args.sa_online_only:
        modes = ["sa-online"]
    elif args.sa_nofc_only:
        modes = ["sa-nofc"]
    elif args.learned_only:
        modes = ["learned"]
    elif args.flat_only:
        modes = ["flat"]
    else:
        modes = ["learned", "sa", "sa-online", "sa-nofc", "fcfs", "cruise"]
        if args.flat_ckpt:
            modes.append("flat")

    # modes -> list of seed-block -> list of episode results (same seeds per mode)
    per_mode = {}
    for mode in modes:
        blocks = []
        for b in range(args.blocks):
            base = args.seeds + b * 1_000_000
            eps = [run_episode(cfg, tr, l3, s, mode, dev, args.incident, args.vt,
                               args.no_delay, args.extra_train, args.severity,
                               flat_actor, sa_iters=args.sa_iters,
                               relevant_l2=(args.relevant_l2
                                            or args.relevant_plus_l2),
                               yield_cap=args.yield_cap,
                               closure_on=args.closure,
                               coverage=("plus" if args.relevant_plus_l2
                                         else "base"))
                   for s in range(base, base + args.n)]
            blocks.append(eps)
        per_mode[mode] = blocks

    # ---- per-mode reports ----
    stats_by_mode = {}
    for mode in modes:
        all_eps = [r for b in per_mode[mode] for r in b]
        print(f"\n== {mode} (held-out {args.blocks}x{args.n} seeds, start {args.seeds}) ==")
        if args.incident or args.closure:
            stats_by_mode[mode] = report_incident(all_eps, mode)
        elif args.no_delay:
            # completely clean arm — no origin bunching to stratify on
            stats_by_mode[mode] = report_clean(all_eps, mode)
        else:
            by_bunch = {"bunched": [], "non": []}
            for r in all_eps:
                by_bunch["bunched" if r["bunched"] else "non"].append(r)
            report_legacy("  combined       ", by_bunch["bunched"] + by_bunch["non"])
            report_legacy(f"  bunched ({len(by_bunch['bunched'])})", by_bunch["bunched"])
            report_legacy(f"  non-bunched ({len(by_bunch['non'])})", by_bunch["non"])
            stats_by_mode[mode] = report_legacy(
                "  combined       ", by_bunch["bunched"] + by_bunch["non"])

    # ---- delay dissipation (DDT): how FAST the disturbance clears ----
    # The third leg of the metric set: tail severity = how bad, mean/on-time =
    # how good on average, DDT = how quickly back to normal.  Skipped in the
    # clean arm — no disturbance means no anchor and nothing to recover from.
    ddt_by_mode = {}
    ddt_grid_rows = []
    if not args.no_delay and modes:
        ref = ([r for b in per_mode["cruise"] for r in b]
               if "cruise" in per_mode else None)
        print("\n== delay dissipation (recovery time, pre-registered theta=30s K=3) ==")
        for mode in modes:
            all_eps = [r for b in per_mode[mode] for r in b]
            # Incident arm: the restriction's delay is not fully recoverable, so
            # measure the PAIRED excess over CRUISE (cruise itself has no floor).
            flo = ref if ((args.incident or args.closure) and ref is not None
                          and mode != "cruise") else None
            print(f"-- {mode} --")
            st, rows = report_recovery(all_eps, mode, floor_eps=flo)
            ddt_by_mode[mode] = st
            for r in rows:
                ddt_grid_rows.append({"mode": mode, **r})
        # Rank stability, not absolute values: theta/K are chosen, so what the
        # paper claims is that this ordering survives the whole grid.
        # n_rec > 0 guarantees a real median (recovery_stats leaves it NaN otherwise).
        # In the incident arm CRUISE is the floor, so its own DDT is on the
        # ABSOLUTE scale and is not comparable — exclude it from the ranking.
        ok = [m for m in modes if ddt_by_mode.get(m, {}).get("n_rec", 0) > 0
              and not ((args.incident or args.closure) and m == "cruise")]
        if len(ok) > 1:
            ranked = sorted(ok, key=lambda m: ddt_by_mode[m]["ddt_med"])
            print("  pre-registered cell ranking (fastest recovery first): " +
                  " < ".join(f"{m}({ddt_by_mode[m]['ddt_med']:.0f}s)" for m in ranked))

    # ---- Layer-1 mask diagnostics: how often the mask actually rejects ----
    mask_by_mode = {}
    for mode in modes:
        agg = Counter()
        for b in per_mode[mode]:
            for r in b:
                agg.update(r.get("mask_stats", {}))
        if not agg:
            continue
        mask_by_mode[mode] = agg
        print(f"  mask [{mode}]: decisions={agg['decisions']} "
              f"candidates={agg['candidates']} masked_out={agg['masked_out']} "
              f"({agg['masked_out'] / max(agg['candidates'], 1) * 100:.1f}%) "
              f"decisions_with_mask={agg['decisions_with_mask']} "
              f"({agg['decisions_with_mask'] / max(agg['decisions'], 1) * 100:.1f}%) "
              f"no_legal_action={agg['no_legal_action']} "
              f"postcheck_reject={agg['postcheck_reject']}")
        print("        by bound: " + (", ".join(
            f"{k}={agg[k]}" for k in MASK_REASONS if agg[k]) or "none"))

    if args.out_dir:
        import csv as _csv
        args.out_dir.mkdir(parents=True, exist_ok=True)

        def write_metric(name, fields):
            with open(args.out_dir / name, "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=["mode"] + fields)
                w.writeheader()
                for mode in modes:
                    row = {"mode": mode}
                    for k in fields:
                        v = stats_by_mode[mode].get(k)
                        row[k] = round(v, 3) if isinstance(v, (int, float)) else v
                    w.writerow(row)

        # Per-block rows (NEW file — never change the schema of the existing
        # per-metric CSVs: plot_conv_scalability.py:144 reads ontime.csv as
        # mode,ontime).  These give the only available error bar: n = blocks,
        # paired across modes because every mode uses the same seeds per block.
        with open(args.out_dir / "per_block.csv", "w", newline="") as f:
            fields = ["mode", "block", "n", "mean", "aff_n", "aff_mean",
                      "unaff_n", "unaff_mean", "ontime",
                      "tail10", "tail10abs", "p90", "max",
                      "aff_tail10", "aff_max", "pw_arr_delay",
                      "overtakes", "yields"]
            w = _csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for mode in modes:
                for b, block_eps in enumerate(per_mode[mode]):
                    st = _block_stats(block_eps)
                    w.writerow({"mode": mode, "block": b,
                                **{k: (round(v, 3) if isinstance(v, float) else v)
                                   for k, v in st.items()}})

        # PER-EPISODE rows — the iid unit for the tail-error bars.  Written here
        # (not derived later) because the eval loop is the only place that sees
        # each episode's transit delays; the aggregates throw them away.
        with open(args.out_dir / "per_episode.csv", "w", newline="") as f:
            fields = ["mode", "block", "seed", "n", "mean", "tail10", "tail10abs",
                      "p90", "max", "ontime", "aff_n", "aff_tail10", "aff_max",
                      "aff_plan_n", "aff_plan_tail10", "pw_arr_delay",
                      "overtakes", "yields", "deliberate_yields",
                      "priority_overtakes", "pass_overtakes",
                      "closure_overtakes", "closure_policy_overtakes",
                      "closure_hold_sec", "status"]
            w = _csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for mode in modes:
                for b, block_eps in enumerate(per_mode[mode]):
                    for r in block_eps:
                        st = _ep_stats(r)
                        w.writerow({"mode": mode, "block": b, "seed": r.get("seed"),
                                    "status": r.get("status", "?"),
                                    "priority_overtakes": r.get("priority_overtakes", 0),
                                    "pass_overtakes": r.get("pass_overtakes", 0),
                                    **{k: (round(v, 3) if isinstance(v, float) else v)
                                       for k, v in st.items()}})

        # PER-TRANSIT rows — the raw delays.  Needed because the paper's tail
        # quantity is the mean of the worst 10% of the POOLED delay distribution,
        # and a pooled top-decile can only be RECOMPUTED under episode
        # resampling if the raw values survive.  Verified 2026-09-21: the
        # per-episode-then-average variant is nearly policy-invariant (each
        # 36-transit episode's worst 4 always contain the incident-pinned
        # transits, which are physically unrecoverable), so it must NOT be the
        # figure's quantity — it flattens every method to the same bar.
        with open(args.out_dir / "per_transit.csv", "w", newline="") as f:
            fields = ["mode", "block", "seed", "train_id", "corridor", "arr_delay",
                      "affected", "affected_plan", "priority", "front_train",
                      "dep_station", "arr_station", "is_pass_dep",
                      "headway_delay"]
            w = _csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for mode in modes:
                for b, block_eps in enumerate(per_mode[mode]):
                    for r in block_eps:
                        for e in r["arr_events"]:
                            w.writerow({"mode": mode, "block": b, "seed": r.get("seed"),
                                        "train_id": e["train_id"], "corridor": e["corridor"],
                                        "arr_delay": round(e["arr_delay"], 3),
                                        "affected": int(bool(e.get("affected"))),
                                        "affected_plan": int(bool(e.get("affected_plan"))),
                                        "priority": e.get("priority"),
                                        "front_train": e.get("front_train"),
                                        "dep_station": e.get("dep_station"),
                                        "arr_station": e.get("arr_station"),
                                        "is_pass_dep": int(bool(e.get("is_pass_dep"))),
                                        "headway_delay": round(
                                            e.get("headway_delay", 0.0), 3)})

        # Overtake/yield summary (policy-independent, env-side counters).
        with open(args.out_dir / "overtake.csv", "w", newline="") as f:
            fields = ["mode", "block", "seed", "overtakes", "yields",
                      "deliberate_yields", "priority_overtakes",
                      "pass_overtakes", "closure_overtakes",
                      "closure_policy_overtakes", "closure_hold_sec"]
            w = _csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for mode in modes:
                for b, block_eps in enumerate(per_mode[mode]):
                    for r in block_eps:
                        w.writerow({
                            "mode": mode, "block": b, "seed": r.get("seed"),
                            "overtakes": r.get("overtakes", 0),
                            "yields": r.get("yields", 0),
                            "deliberate_yields": r.get("deliberate_yields", 0),
                            "priority_overtakes": r.get("priority_overtakes", 0),
                            "pass_overtakes": r.get("pass_overtakes", 0),
                            "closure_overtakes": r.get("closure_overtakes", 0),
                            "closure_policy_overtakes": r.get(
                                "closure_policy_overtakes", 0),
                            "closure_hold_sec": round(
                                r.get("closure_hold_sec", 0.0), 1)})

        if args.incident or args.closure:
            write_metric("ontime.csv", ["ontime", "p1", "p2", "p3"])
            write_metric("delay.csv", ["n", "mean", "p50", "p90", "p95", "max"])
            write_metric("affected.csv", ["aff_n", "aff_mean", "aff_p90", "aff_max",
                                          "unaff_n", "unaff_mean", "unaff_max"])
            write_metric("tail.csv", ["sev_rate", "sev_mean", "sev_max",
                                      "aff_sev_mean", "aff_sev_max"])
            write_metric("compute.csv", ["n_transits", "n_sections", "method_ms",
                                         "sa_resolve_ms", "learned_ms", "resolve_ratio"])
            write_metric("episode.csv", ["succ", "deadlock", "trunc", "inc_eps", "aff_eps"])
        elif args.no_delay:
            # clean arm: same schema as the incident arm, minus the
            # incident-only affected subset / incident-episode counters
            write_metric("ontime.csv", ["ontime", "p1", "p2", "p3"])
            write_metric("delay.csv", ["n", "mean", "p50", "p90", "p95", "max"])
            write_metric("tail.csv", ["sev_rate", "sev_mean", "sev_max"])
            write_metric("compute.csv", ["n_transits", "n_sections", "method_ms",
                                         "sa_resolve_ms", "learned_ms", "resolve_ratio"])
            write_metric("episode.csv", ["succ", "deadlock", "trunc"])
        else:
            write_metric("ontime.csv", ["ontime"])
            write_metric("delay.csv", ["n", "mean", "p50", "p90", "p95", "max"])
            write_metric("tail.csv", ["sev_rate", "sev_mean", "sev_max"])
            write_metric("episode.csv", ["succ", "deadlock", "trunc"])

        # DDT lives in ddt_by_mode, NOT stats_by_mode: keeping it separate leaves
        # every existing per-metric schema (and --out-csv's fixed fieldname list)
        # exactly as it was.  Absent in the clean arm (no disturbance -> no
        # anchor -> nothing to recover from).
        if ddt_by_mode:
            fields = ["n_eps_dist", "n_rec", "ddt_med", "ddt_p25", "ddt_p75",
                      "ddt_censor_pct", "ddt_short_pct"]
            if args.incident or args.closure:
                fields.append("ddt_pair_pct")
            with open(args.out_dir / "recovery.csv", "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=["mode"] + fields)
                w.writeheader()
                for mode in modes:
                    st = ddt_by_mode.get(mode, {})
                    w.writerow({"mode": mode, **{
                        k: (round(st[k], 3) if isinstance(st.get(k), (int, float)) else None)
                        for k in fields}})

        # DDT sensitivity grid (long format) — theta and K are chosen, so the
        # claim is rank stability ACROSS the grid, which needs every cell.
        if ddt_grid_rows:
            with open(args.out_dir / "recovery_grid.csv", "w", newline="") as f:
                fields = ["mode", "theta", "k", "n_eps_dist", "n_rec",
                          "censor_pct", "short_pct", "med", "p25", "p75"]
                w = _csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in ddt_grid_rows:
                    w.writerow({k: (round(r[k], 3) if isinstance(r.get(k), float) else r.get(k))
                                for k in fields})

        if mask_by_mode:
            fields = ["decisions", "decisions_with_mask", "dw_mask_pct",
                      "candidates", "masked_out", "masked_pct",
                      "no_legal_action", "postcheck_reject"] + MASK_REASONS
            with open(args.out_dir / "mask.csv", "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=["mode"] + fields)
                w.writeheader()
                for mode in modes:
                    a = mask_by_mode.get(mode)
                    if not a:
                        continue
                    row = {"mode": mode,
                           "masked_pct": round(
                               a["masked_out"] / max(a["candidates"], 1) * 100, 3),
                           "dw_mask_pct": round(
                               a["decisions_with_mask"] / max(a["decisions"], 1) * 100, 3)}
                    for k in fields:
                        if k not in ("masked_pct", "dw_mask_pct"):
                            row[k] = a.get(k, 0)
                    w.writerow(row)
        print(f"saved per-metric CSVs to {args.out_dir}")

    if args.out_csv and stats_by_mode:
        import csv as _csv
        with open(args.out_csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=["mode", "ontime", "p1", "p2", "p3",
                                               "aff_mean", "aff_p90", "aff_n",
                                               "unaff_mean", "unaff_n",
                                               "sev_rate", "sev_mean", "aff_sev_mean"])
            w.writeheader()
            for mode in modes:
                row = {"mode": mode, **{k: round(v, 3)
                                        for k, v in stats_by_mode[mode].items()}}
                row = {k: v for k, v in row.items() if k in w.fieldnames}
                w.writerow(row)
        print(f"saved {args.out_csv}")

    # ---- incident arm: paired excess vs CRUISE + seed-block variance ----
    if (args.incident or args.closure) and "cruise" in per_mode:
        ref = [r for b in per_mode["cruise"] for r in b]
        print("\n== disturbance arm: PAIRED excess vs CRUISE on affected subset ==")
        print("  (arrival-delay delta; negative = baseline is WORSE than CRUISE "
              "= coordination helps)  mean ± std (over episodes where both have affected)")
        for mode in ["learned", "sa", "sa-online", "sa-nofc"]:
            if mode not in per_mode:
                continue
            base = [r for b in per_mode[mode] for r in b]
            for key in ["p90", "max"]:
                m, sd, n = paired_excess(base, ref, key)
                print(f"  {mode} - cruise ({key:>3}): {m:7.1f} ± {sd:5.1f}s  (n_paired={n})")
        # seed-block variance of the headline (learned affected p90 minus cruise)
        if "learned" in per_mode:
            print("  per-seed-block affected p90 excess (learned - cruise):")
            for b, (leps, ceps) in enumerate(zip(per_mode["learned"], per_mode["cruise"])):
                m, _, n = paired_excess(leps, ceps, "p90")
                print(f"    block {b} (seeds {args.seeds+b*1_000_000}..): "
                      f"{m:7.1f}s  (n_paired={n})")


if __name__ == "__main__":
    main()
