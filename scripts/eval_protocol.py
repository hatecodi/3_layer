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
from src.env.reward import ONTIME_THRESHOLD_SEC

# 严重晚点 threshold: > 60 sim-s (~25 real min) — the TAIL metric.  Captures
# "尾端恢复": how SEVERE the worst-delayed trains are, i.e. whether the system
# bounds the long tail or leaves some trains badly late.
SEVERE_DELAY_SEC = 60.0


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


def _load_l2(cfg, l2_ckpt, device):
    obs_dim = get_obs_dim(cfg)
    ckpt = torch.load(l2_ckpt, map_location=device, weights_only=False)
    ac = Layer2ActorCritic(obs_dim=obs_dim, k_local=1, hidden_dim=128)
    ac.load_state_dict(ckpt["ac_state"]); ac.eval()
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
    c_local.load_state_dict(ckpt["l3_critic_local"]); c_local.eval()
    c_global = CorridorCritic(L3_OBS_DIM, hidden_dim=128)
    c_global.load_state_dict(ckpt["l3_critic_global"]); c_global.eval()
    for p in list(a.parameters()) + list(c_local.parameters()) + list(c_global.parameters()):
        p.requires_grad = False
    return L3WorkerTrainer(a, c_local, c_global, lr=1e-3, device=device)


def run_episode(cfg, tr, l3, seed, mode, dev, incident_on, vt, no_delay=False,
                extra_train=False, severity=1.0, flat_actor=None):
    env = FiveStationEnv(cfg=cfg, seed=seed, enable_vt=vt,
                         enable_initial_delay=(not incident_on and not no_delay),
                         enable_corridor_incident=incident_on,
                         l3_worker=(l3 if mode == "learned" else None),
                         l3_sa=(mode == "sa"),
                         l3_sa_online=(mode == "sa-online"),
                         l3_sa_nofc=(mode == "sa-nofc"),
                         enable_extra_train=extra_train,
                         incident_scale=severity)
    obs, info = env.reset()
    incident = dict(env._corridor_incidents) if incident_on else {}
    # legacy bunching classification (only meaningful without --incident)
    bunched = 0
    for tid, j in env._journeys.items():
        t = cfg.trains[tid]
        if j.current_station == t.origin_station:
            p = t.planned.get(t.origin_station)
            if p and j.arrival_at_current > p.dep + 1:
                bunched += 1
    g = 0; o = 0; arr_events = []; status = "?"
    for _ in range(600):
        masks = info.get("action_masks", {})
        if mode == "flat" and flat_actor is not None:
            # flat MAPPO: joint 72-action (platform x entry x speed), the speed
            # applied as a single per-transit mode via env._flat_speed_choices.
            ad, flat_speeds = {}, {}
            for tid in info.get("pending", []):
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
        else:
            g0 = build_conflict_graph(info.get("requests", []))
            go = {aid: critic_input(aid, obs, g0, k_max=1) for aid in obs}
            gs = compact_global_state(info.get("metrics", {}))
            aa = tr.collect_step(obs, masks, go, gs, device=dev)
            ad = {aid: (a.platform, a.entry) for aid, a in aa.items()}
        obs, rew, term, trunc, info = env.step(ad, lambda_team=0.3)
        m = info.get("metrics", {})
        g += m.get("n_arrivals", 0); o += m.get("n_ontime_arr", 0)
        for ev in info.get("l3_transit_events", []):
            ev["priority"] = cfg.trains[ev["train_id"]].priority
            arr_events.append(ev)
        if term or trunc:
            status = info["episode_status"]; break
    return {"incident": incident, "bunched": bunched >= 1,
            "g": g, "o": o, "arr_events": arr_events, "status": status,
            "n_sections": len(env._l3_behavior)}


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
    }


def report_incident(eps, mode="learned"):
    """Incident-aware: realised arrival delay, affected (transit overlapped the
    restriction window) vs unaffected, and incident-episode vs non-incident."""
    aff = [ev["arr_delay"] for r in eps for ev in r["arr_events"] if ev["affected"]]
    unaff = [ev["arr_delay"] for r in eps for ev in r["arr_events"] if not ev["affected"]]
    inc_eps = [r for r in eps if r["incident"]]
    non_eps = [r for r in eps if not r["incident"]]
    n_aff_ep = sum(1 for r in eps if any(ev["affected"] for ev in r["arr_events"]))
    print(f"  affected   : {_fmt_dist(_dist(aff))}  (transits in window)")
    print(f"  unaffected : {_fmt_dist(_dist(unaff))}")
    print(f"  inc episodes: {len(inc_eps)}/{len(eps)}, with >=1 affected transit: {n_aff_ep}")
    print(f"  -- episodes WITH incident ({len(inc_eps)}) --")
    print(f"     {_report_episode_level(inc_eps)}")
    if non_eps:
        print(f"  -- episodes WITHOUT incident ({len(non_eps)}) --")
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
    # E1 re-solve compute (per MODE): SA re-solves once per corridor transit
    # (3.48 ms, the 300-iteration anneal); learned reacts per section (0.36 ms,
    # one forward); sa-online is a per-section greedy (~0.5 ms); cruise ~0.
    # This is the "SA 重排代价 vs learned 即时响应成本" — with the random extra
    # train, each extra transit is one more SA re-solve vs one more learned pass.
    n_transits = sum(len(r["arr_events"]) for r in eps)
    n_sections = sum(r.get("n_sections", 0) for r in eps)
    per_section_ms = 0.36 if mode == "learned" else 0.5  # learned vs greedy sa-online
    if mode in ("sa", "sa-nofc"):
        method_ms = n_transits * 3.48
    elif mode in ("learned", "sa-online"):
        method_ms = n_sections * per_section_ms
    elif mode == "flat":
        method_ms = n_transits * 0.5  # one joint 72-action forward per transit (approx)
    else:
        method_ms = 0.0
    sa_ms = n_transits * 3.48
    lr_ms = n_sections * 0.36
    print(f"  E1 re-solve compute [{mode}]: transits={n_transits} sections={n_sections} | "
          f"method ~{method_ms:.0f}ms | SA 重排 ~{sa_ms:.0f}ms vs learned 即时 ~{lr_ms:.0f}ms "
          f"({sa_ms/max(lr_ms,1e-9):.1f}x)")
    return {
        "ontime": all_on / max(all_n, 1) * 100,
        "p1": byp_all[1][0] / max(byp_all[1][1], 1) * 100,
        "p2": byp_all[2][0] / max(byp_all[2][1], 1) * 100,
        "p3": byp_all[3][0] / max(byp_all[3][1], 1) * 100,
        "n": dall[0], "mean": dall[1], "p50": dall[2], "p90": dall[3],
        "p95": dall[4], "max": dall[5],
        "n_transits": n_transits, "n_sections": n_sections,
        "sa_resolve_ms": round(sa_ms, 1), "learned_ms": round(lr_ms, 1),
        "method_ms": round(method_ms, 1),
        "resolve_ratio": round(sa_ms / max(lr_ms, 1e-9), 1),
        "aff_n": da[0], "aff_mean": da[1], "aff_p90": da[3], "aff_max": da[5],
        "unaff_n": du[0], "unaff_mean": du[1], "unaff_max": du[5],
        "sev_rate": s_all[1], "sev_mean": s_all[2], "sev_max": s_all[3],
        "aff_sev_mean": s_aff[2], "aff_sev_max": s_aff[3],
        "succ": st.get("success", 0), "deadlock": st.get("deadlock", 0),
        "trunc": st.get("truncated", 0),
        "inc_eps": len(inc_eps), "aff_eps": n_aff_ep,
    }


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
    ap.add_argument("--sa-online-only", action="store_true",
                    help="partial-observation SA baseline (per-section greedy, "
                         "current state only — the fair match for the learned worker)")
    ap.add_argument("--sa-nofc-only", action="store_true",
                    help="SA with NO incident forecast: plans as if clear but "
                         "executes the real restriction (the cost of needing a "
                         "forecast vs MAPPO's reactive adaptation)")
    ap.add_argument("--cruise-only", action="store_true")
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
    ap.add_argument("--vt", action="store_true", help="enable visitor trains")
    ap.add_argument("--blocks", type=int, default=3,
                    help="independent seed blocks (seed sets are {seeds+b*1e6..}); "
                         "variance across blocks addresses seed-set sensitivity")
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
    elif args.sa_only:
        modes = ["sa"]
    elif args.sa_online_only:
        modes = ["sa-online"]
    elif args.sa_nofc_only:
        modes = ["sa-nofc"]
    elif args.learned_only:
        modes = ["learned"]
    elif args.flat_only:
        modes = ["flat"]
    else:
        modes = ["learned", "sa", "sa-online", "sa-nofc", "cruise"]
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
                               flat_actor)
                   for s in range(base, base + args.n)]
            blocks.append(eps)
        per_mode[mode] = blocks

    # ---- per-mode reports ----
    stats_by_mode = {}
    for mode in modes:
        all_eps = [r for b in per_mode[mode] for r in b]
        print(f"\n== {mode} (held-out {args.blocks}x{args.n} seeds, start {args.seeds}) ==")
        if args.incident:
            stats_by_mode[mode] = report_incident(all_eps, mode)
        else:
            by_bunch = {"bunched": [], "non": []}
            for r in all_eps:
                by_bunch["bunched" if r["bunched"] else "non"].append(r)
            report_legacy("  combined       ", by_bunch["bunched"] + by_bunch["non"])
            report_legacy(f"  bunched ({len(by_bunch['bunched'])})", by_bunch["bunched"])
            report_legacy(f"  non-bunched ({len(by_bunch['non'])})", by_bunch["non"])
            stats_by_mode[mode] = report_legacy(
                "  combined       ", by_bunch["bunched"] + by_bunch["non"])

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

        if args.incident:
            write_metric("ontime.csv", ["ontime", "p1", "p2", "p3"])
            write_metric("delay.csv", ["n", "mean", "p50", "p90", "p95", "max"])
            write_metric("affected.csv", ["aff_n", "aff_mean", "aff_p90", "aff_max",
                                          "unaff_n", "unaff_mean", "unaff_max"])
            write_metric("tail.csv", ["sev_rate", "sev_mean", "sev_max",
                                      "aff_sev_mean", "aff_sev_max"])
            write_metric("compute.csv", ["n_transits", "n_sections", "method_ms",
                                         "sa_resolve_ms", "learned_ms", "resolve_ratio"])
            write_metric("episode.csv", ["succ", "deadlock", "trunc", "inc_eps", "aff_eps"])
        else:
            write_metric("ontime.csv", ["ontime"])
            write_metric("delay.csv", ["n", "mean", "p50", "p90", "p95", "max"])
            write_metric("tail.csv", ["sev_rate", "sev_mean", "sev_max"])
            write_metric("episode.csv", ["succ", "deadlock", "trunc"])
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
                row = {"mode": mode, **{k: round(v, 3) for k, v in stats_by_mode[mode].items()}}
                w.writerow(row)
        print(f"saved {args.out_csv}")

    # ---- incident arm: paired excess vs CRUISE + seed-block variance ----
    if args.incident and "cruise" in per_mode:
        ref = [r for b in per_mode["cruise"] for r in b]
        print("\n== incident arm: PAIRED excess vs CRUISE on affected subset ==")
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
