#!/usr/bin/env python3
"""Delay-propagation report: MAPPO (3 training seeds) vs SA-online.

Writes a Markdown/CSV table into the output root and two figures:
  - qq_plot.png       quantile-quantile of arrival delay
  - forest_plot.png   paired effects with 95% CI
All pairing uses the same 180 evaluation seeds; uncertainty across the 3 MAPPO
training seeds uses t_{0.975,2}=4.303.  Run after the 180-ep evals.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.experiment.scenario_loader import load_scenario

T_DF2 = 4.303
POS_U = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
POS_D = {"E": 0, "D": 1, "C": 2, "B": 3, "A": 4}


def metrics(df, cfg):
    d = df.copy()
    d["hw"] = d.headway_delay.clip(lower=0)
    d["pos"] = [
        (POS_U if cfg.trains[t].direction == "up" else POS_D)[a]
        for t, a in zip(d.train_id, d.arr_station)]

    hw = d.groupby(["block", "seed"]).hw.sum().rename("hw_sum").reset_index()
    growth, pers, pct = [], [], []
    for (b, s, t), g in d.groupby(["block", "seed", "train_id"]):
        g = g.sort_values("pos")
        x = g.arr_delay.values
        if len(x) >= 2:
            growth.append(dict(block=b, seed=s, train_id=t, delta=x[-1] - x[0]))
        late = [i for i in range(len(x) - 1) if x[i] > 15]
        if late:
            pers.append(dict(block=b, seed=s,
                             persist=float(np.mean([x[i + 1] > 15 for i in late]))))
    for (b, s), g in d.groupby(["block", "seed"]):
        x = g.arr_delay.values
        pct.append(dict(block=b, seed=s,
                        p50=np.percentile(x, 50),
                        p90=np.percentile(x, 90),
                        p95=np.percentile(x, 95)))
    return dict(
        hw=hw, growth=pd.DataFrame(growth), pers=pd.DataFrame(pers),
        pct=pd.DataFrame(pct), hw_per_transit=float(d.hw.mean()),
        overall_p50=float(np.percentile(d.arr_delay, 50)),
        overall_p90=float(np.percentile(d.arr_delay, 90)),
        overall_p95=float(np.percentile(d.arr_delay, 95)))


def paired_delta(m, s, key, metric, keys):
    a = m[key]
    b = s[key]
    mm = a.merge(b, on=keys, suffixes=("_m", "_s"))
    return float((mm[f"{metric}_m"] - mm[f"{metric}_s"]).mean())


def agg(vals):
    v = np.array(vals, float)
    mean = float(v.mean())
    sd = float(v.std(ddof=1))
    ci = T_DF2 * sd / np.sqrt(len(v))
    return mean, sd, ci


def ddt_from_log(path):
    try:
        txt = Path(path).read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"DDT median=([\d.]+)s", txt)
        return float(m.group(1)) if m else None
    except FileNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mappo", nargs="+", default=[
        "output/11_09/eval_l3scratch4k_seed1L2_180/per_transit.csv",
        "output/11_09/eval_l3scratch4k_trains1_180/per_transit.csv",
        "output/11_09/eval_l3scratch4k_trains2_180/per_transit.csv"])
    ap.add_argument("--mappo-logs", nargs="+", default=[
        "output/11_09/eval_l3scratch4k_seed1L2_180.log",
        "output/11_09/eval_l3scratch4k_trains1_180.log",
        "output/11_09/eval_l3scratch4k_trains2_180.log"])
    ap.add_argument("--sa", default="output/11_09/eval_saonline_seed1L2_180/per_transit.csv")
    ap.add_argument("--sa-log", default="output/11_09/eval_saonline_seed1L2_180.log")
    ap.add_argument("--out-dir", type=Path, default=Path("output/11_09"))
    ap.add_argument("--scenario", default="five_station_15t")
    args = ap.parse_args()

    cfg = load_scenario(scenario_name=args.scenario)
    M = [metrics(pd.read_csv(p), cfg) for p in args.mappo]
    S = metrics(pd.read_csv(args.sa), cfg)

    spec = [
        ("Headway delay / episode", "hw", "hw_sum", ["block", "seed"], "s", 1.0),
        ("Late persistence", "pers", "persist", ["block", "seed"], "pt", 100.0),
        ("p50 (per-episode)", "pct", "p50", ["block", "seed"], "s", 1.0),
        ("p90 (per-episode)", "pct", "p90", ["block", "seed"], "s", 1.0),
        ("p95 (per-episode)", "pct", "p95", ["block", "seed"], "s", 1.0),
        ("Route growth Δlast−first", "growth", "delta",
         ["block", "seed", "train_id"], "s", 1.0),
    ]
    paired = {}
    for label, key, metric, keys, unit, scale in spec:
        vals = [paired_delta(m, S, key, metric, keys) * scale for m in M]
        mean, sd, ci = agg(vals)
        paired[label] = (mean, ci, vals)

    def mappo_mean_sd(fn):
        v = np.array([fn(m) for m in M], float)
        return float(v.mean()), float(v.std(ddof=1))

    hw_abs = mappo_mean_sd(lambda m: m["hw"].hw_sum.mean())
    hw_sa = float(S["hw"].hw_sum.mean())
    hwt_abs = mappo_mean_sd(lambda m: m["hw_per_transit"])
    hwt_sa = S["hw_per_transit"]
    gr_abs = mappo_mean_sd(lambda m: m["growth"].delta.mean())
    gr_sa = float(S["growth"].delta.mean())
    grm_abs = mappo_mean_sd(lambda m: m["growth"].delta.median())
    grm_sa = float(S["growth"].delta.median())
    pe_abs = mappo_mean_sd(lambda m: 100.0 * m["pers"].persist.mean())
    pe_sa = 100.0 * float(S["pers"].persist.mean())
    p50_abs = mappo_mean_sd(lambda m: m["pct"].p50.mean())
    p50_sa = float(S["pct"].p50.mean())
    p90_abs = mappo_mean_sd(lambda m: m["pct"].p90.mean())
    p90_sa = float(S["pct"].p90.mean())
    p95_abs = mappo_mean_sd(lambda m: m["pct"].p95.mean())
    p95_sa = float(S["pct"].p95.mean())
    ddt_m = [ddt_from_log(p) for p in args.mappo_logs]
    ddt_sa = ddt_from_log(args.sa_log)

    src = args.out_dir
    src.mkdir(parents=True, exist_ok=True)
    rows = []

    def add(metric, m_abs, sa, delta, ci, unit, verdict):
        delta_txt = "—" if delta is None else (
            f"{delta:+.2f} [{delta-ci:+.2f},{delta+ci:+.2f}]")
        rows.append(dict(metric=metric,
                         mappo_mean=None if m_abs is None else m_abs[0],
                         mappo_sd=None if m_abs is None else m_abs[1],
                         sa=sa, delta=delta, ci_low=None if delta is None else delta - ci,
                         ci_high=None if delta is None else delta + ci,
                         unit=unit, verdict=verdict, delta_txt=delta_txt))

    add("Headway delay / episode", hw_abs, hw_sa,
        *paired["Headway delay / episode"][:2], "s", "MAPPO better")
    add("Headway delay / transit", hwt_abs, hwt_sa, None, 0, "s", "MAPPO lower")
    add("Route growth Δlast−first (mean)", gr_abs, gr_sa,
        *paired["Route growth Δlast−first"][:2], "s", "flat")
    add("Route growth (median)", grm_abs, grm_sa, None, 0, "s",
        "MAPPO slightly higher")
    add("Late persistence", pe_abs, pe_sa,
        *paired["Late persistence"][:2], "pt", "directionally better, n.s.")
    add("p50 (per-episode)", p50_abs, p50_sa,
        *paired["p50 (per-episode)"][:2], "s", "flat")
    add("p90 (per-episode)", p90_abs, p90_sa,
        *paired["p90 (per-episode)"][:2], "s",
        "directionally better (CI crosses 0)")
    add("p95 (per-episode)", p95_abs, p95_sa,
        *paired["p95 (per-episode)"][:2], "s", "flat")
    add("DDT median", None, ddt_sa, None, 0, "s",
        "not robust (per-seed " + "/".join(
            "n/a" if x is None else f"{x:.0f}" for x in ddt_m) + ")")

    out_csv = src / "propagation_table.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    lines = [
        "# Delay propagation: MAPPO (4k, 3 seeds) vs SA-online",
        "",
        f"Paired by the same 180 evaluation seeds; 95% CI across 3 MAPPO training "
        f"seeds (t_0.975,2={T_DF2}). MAPPO absolute = mean ± sd across training seeds.",
        "",
        "| Metric | MAPPO | SA-online | Paired Δ [95% CI] | Verdict |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        m = "—" if r["mappo_mean"] is None else (
            f"{r['mappo_mean']:.2f} ± {r['mappo_sd']:.2f}{r['unit']}")
        sa = "—" if r["sa"] is None else f"{r['sa']:.2f}{r['unit']}"
        lines.append(f"| {r['metric']} | {m} | {sa} | {r['delta_txt']} | {r['verdict']} |")
    lines += [
        "",
        f"Overall pooled p50/p90/p95: MAPPO ≈ "
        f"{np.mean([m['overall_p50'] for m in M]):.2f}/"
        f"{np.mean([m['overall_p90'] for m in M]):.2f}/"
        f"{np.mean([m['overall_p95'] for m in M]):.2f}s; "
        f"SA-online {S['overall_p50']:.2f}/{S['overall_p90']:.2f}/"
        f"{S['overall_p95']:.2f}s.",
        "",
        "Conclusion: MAPPO significantly reduces moving-block (headway) delay "
        "propagation; it does not reduce end-to-end route accumulation or the "
        "extreme p95 tail, which are dominated by G238@C_to_D and G1487@B_to_A.",
    ]
    out_md = src / "propagation_table.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Q-Q plot
    L = pd.concat([pd.read_csv(p) for p in args.mappo]).arr_delay.values
    Ss = pd.read_csv(args.sa).arr_delay.values
    q = np.linspace(0, 100, 201)
    ql, qs = np.percentile(L, q), np.percentile(Ss, q)
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.plot(qs, ql, color="#1f77b4", lw=2, label="MAPPO pooled (3 seeds)")
    lim = [min(qs.min(), ql.min()) - 5, max(qs.max(), ql.max()) + 5]
    ax.plot(lim, lim, color="#888888", ls="--", lw=1, label="y=x")
    for qq, lab in [(50, "p50"), (90, "p90"), (95, "p95")]:
        a, b = np.percentile(Ss, qq), np.percentile(L, qq)
        ax.scatter([a], [b], color="#d62728", zorder=5, s=30)
        ax.annotate(lab, (a, b), textcoords="offset points",
                    xytext=(6, 6), fontsize=9, color="#d62728")
    ax.set_xlabel("SA-online delay quantile (s)")
    ax.set_ylabel("MAPPO delay quantile (s)")
    ax.set_title("Q–Q of arrival delay: MAPPO vs SA-online")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    qq_out = src / "propagation_qq.png"
    fig.savefig(qq_out, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # forest plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    labels = ["route growth", "p50", "p90", "p95"]
    keys = ["Route growth Δlast−first", "p50 (per-episode)",
            "p90 (per-episode)", "p95 (per-episode)"]
    means = [paired[k][0] for k in keys]
    cis = [paired[k][1] for k in keys]
    y = np.arange(len(labels))
    ax = axes[0]
    ax.errorbar(means, y, xerr=cis, fmt="o", color="#1f77b4", capsize=4)
    ax.axvline(0, color="#888888", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("paired Δ (MAPPO − SA), s")
    ax.set_title("Delay metrics")
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    d = paired["Headway delay / episode"]
    ax.errorbar([d[0]], [0], xerr=[d[1]], fmt="o", color="#2ca02c", capsize=5)
    ax.axvline(0, color="#888888", ls="--", lw=1)
    ax.set_yticks([0])
    ax.set_yticklabels(["headway sum / episode"])
    ax.set_xlabel("paired Δ (MAPPO − SA), s")
    ax.set_title("Moving-block propagation")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(-450, 50)

    ax = axes[2]
    d = paired["Late persistence"]
    ax.errorbar([d[0]], [0], xerr=[d[1]], fmt="o", color="#9467bd", capsize=5)
    ax.axvline(0, color="#888888", ls="--", lw=1)
    ax.set_yticks([0])
    ax.set_yticklabels(["late persistence"])
    ax.set_xlabel("paired Δ (MAPPO − SA), percentage points")
    ax.set_title("Late-delay persistence")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(-40, 20)

    fig.suptitle("Paired effects (3 MAPPO training seeds vs SA-online, 95% CI)",
                 fontsize=12)
    fig.tight_layout()
    forest_out = src / "propagation_forest.png"
    fig.savefig(forest_out, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"saved {out_md}")
    print(f"saved {out_csv}")
    print(f"saved {qq_out}")
    print(f"saved {forest_out}")


if __name__ == "__main__":
    main()
