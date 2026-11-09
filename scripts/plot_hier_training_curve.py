#!/usr/bin/env python3
"""Smoothed training-trajectory figure + convergence check for a hierarchical run.

CSV-first: reads `<run>/training_metrics.csv` (and sibling `eval_ontime.csv` if
present), writes
  - `training_curve_smoothed.csv`  rolling mean/std per metric (window W)
  - `conv_check.csv`               phase-wise and last-vs-previous window stats
  - `fig_hier_training_curve.png`  4-panel smoothed curves with Phase A/B split

Usage:
    python scripts/plot_hier_training_curve.py --run-dir output/10_29/training/five_station_15t_hierarchical/<ts>
    # or omit --run-dir to pick the latest run under --train-root
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 0.8, "axes.edgecolor": "#444444",
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8, "legend.frameon": True, "legend.framealpha": 0.85,
    "legend.edgecolor": "#bbbbbb", "savefig.dpi": 300, "savefig.bbox": "tight",
})
C_MAIN = "#1f4e79"
C_EVAL = "#c0392b"


def rolling_stats(vals: np.ndarray, win: int):
    mean = np.full(len(vals), np.nan)
    std = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        seg = vals[max(0, i - win + 1):i + 1]
        mean[i] = np.mean(seg)
        std[i] = np.std(seg)
    return mean, std


def _f(row, key):
    v = row.get(key)
    if v is None or v == "" or v == "nan":
        return float("nan")
    return float(v)


def load_metrics(path: Path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["episode"]))
    eps = np.array([int(r["episode"]) for r in rows])
    metric = {}
    for key in ("total_reward", "mean_reward", "mean_dep_delay",
                "ontime_rate", "arr_ontime"):
        metric[key] = np.array([_f(r, key) for r in rows])
    phase = np.array([r.get("phase", "") for r in rows])
    status = np.array([r.get("status", "") for r in rows])
    return eps, metric, phase, status


def load_eval(path: Path):
    if not path.exists():
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return (np.array([int(r["episode"]) for r in rows]),
            np.array([float(r["arr_ontime"]) for r in rows]))


def phase_bounds(eps, phase):
    """[(name, start_ep, end_ep)] in run order."""
    out = []
    for name in dict.fromkeys(phase):
        idx = np.where(phase == name)[0]
        out.append((name, int(eps[idx[0]]), int(eps[idx[-1]])))
    return out


def window_mean(vals, lo, hi):
    seg = vals[(np.arange(len(vals)) >= lo) & (np.arange(len(vals)) < hi)]
    seg = seg[~np.isnan(seg)]
    return float(np.mean(seg)) if len(seg) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="timestamped run dir; default = latest under --train-root")
    ap.add_argument("--train-root", type=Path,
                    default=Path("output/10_29/training/five_station_15t_hierarchical"))
    ap.add_argument("--window", type=int, default=100,
                    help="rolling window (episodes) for smoothing")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        cands = [p for p in args.train_root.iterdir()
                 if p.is_dir() and len(p.name) == 15 and p.name[:8].isdigit()]
        if not cands:
            sys.exit(f"no timestamped run dirs under {args.train_root}")
        run_dir = max(cands, key=lambda p: p.name)
    out_dir = args.out_dir or run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    eps, metric, phase, status = load_metrics(run_dir / "training_metrics.csv")
    n = len(eps)
    W = args.window
    bounds = phase_bounds(eps, phase)
    print(f"Run: {run_dir}  ({n} episodes; phases: "
          + ", ".join(f"{b[0]}={b[2] - b[1] + 1}" for b in bounds) + ")")

    # ---- rolling curves CSV ----
    smooth = {k: rolling_stats(v, W) for k, v in metric.items()}
    with open(out_dir / "training_curve_smoothed.csv", "w", newline="") as f:
        keys = ["episode", "phase"]
        for k in metric:
            keys += [f"{k}_mean", f"{k}_std"]
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(n):
            vals = [int(eps[i]), phase[i]]
            for k in metric:
                vals += [round(float(smooth[k][0][i]), 5),
                         round(float(smooth[k][1][i]), 5)]
            w.writerow(vals)

    # ---- convergence table: phase-wise + last-vs-previous windows ----
    def stats(lo, hi):
        out = {}
        for k in metric:
            seg = metric[k][lo:hi]
            seg = seg[~np.isnan(seg)]
            out[k] = (float(np.mean(seg)), float(np.std(seg)), len(seg))
        return out

    rows = []
    for name, lo, hi in bounds:
        s = stats(int(np.searchsorted(eps, lo)), int(np.searchsorted(eps, hi, side="right")))
        for k, (mu, sd, cnt) in s.items():
            rows.append({"scope": f"phase_{name}", "metric": k, "n": cnt,
                         "mean": round(mu, 5), "std": round(sd, 5),
                         "delta_mean": "", "delta_sigma": "", "stationary": ""})
    for win in (500, 1000):
        if n < 2 * win:
            continue
        last = stats(n - win, n)
        prev = stats(n - 2 * win, n - win)
        for k in metric:
            mu_l, sd_l, _ = last[k]
            mu_p, sd_p, _ = prev[k]
            # Stationary if the mean shift between consecutive windows is small
            # against the episode-to-episode spread of the last window
            # (|Δmean| <= 0.5 σ_last), i.e. the smoothed curve is flat within
            # its own noise band.
            shift = abs(mu_l - mu_p)
            shift_sigma = shift / sd_l if sd_l > 0 else float("inf")
            stationary = bool(shift_sigma <= 0.5)
            rows.append({"scope": f"last{win}_vs_prev{win}", "metric": k, "n": win,
                         "mean": round(mu_l, 5), "std": round(sd_l, 5),
                         "delta_mean": round(mu_l - mu_p, 5),
                         "delta_sigma": round(shift_sigma, 4),
                         "stationary": int(stationary)})
    with open(out_dir / "conv_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scope", "metric", "n", "mean", "std",
                                          "delta_mean", "delta_sigma", "stationary"])
        w.writeheader()
        w.writerows(rows)

    # ---- figure ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharex=True)
    panels = [
        ("mean_reward", "Mean per-train reward", None),
        ("mean_dep_delay", "Mean departure delay (s)", None),
        ("ontime_rate", "Departure on-time rate", (0, 1)),
        ("arr_ontime", "Arrival on-time rate (realised)", (0, 1)),
    ]
    ev = load_eval(run_dir / "eval_ontime.csv")
    x_max = int(eps[-1])
    for ax, (key, title, ylim) in zip(axes.flat, panels):
        mu, sd = smooth[key]
        ax.fill_between(eps, mu - sd, mu + sd, color=C_MAIN, alpha=0.15, linewidth=0)
        ax.plot(eps, mu, color=C_MAIN, linewidth=1.5,
                label=f"rolling mean ±1 std ({W} ep)")
        if key == "arr_ontime" and ev is not None:
            ax.plot(ev[0], ev[1], "o", color=C_EVAL, markersize=3.2,
                    markeredgewidth=0, label="held-out eval (12 ep)")
        if len(bounds) > 1:
            b_ep = bounds[0][2] + 1
            ax.axvline(b_ep, color="#7f8c8d", ls="--", lw=0.9)
            ax.axvspan(0, b_ep, color="#d5dbdb", alpha=0.25)
        ax.set_title(title)
        ax.grid(alpha=0.25, ls=":")
        if ylim is not None:
            ax.set_ylim(*ylim)
        if key in ("mean_dep_delay",):
            ax.set_ylim(bottom=0)
        ax.legend(loc="lower right", fontsize=7.5)
    for ax in axes[1]:
        ax.set_xlabel("Training episode")
        ax.set_xlim(0, x_max)
    if len(bounds) > 1:
        axes[0, 0].text(bounds[0][2] * 0.12, 0.92,
                        f"Phase {bounds[0][0]} (L2 + SA-L3)", transform=axes[0, 0].get_xaxis_transform(),
                        fontsize=8, color="#555555")
        axes[0, 0].text(bounds[0][2] + 0.03 * x_max, 0.92,
                        f"Phase {bounds[1][0]} (L3 learned)", transform=axes[0, 0].get_xaxis_transform(),
                        fontsize=8, color="#c0392b")
    fig.suptitle(f"Hierarchical MAPPO — 15 trains, {n} episodes, seed 0 "
                 f"(corridor incident disturbance)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_path = out_dir / "fig_hier_training_curve.png"
    fig.savefig(fig_path)
    print(f"Wrote {out_dir/'training_curve_smoothed.csv'}")
    print(f"Wrote {out_dir/'conv_check.csv'}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
