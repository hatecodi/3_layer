#!/usr/bin/env python3
"""Scalability paper figures + convergence analysis from training/eval CSVs (CSV-first).

Reads:
  - training_metrics.csv from from-scratch runs (per-episode `arr_ontime` column,
    added to run_flat_mappo.py / run_train_hierarchical.py): the convergence data.
  - eval ontime.csv per scale (eval_protocol --out-dir): the transfer quality curve.

Writes into a dated out-dir (default output/9_4/scalability):
  - conv_summary.csv      — (scale, method) x {episodes_to_50pct, final_roll_arr_ontime}
  - conv_curves.png       — 2-panel (15t, 25t): rolling arr-ontime vs episodes, hier vs flat
  - scalability_quality.png — x=N(10..30), transfer ontime% per mode + from-scratch markers

Convergence definition (fixed, comparable across methods):
  rolling arr-ontime over a 50-episode window; "episodes to 50%" = first episode whose
  rolling value stays >= 0.50 for the following 50 episodes (sustained). None if never.

Usage:
    python scripts/plot_conv_scalability.py [--scal-dir output/9_4/scalability]
        [--train-root output/8_16/training] [--legacy-scal-dir PATH]
        [--flat-15t PATH] [--flat-25t PATH] [--hier-15t PATH] [--hier-25t PATH]
    # --scal-dir holds per-scale ontime.csv (P3-fixed transfer eval in output/9_4),
    # --legacy-scal-dir is an OPTIONAL fallback for scales missing there.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# SCI figure style: serif font, thin axes, light dashed grid, no chartjunk.
plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 0.8, "axes.edgecolor": "#444444",
    "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "legend.frameon": True, "legend.framealpha": 0.85,
    "legend.edgecolor": "#bbbbbb",
    "lines.linewidth": 1.6,
})
COLORS = {"learned": "#c0392b", "sa": "#2874a6", "sa-online": "#148f77",
          "sa-nofc": "#b7950b", "cruise": "#7d6608", "flat": "#6c3483"}
WIN = 50
SMOOTH = 100          # rolling window (episodes) for the mean/std band
TARGET = 0.50
SCALES = [10, 15, 20, 25, 30]


def rolling_stats(vals, win):
    """(rolling_mean, rolling_std) over a trailing window of `win` episodes."""
    n = len(vals)
    mean = np.full(n, np.nan)
    std = np.full(n, np.nan)
    for i in range(n):
        seg = vals[max(0, i - win + 1):i + 1]
        mean[i] = np.mean(seg)
        std[i] = np.std(seg)
    return mean, std


def latest_run(root: Path) -> Path:
    """Latest timestamped run dir under root (flat_mappo or {env}_hierarchical)."""
    cands = [p for p in root.iterdir() if p.is_dir() and len(p.name) == 15
             and p.name[:8].isdigit() and p.name[8] == "_" and p.name[9:].isdigit()]
    if not cands:
        raise FileNotFoundError(f"no timestamped run dirs under {root}")
    return max(cands, key=lambda p: p.name)


def load_arr_ontime(csv_path: Path):
    """Return (eps, vals, phase_b_start_episode). phase_b_start is the cumulative
    episode where the L3 phase begins (hierarchical only); None for flat (no phases)."""
    rows = []
    phase_b = None
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            v = r.get("arr_ontime")
            if v is None or v == "":
                continue
            ph = r.get("phase")
            rows.append((int(r["episode"]), float(v), ph))
    rows.sort()
    eps = [p[0] for p in rows]
    vals = np.array([p[1] for p in rows])
    for ep, _, ph in rows:
        if ph == "B":
            phase_b = ep
            break
    return eps, vals, phase_b


def rolling(vals, win):
    out = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        lo = max(0, i - win + 1)
        out[i] = np.mean(vals[lo:i + 1])
    return out


def resolve_curve(run_csv):
    """Given a training_metrics.csv path, prefer the sibling eval_ontime.csv
    (held-out eval at checkpoints — cleaner, matches the paper's eval protocol).
    Returns (eps, vals, phase_b, is_heldout)."""
    p = Path(run_csv)
    ev = p.parent / "eval_ontime.csv"
    if ev.exists():
        eps, vals, pb = load_arr_ontime(ev)
        return eps, vals, pb, True
    eps, vals, pb = load_arr_ontime(p)
    return eps, vals, pb, False


def episodes_to_sustained_cp(vals, eps, target, ncp=3, start_idx=0):
    """Sparse held-out curve: first checkpoint whose NEXT ncp consecutive
    checkpoints all have arr_ontime >= target. None if never."""
    for i in range(start_idx, len(vals) - ncp + 1):
        if np.all(vals[i:i + ncp] >= target):
            return eps[i]
    return None


def episodes_to_sustained(vals, eps, win, target, start_idx=0):
    """First episode (>= start_idx) whose rolling>=target is followed by a FULL
    2*win window still >=target (so an end-of-run noise spike can't qualify).
    None if never."""
    n = len(vals)
    if n < win * 2:
        return None
    for i in range(start_idx, n - win * 2 + 1):
        if np.mean(vals[i:i + win]) >= target and np.mean(vals[i:i + win * 2]) >= target:
            return eps[i]
    return None


def load_eval_ontime(scal_dir: Path, scale: int):
    f = scal_dir / f"{scale}t" / "ontime.csv"
    if not f.exists():
        return {}
    out = {}
    with open(f, newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["mode"]] = float(r["ontime"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scal-dir", type=Path, default=Path("output/9_4/scalability"))
    ap.add_argument("--legacy-scal-dir", type=Path, default=None)
    ap.add_argument("--train-root", type=Path, default=Path("output/8_16/training"))
    ap.add_argument("--flat-15t", type=Path, default=None)
    ap.add_argument("--flat-25t", type=Path, default=None)
    ap.add_argument("--hier-15t", type=Path, default=None)
    ap.add_argument("--hier-25t", type=Path, default=None)
    args = ap.parse_args()
    args.scal_dir.mkdir(parents=True, exist_ok=True)

    def resolve(p, root, sub):
        if p is not None:
            return p
        return latest_run(root / sub) / "training_metrics.csv"

    runs = {
        "flat": {
            15: resolve(args.flat_15t, args.train_root, "flat_mappo"),
            25: resolve(args.flat_25t, args.train_root, "flat_mappo"),
        },
        "hier": {
            15: resolve(args.hier_15t, args.train_root, "five_station_15t_hierarchical"),
            25: resolve(args.hier_25t, args.train_root, "five_station_25t_hierarchical"),
        },
    }

    # ---- convergence summary (held-out eval preferred; per-episode fallback) ----
    conv_rows = []
    curves = {}   # (method, scale) -> per-episode (eps, vals, phase_b) for the curves
    for method, scales in runs.items():
        for scale, csv_path in scales.items():
            # per-episode training data (for the mean+std band / line)
            te_eps, te_vals, te_pb = load_arr_ontime(csv_path)
            # held-out eval at checkpoints (for the rigorous summary metrics)
            he_eps, he_vals, he_pb, is_heldout = resolve_curve(csv_path)
            if is_heldout:
                final_roll = float(he_vals[-1])
                e2t = episodes_to_sustained_cp(he_vals, he_eps, TARGET, ncp=3)
                pb_ep_list, pb_vals, pb_start = he_eps, he_vals, he_pb
            else:
                roll = rolling(te_vals, WIN)
                final_roll = float(roll[-1])
                e2t = episodes_to_sustained(roll, te_eps, WIN, TARGET)
                pb_ep_list, pb_vals, pb_start = te_eps, roll, te_pb
            # Phase-B-only: when does the LEARNED L3 (not the SA-pretrain) first
            # sustain >= target? Hierarchical only; flat has no phases.
            e2t_b = None
            if pb_start is not None:
                idx = min(range(len(pb_ep_list)), key=lambda i: abs(pb_ep_list[i] - pb_start))
                e2t_b = (episodes_to_sustained_cp(pb_vals, pb_ep_list, TARGET, ncp=3,
                                                  start_idx=idx)
                         if is_heldout else
                         episodes_to_sustained(pb_vals, pb_ep_list, WIN, TARGET,
                                               start_idx=idx))
            src = "heldout" if is_heldout else "train-roll"
            print(f"{method} {scale}t [{src}]: final={final_roll:.3f} "
                  f"episodes_to_50pct={e2t} phaseB_only={e2t_b}")
            conv_rows.append({
                "scale": scale, "method": method, "csv": str(csv_path),
                "source": src,
                "final_roll_arr_ontime": round(final_roll, 4),
                "episodes_to_50pct": "" if e2t is None else e2t,
                "phaseB_episodes_to_50pct": "" if e2t_b is None else e2t_b,
            })
            curves[(method, scale)] = (te_eps, te_vals, te_pb)

    with open(args.scal_dir / "conv_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scale", "method", "csv", "source",
                                          "final_roll_arr_ontime",
                                          "episodes_to_50pct",
                                          "phaseB_episodes_to_50pct"])
        w.writeheader()
        w.writerows(conv_rows)

    C_HIER = "#1f77b4"    # blue
    C_FLAT = "#d62728"    # red (colorblind-safe blue/orange-like pair)

    def band_plot(ax, eps, vals, color, label, mark_every, ms=3.5):
        """SCI-style curve: rolling-mean line + rolling-std shaded band + sparse markers."""
        mu, sd = rolling_stats(vals, SMOOTH)
        ax.fill_between(eps, (mu - sd) * 100, (mu + sd) * 100,
                        color=color, alpha=0.15, linewidth=0)
        ax.plot(eps, mu * 100, color=color, label=label, linewidth=1.6)
        ax.plot(eps[::mark_every], mu[::mark_every] * 100, "o",
                color=color, markersize=ms, markeredgewidth=0.4, markeredgecolor="white")

    # ---- Fig 1: full-pipeline convergence, Phase A/B annotated ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, scale in zip(axes, [15, 25]):
        hier_eps, hier_vals, hier_pb = curves[("hier", scale)]
        flat_eps, flat_vals, _ = curves[("flat", scale)]
        if hier_pb is not None:
            ax.axvspan(0, hier_pb, color="#f5b7b1", alpha=0.30,
                       label="Phase A (L2 + SA-L3)")
            ax.axvline(hier_pb, color="#1f77b4", ls="--", lw=1.0)
        band_plot(ax, flat_eps, flat_vals, C_FLAT, "flat MAPPO", 400)
        band_plot(ax, hier_eps, hier_vals, C_HIER, "hierarchical (Phase A+B)", 400)
        ax.axhline(TARGET * 100, color="gray", ls="--", lw=0.8)
        ax.text(hier_eps[-1] * 0.01, TARGET * 100 + 2, f"{TARGET:.0%}",
                color="gray", fontsize=8)
        ax.set_xlabel("Cumulative training episodes")
        ax.set_ylim(0, 90)
        ax.set_title(f"{scale} trains (from scratch)")
        ax.grid(alpha=0.25, ls=":")
        ax.legend(loc="lower right", fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Arrival on-time (%)")
    fig.tight_layout()
    fig.savefig(args.scal_dir / "conv_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.scal_dir / 'conv_curves.png'}")

    # ---- Fig 2: learned-component convergence (Phase B vs flat, re-zeroed) ----
    # Hierarchical's deployable learned policy exists only after Phase A; Phase B
    # trains the L3 worker from scratch. Re-zero both methods to a common axis of
    # "episodes of learned-RL training" so the learned L3's fast convergence is
    # visible without the SA-backed Phase A inflating the curve.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, scale in zip(axes, [15, 25]):
        hier_eps, hier_vals, hier_pb = curves[("hier", scale)]
        flat_eps, flat_vals, _ = curves[("flat", scale)]
        hi = min(range(len(hier_eps)), key=lambda i: abs(hier_eps[i] - hier_pb))
        b_eps = np.array(hier_eps[hi:]) - hier_pb
        b_vals = hier_vals[hi:]
        band_plot(ax, flat_eps, flat_vals, C_FLAT, "flat MAPPO (all learned)", 400)
        band_plot(ax, b_eps, b_vals, C_HIER,
                  "hierarchical learned L3 (Phase B)", 200)
        row = next(r for r in conv_rows if r["method"] == "hier" and r["scale"] == scale)
        if row["phaseB_episodes_to_50pct"] != "":
            pb = int(row["phaseB_episodes_to_50pct"]) - hier_pb
            ax.axvline(pb, color="#1f77b4", ls=":", lw=1.2)
            ax.text(pb + 30, 62, f"L3 ≥{TARGET:.0%} @ {pb} Phase-B ep",
                    fontsize=7.5, color="#1f77b4")
        ax.axhline(TARGET * 100, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("Episodes of learned-RL training (Phase B; flat: all)")
        ax.set_ylim(0, 90)
        ax.set_title(f"{scale} trains: learned-component convergence")
        ax.grid(alpha=0.25, ls=":")
        ax.legend(fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Arrival on-time (%)")
    fig.tight_layout()
    fig.savefig(args.scal_dir / "conv_phaseB.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.scal_dir / 'conv_phaseB.png'}")

    # ---- Fig: transfer quality curve 10..30 + from-scratch markers ----
    modes = ["learned", "sa", "sa-online", "cruise", "flat"]
    x = np.array(SCALES)
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in modes:
        ys = []
        for s in SCALES:
            # primary = --scal-dir (output/9_4, P3-fixed, 10..30 self-contained);
            # optional legacy fallback for scales absent there.
            d = load_eval_ontime(args.scal_dir, s)
            if not d and args.legacy_scal_dir:
                d = load_eval_ontime(args.legacy_scal_dir, s)
            ys.append(d.get(mode, np.nan))
        ax.plot(x, ys, marker="o", label=mode, color=COLORS.get(mode, None), linewidth=1.8)
    # from-scratch converged markers
    for method, mcolor, msym in [("hier", "#c0392b", "^"), ("flat", "#6c3483", "s")]:
        for s in (15, 25):
            row = next(r for r in conv_rows if r["method"] == method and r["scale"] == s)
            ax.scatter([s], [row["final_roll_arr_ontime"] * 100], marker=msym,
                       color=mcolor, s=90, zorder=5,
                       label=f"{method} from-scratch" if s == 15 else None)
    ax.set_xlabel("trains (scale)")
    ax.set_ylabel("arrival ontime (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}t" for s in SCALES])
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    ax.set_title("Scalability: transfer ontime% vs scale (incident 3x60) + from-scratch markers")
    fig.tight_layout()
    fig.savefig(args.scal_dir / "scalability_quality.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.scal_dir / 'scalability_quality.png'}")
    print(f"saved {args.scal_dir / 'conv_summary.csv'}")


if __name__ == "__main__":
    main()
