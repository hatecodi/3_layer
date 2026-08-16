#!/usr/bin/env python3
"""Scalability paper figures + convergence analysis from training/eval CSVs (CSV-first).

Reads:
  - training_metrics.csv from from-scratch runs (per-episode `arr_ontime` column,
    added to run_flat_mappo.py / run_train_hierarchical.py): the convergence data.
  - eval ontime.csv per scale (eval_protocol --out-dir): the transfer quality curve.

Writes into a dated out-dir (default output/8_16/scalability):
  - conv_summary.csv      — (scale, method) x {episodes_to_50pct, final_roll_arr_ontime}
  - conv_curves.png       — 2-panel (15t, 25t): rolling arr-ontime vs episodes, hier vs flat
  - scalability_quality.png — x=N(10..30), transfer ontime% per mode + from-scratch markers

Convergence definition (fixed, comparable across methods):
  rolling arr-ontime over a 50-episode window; "episodes to 50%" = first episode whose
  rolling value stays >= 0.50 for the following 50 episodes (sustained). None if never.

Usage:
    python scripts/plot_conv_scalability.py [--scal-dir output/8_16/scalability]
        [--train-root output/8_16/training] [--legacy-scal-dir output/8_14/scalability]
        [--flat-15t PATH] [--flat-25t PATH] [--hier-15t PATH] [--hier-25t PATH]
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

FONT = {"font.family": "serif", "font.size": 12, "mathtext.fontset": "dejavuserif"}
plt.rcParams.update(FONT)
COLORS = {"learned": "#c0392b", "sa": "#2874a6", "sa-online": "#148f77",
          "sa-nofc": "#b7950b", "cruise": "#7d6608", "flat": "#6c3483"}
WIN = 50
TARGET = 0.50
SCALES = [10, 15, 20, 25, 30]


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
    ap.add_argument("--scal-dir", type=Path, default=Path("output/8_16/scalability"))
    ap.add_argument("--legacy-scal-dir", type=Path, default=Path("output/8_14/scalability"))
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

    # ---- convergence summary ----
    conv_rows = []
    curves = {}
    for method, scales in runs.items():
        for scale, csv_path in scales.items():
            eps, vals, phase_b, is_heldout = resolve_curve(csv_path)
            if is_heldout:
                final_roll = float(vals[-1])          # last checkpoint (already 12-ep eval)
                e2t = episodes_to_sustained_cp(vals, eps, TARGET, ncp=3)
                roll = np.copy(vals)                   # sparse: no 50-ep rolling
                WIN_PLOT = 1
            else:
                roll = rolling(vals, WIN)
                final_roll = float(roll[-1])
                e2t = episodes_to_sustained(roll, eps, WIN, TARGET)
                WIN_PLOT = WIN
            # Phase-B-only: when does the LEARNED L3 (not the SA-pretrain) first
            # sustain >= target? Hierarchical only; flat has no phases.
            e2t_b = None
            if phase_b is not None:
                idx = min(range(len(eps)), key=lambda i: abs(eps[i] - phase_b))
                e2t_b = (episodes_to_sustained_cp(vals, eps, TARGET, ncp=3, start_idx=idx)
                         if is_heldout else
                         episodes_to_sustained(roll, eps, WIN, TARGET, start_idx=idx))
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
            curves[(method, scale)] = (eps, roll, phase_b)

    with open(args.scal_dir / "conv_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scale", "method", "csv", "source",
                                          "final_roll_arr_ontime",
                                          "episodes_to_50pct",
                                          "phaseB_episodes_to_50pct"])
        w.writeheader()
        w.writerows(conv_rows)

    # ---- Fig 1: full-pipeline convergence curves, Phase A/B annotated ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, scale in zip(axes, [15, 25]):
        hier_eps, hier_roll, hier_pb = curves[("hier", scale)]
        flat_eps, flat_roll, _ = curves[("flat", scale)]
        if hier_pb is not None:
            ax.axvspan(0, hier_pb, color="#f5b7b1", alpha=0.30,
                       label="Phase A (L2 + SA-L3)")
            ax.axvline(hier_pb, color="#c0392b", ls="--", lw=1.0)
        ax.plot(flat_eps, flat_roll, color="#6c3483", label="flat MAPPO", linewidth=1.8)
        ax.plot(hier_eps, hier_roll, color="#c0392b",
                label="hierarchical (Phase A+B)", linewidth=1.8)
        ax.axhline(TARGET, color="gray", ls="--", lw=0.8)
        ax.text(hier_eps[-1] * 0.02, TARGET + 0.012, f"{TARGET:.0%}",
                color="gray", fontsize=9)
        ax.set_xlabel("cumulative training episodes")
        ax.set_ylim(0, 0.9)
        ax.set_title(f"{scale} trains (from scratch)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="center right")
    axes[0].set_ylabel("rolling arrival ontime (50-ep)")
    fig.tight_layout()
    fig.savefig(args.scal_dir / "conv_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.scal_dir / 'conv_curves.png'}")

    # ---- Fig 2: learned-component convergence (Phase B vs flat, re-zeroed) ----
    # Hierarchical's deployable learned policy exists only after Phase A; Phase B
    # trains the L3 worker from scratch. Re-zero both methods to a common axis of
    # "episodes of learned-RL training" so the learned L3's fast convergence is
    # visible without the SA-backed Phase A inflating the curve.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, scale in zip(axes, [15, 25]):
        hier_eps, hier_roll, hier_pb = curves[("hier", scale)]
        flat_eps, flat_roll, _ = curves[("flat", scale)]
        hi = min(range(len(hier_eps)), key=lambda i: abs(hier_eps[i] - hier_pb))
        b_eps = np.array(hier_eps[hi:]) - hier_pb
        b_roll = hier_roll[hi:]
        ax.plot(flat_eps, flat_roll, color="#6c3483",
                label="flat MAPPO (all learned)", linewidth=1.8)
        ax.plot(b_eps, b_roll, color="#c0392b",
                label="hierarchical learned L3 (Phase B)", linewidth=1.8)
        row = next(r for r in conv_rows if r["method"] == "hier" and r["scale"] == scale)
        if row["phaseB_episodes_to_50pct"] != "":
            pb = int(row["phaseB_episodes_to_50pct"]) - hier_pb
            ax.axvline(pb, color="#c0392b", ls=":", lw=1.2)
            ax.text(pb + 20, 0.58, f"learned L3 ≥{TARGET:.0%} @ {pb} Phase-B eps",
                    fontsize=8, color="#c0392b")
        ax.axhline(TARGET, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("episodes of learned-RL training (Phase B; flat: all)")
        ax.set_ylim(0, 0.9)
        ax.set_title(f"{scale} trains: learned-component convergence")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("rolling arrival ontime (50-ep)")
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
            # 10/15/20 evals live in output/8_14; 25/30 (new today) in output/8_16
            d = load_eval_ontime(args.legacy_scal_dir, s)
            d = d or load_eval_ontime(args.scal_dir, s)
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
