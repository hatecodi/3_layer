#!/usr/bin/env python3
"""Decision-layer compute vs fleet size: HMAPPO vs SA vs FlatMAPPO.

Reads (CSV-first: nothing recomputed here) the WIDE table
    output/9.20/compute_time_scaling.csv
and plots exactly three of its columns:
    n_trains                -> x axis
    <METHOD>_ms_per_episode -> y axis, one line per METHOD
    (panel b derives SA/HMAPPO from those same two series)

BASIS: milliseconds the decision layer consumes to run ONE episode
(L2+L3 for HMAPPO/SA; the joint forwards for FlatMAPPO).  This is the intuitive
"compute cost per episode" number.

Honest reading -- put this in the caption:
  * All three are ~O(N), so the curves come out nearly PARALLEL: log-log slopes
    0.84 (HMAPPO) vs 1.02 (SA).  The advantage is a CONSTANT factor (~4x), shown
    explicitly in panel (b).  Do NOT claim it widens with scale.
  * FlatMAPPO is the CHEAPEST by ~25-45x.  It commits one joint action per pending
    train per step (~7-9 forwards/episode) and only ~21% of those reach the
    corridor's speed layer, so its low cost reflects doing strictly LESS decision
    work -- which is also why its on-time quality is far worse (section 6.3).
    Compute alone favours the baseline: always quote it WITH quality.

If you need a view where the curves visibly fan out, use the *_ms_per_section
columns from the same CSV instead: there HMAPPO is flat (log-log slope -0.02)
while SA rises (+0.12) -- HMAPPO's unit cost is scale-invariant, SA's is not.

Usage:
    python scripts/plot_l3_cost_scaling.py
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Requested colours: HMAPPO blue, SA yellow, FlatMAPPO green.
# NOTE on yellow: pure #FFFF00 is close to invisible on a white background and
# fails greyscale/colour-blind checks. "#FFC000" is the readable amber that still
# reads as yellow — swap to a darker gold (#E8A33D) if the venue is strict.
STYLE = {
    "hier": dict(color="#1f77b4", marker="o", ls="-",  lw=1.8, label="HMAPPO (hierarchical)"),
    "sa":   dict(color="#FFC000", marker="s", ls="-",  lw=1.8, label="SA (full re-solve per transit)"),
    "flat": dict(color="#2ca02c", marker="^", ls="-",  lw=1.8, label="FlatMAPPO (joint action)"),
}
ORDER = ["hier", "sa", "flat"]
# CSV column prefix per method (must match make_compute_scaling_csv.py)
DISPLAY = {"hier": "HMAPPO", "sa": "SA", "flat": "FlatMAPPO"}


BASIS = {
    # basis -> (CSV column suffix, y label, panel-a title, default out name)
    "episode": ("_ms_per_episode", "decision-layer cost per episode (ms)",
                "(a) decision cost per episode (L2+L3)",
                "fig_l3_cost_compare.png"),
    "section": ("_ms_per_section", "L3 decision cost per corridor section (ms)",
                "(a) per-decision-unit cost (L3 only)",
                "fig_l3_cost_per_section.png"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=PROJECT_ROOT / "output/9.20/compute_time_scaling.csv")
    ap.add_argument("--basis", choices=sorted(BASIS), default="episode",
                    help="episode: ms per episode (L2+L3, near-parallel curves). "
                         "section: ms per corridor section, L3 only (HMAPPO flat vs "
                         "SA rising -- the curves fan out).")
    ap.add_argument("--out", type=Path, default=None,
                    help="defaults to fig_l3_cost_compare.png / fig_l3_cost_per_section.png")
    args = ap.parse_args()

    suffix, ylabel, title_a, out_name = BASIS[args.basis]
    out = args.out or (PROJECT_ROOT / "output/9.20" / out_name)

    # Read the WIDE table so the plotted quantity is computed in exactly one
    # place (make_compute_scaling_csv.py).
    series = defaultdict(list)
    with open(args.csv, newline="") as f:
        for r in csv.DictReader(f):
            n = int(r["n_trains"])
            for method in ORDER:
                col = f"{DISPLAY[method]}{suffix}"
                if col in r and r[col] != "":
                    series[method].append((n, float(r[col])))

    plt.rcParams.update({
        "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))

    for method in ORDER:
        if method not in series:
            continue
        pts = sorted(series[method])
        ax.semilogy([p[0] for p in pts], [p[1] for p in pts], **STYLE[method])

    ax.set_xlabel("number of trains $N$")
    ax.set_ylabel(ylabel)
    ax.set_title(title_a, fontsize=10)
    ax.legend(fontsize=8, loc="center right", framealpha=0.9)

    # Log-log slopes: both SA and HMAPPO are ~O(N), so these lines ARE near
    # parallel -- that is the honest result, not a plotting defect.  The
    # divergence is a constant-factor one, so it is shown explicitly in (b).
    if "hier" in series and "sa" in series:
        h = dict(series["hier"]); s = dict(series["sa"])
        common = sorted(set(h) & set(s))

        def slope(d):
            xs = sorted(d)
            return (math.log(d[xs[-1]] / d[xs[0]])
                    / math.log(xs[-1] / xs[0]))
        ax.annotate(
            f"log-log slope\nHMAPPO {slope(h):.2f}   SA {slope(s):.2f}",
            xy=(0.03, 0.28), xycoords="axes fraction", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7",
                      ec="#bbbbbb", lw=0.6))

        ratios = [s[n] / h[n] for n in common]
        ax2.plot(common, ratios, color="#8e44ad", marker="D", ls="-", lw=1.8)
        ax2.set_xlabel("number of trains $N$")
        ax2.set_ylabel("SA / HMAPPO decision-cost ratio")
        ax2.set_title("(b) the advantage is a constant factor\n(ratio grows only "
                      f"{100*(ratios[-1]/ratios[0]-1):.0f}% over {common[0]}\u2013{common[-1]}t)",
                      fontsize=10)
        ax2.set_ylim(bottom=0.0, top=max(ratios) * 1.14)  # headroom for the labels
        for n, r in zip(common, ratios):
            ax2.annotate(f"{r:.2f}$\\times$", (n, r), textcoords="offset points",
                         xytext=(0, 7), ha="center", fontsize=7.5)
        ax2.annotate("Do NOT claim this widens fast:\nratios are near-flat by "
                     "construction ($\\propto N$ vs $\\propto N$)",
                     xy=(0.03, 0.10), xycoords="axes fraction", fontsize=7.5,
                     bbox=dict(boxstyle="round,pad=0.3", fc="#fff8e1",
                               ec="#e0c060", lw=0.6))

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"saved {out}")
    for method in ORDER:
        if method in series:
            pts = sorted(series[method])
            print(f"  {method:>5}: " + "  ".join(f"{n}t={v:.2f}ms" for n, v in pts))


if __name__ == "__main__":
    main()
