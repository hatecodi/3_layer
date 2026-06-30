#!/usr/bin/env python3
"""5-station railway physical layout — to-scale topology diagram."""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Dimensions (meters)
# ---------------------------------------------------------------------------
PLATFORM = 250      # platform length
THROAT = 80         # throat (switch area)
APPROACH = 100      # approach/depart connector
CORRIDOR_AB = 1200
CORRIDOR_BC = 2400
CORRIDOR_CD = 1200
CORRIDOR_DE = 1200

# Station geometry (west to east):
#   approach_w + throat_w + platform + throat_e + approach_e
STATION_WIDTH = APPROACH + THROAT + PLATFORM + THROAT + APPROACH  # = 610m

# Cumulative x positions
# Station A west edge at x=0
x = 0
station_x = {}
corridor_spans = []

for i, st in enumerate(["A", "B", "C", "D", "E"]):
    station_x[st] = x  # west edge of station
    # corridor after this station (except last)
    if i < 4:
        corridor_len = [CORRIDOR_AB, CORRIDOR_BC, CORRIDOR_CD, CORRIDOR_DE][i]
        next_st = ["B", "C", "D", "E"][i]
        corridor_start = x + STATION_WIDTH
        corridor_end = corridor_start + corridor_len
        corridor_spans.append((st, next_st, corridor_len, corridor_start, corridor_end))
        x = corridor_end  # next station starts after corridor
    else:
        x += STATION_WIDTH

total_len = x

# ---------------------------------------------------------------------------
# Platform vertical positions
# ---------------------------------------------------------------------------
Y_CENTER = 0
Y_GAP = 1.0   # gap between platform tracks
PLATFORM_COUNTS = {"A": 4, "B": 4, "C": 4, "D": 2, "E": 4}

COLORS = {
    "A": "#e3f2fd", "B": "#e8f5e9", "C": "#fff3e0", "D": "#fce4ec", "E": "#f3e5f5",
    "platform": "#bdbdbd", "throat": "#ffcc80", "corr_up": "#90caf9", "corr_dn": "#ef9a9a",
    "approach": "#eeeeee",
}

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(24, 8))
fig.suptitle("Five-Station Railway Network  —  A-B-C-D-E (to scale)",
             fontsize=14, fontweight="bold")

for st in ["A", "B", "C", "D", "E"]:
    sx = station_x[st]
    n = PLATFORM_COUNTS[st]
    px = sx + APPROACH + THROAT  # platform start x

    # ---- station background ----
    ax.axvspan(sx, sx + STATION_WIDTH, facecolor=COLORS[st], alpha=0.25, zorder=0)

    # west approach
    ax.axvspan(sx, sx + APPROACH, facecolor=COLORS["approach"], alpha=0.6, zorder=1)
    ax.text(sx + APPROACH / 2, -4.2, "approach", fontsize=6, ha="center", color="#999")

    # west throat
    tx = sx + APPROACH
    ax.axvspan(tx, tx + THROAT, facecolor=COLORS["throat"], alpha=0.35, zorder=1)
    ax.text(tx + THROAT / 2, n * Y_GAP / 2 + 1.0, "throat\nin", fontsize=5, ha="center", color="#e65100")

    # platform area
    ax.axvspan(px, px + PLATFORM, facecolor=COLORS["platform"], alpha=0.2, zorder=1)

    # draw platform tracks
    for p in range(n):
        py = Y_CENTER + (p - (n - 1) / 2) * Y_GAP
        ax.plot([px, px + PLATFORM], [py, py], "-", color="#555", linewidth=1.2, zorder=2)
        ax.text(px + PLATFORM / 2, py + 0.12, f"{st}_p{p + 1}", fontsize=5, ha="center", va="bottom", color="#555")

    # throat → platform connections (simplified)
    for p in range(n):
        py = Y_CENTER + (p - (n - 1) / 2) * Y_GAP
        ax.plot([tx, px], [Y_CENTER, py], "-", color="#90caf9", linewidth=0.4, alpha=0.3, zorder=0)
        ax.plot([px + PLATFORM, px + PLATFORM + THROAT], [py, Y_CENTER], "-",
                color="#90caf9", linewidth=0.4, alpha=0.3, zorder=0)

    # east throat
    ex = px + PLATFORM
    ax.axvspan(ex, ex + THROAT, facecolor=COLORS["throat"], alpha=0.35, zorder=1)
    ax.text(ex + THROAT / 2, n * Y_GAP / 2 + 1.0, "throat\nout", fontsize=5, ha="center", color="#e65100")

    # east approach
    ax.axvspan(ex + THROAT, sx + STATION_WIDTH, facecolor=COLORS["approach"], alpha=0.6, zorder=1)

    # station label
    ax.text(sx + STATION_WIDTH / 2, n * Y_GAP / 2 + 1.5,
            f"{st} ({n}p)", fontsize=13, fontweight="bold", ha="center", color="#333")

    # dwell time label
    dwells = {"A": 60, "B": 30, "C": 60, "D": 30, "E": 30}
    ax.text(sx + STATION_WIDTH / 2, n * Y_GAP / 2 + 0.7,
            f"dwell={dwells[st]}s", fontsize=7, ha="center", color="#888")

    # D station bottleneck highlight
    if st == "D":
        rect = plt.Rectangle((sx - 20, -3.5), STATION_WIDTH + 40, n * Y_GAP + 5,
                              fill=False, edgecolor="red", linewidth=2.5,
                              linestyle="--", zorder=5)
        ax.add_patch(rect)
        ax.text(sx + STATION_WIDTH / 2, n * Y_GAP / 2 + 2.8,
                "BOTTLENECK", fontsize=10, ha="center", color="red", fontweight="bold")

# Corridors between stations
for st_a, st_b, corr_len, cs, ce in corridor_spans:
    mid = (cs + ce) / 2

    # UP track (upper)
    ax.plot([cs, ce], [3.5, 3.5], "-", color=COLORS["corr_up"], linewidth=2.5, alpha=0.7)
    ax.text(mid, 4.0, f"{st_a}→{st_b}: {corr_len}m ({corr_len // 13}s @30m/s)",
            fontsize=8, ha="center", color="#1565c0")
    ax.annotate("", xy=(cs, 3.5), xytext=(ce, 3.5),
                arrowprops=dict(arrowstyle="->", color=COLORS["corr_up"], lw=1.5))

    # DOWN track (lower)
    ax.plot([cs, ce], [-3.5, -3.5], "-", color=COLORS["corr_dn"], linewidth=2.5, alpha=0.7)
    ax.text(mid, -4.0, f"{st_b}→{st_a}: {corr_len}m ({corr_len // 13}s @30m/s)",
            fontsize=8, ha="center", color="#c62828")
    ax.annotate("", xy=(ce, -3.5), xytext=(cs, -3.5),
                arrowprops=dict(arrowstyle="->", color=COLORS["corr_dn"], lw=1.5))

# Speed disturbance annotation (time-limited)
# Show the A→B corridor with disturbance zone
cs_ab = corridor_spans[0][3]  # A→B corridor start
ce_ab = corridor_spans[0][4]
mid_ab = (cs_ab + ce_ab) / 2
rect = plt.Rectangle((cs_ab + 50, 2.5), ce_ab - cs_ab - 100, 2.0,
                      facecolor="yellow", alpha=0.25, edgecolor="orange",
                      linewidth=1.5, linestyle="--")
ax.add_patch(rect)
ax.text(mid_ab, 3.5, "DISTURBANCE ZONE\nt=[200-600s] speed×0.5",
        fontsize=7, ha="center", color="orange", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

# ---- Train length scale ----
ty = -5.5
train_info = [("G", 80, "#e41a1c"), ("D", 100, "#377eb8"),
              ("Z", 140, "#999"), ("K", 160, "#ff7f00")]
tx = 200
for label, length, color in train_info:
    r = plt.Rectangle((tx, ty), length, 0.6, facecolor=color, alpha=0.8, edgecolor="black")
    ax.add_patch(r)
    ax.text(tx + length / 2, ty + 0.3, f"{label} {length}m", fontsize=8,
            ha="center", va="center", color="white", fontweight="bold")
    tx += length + 100

r = plt.Rectangle((tx, ty), PLATFORM, 0.6, facecolor="gray", alpha=0.3, edgecolor="black")
ax.add_patch(r)
ax.text(tx + PLATFORM / 2, ty + 0.3, f"Platform {PLATFORM}m", fontsize=8, ha="center")
ax.text(tx + 30, ty - 0.5, "Train lengths vs Platform", fontsize=7, color="#555")

# Distance ruler
ticks = [0]
for i, (_, _, clen, _, _) in enumerate(corridor_spans):
    ticks.append(ticks[-1] + STATION_WIDTH + clen)

ax.set_xlim(-50, total_len + 100)
ax.set_ylim(-7, 6)
ax.set_aspect("equal")
ax.axis("off")

# Legend
legend = [
    mpatches.Patch(color=COLORS["platform"], alpha=0.4, label=f"Platform ({PLATFORM}m)"),
    mpatches.Patch(color=COLORS["throat"], alpha=0.5, label=f"Throat ({THROAT}m)"),
    mpatches.Patch(color=COLORS["corr_up"], alpha=0.6, label="UP corridor (30m/s)"),
    mpatches.Patch(color=COLORS["corr_dn"], alpha=0.6, label="DOWN corridor (30m/s)"),
    mpatches.Patch(color="orange", alpha=0.3, label="Time-limited disturbance (yellow zone)"),
]
ax.legend(handles=legend, loc="lower right", fontsize=8, ncol=2)

# Save
out = _PROJECT / "output" / "five_station" / "network_layout.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
print(f"\nDistances from A:")
cumul = 0
for i, st in enumerate(["A", "B", "C", "D", "E"]):
    print(f"  {st} at {cumul:.0f}m (center)")
    if i < 4:
        cumul += STATION_WIDTH + [CORRIDOR_AB, CORRIDOR_BC, CORRIDOR_CD, CORRIDOR_DE][i]
print(f"  Total length: {cumul:.0f}m")
print(f"\nTrain lengths: G=80m, D=100m, Z=140m, K=160m → all fit on 250m platform")
print(f"Throat transit (20m/s): G=4s, D=5s, Z=7s, K=8s (currently 20s fixed)")

plt.close("all")
