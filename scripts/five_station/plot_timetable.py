#!/usr/bin/env python3
"""
Train running chart (列车运行图) — planned vs actual.

Shows:
  - Time-space diagram: planned schedule (dashed) vs heuristic actual (solid)
  - Station dwell comparison: planned vs actual per train per station
  - Cumulative delay per train
  - Speed profile between stations

Usage:
    python multistation/scripts/plot_timetable.py [--save path.png]
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT))

import matplotlib
matplotlib.use("Agg")  # non-interactive
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from scripts.five_station.run_heuristic import load_multistation_cfg
from src.env.five_station_env import FiveStationEnv
from src.env.disturbance import CorridorDisturbance
from src.control.dispatch_bounds import DispatchBounds

# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_actual(cfg, corridor_disturbances, initial_delays):
    """Run heuristic episode, return dict: train_id -> [(station, arr_actual, dep_actual), ...]"""
    env = FiveStationEnv(
        cfg,
        corridor_disturbances=corridor_disturbances,
        initial_train_delays=initial_delays,
    )
    obs, info = env.reset()
    actual: dict[str, list[tuple[str, float, float]]] = {tid: [] for tid in cfg.trains}

    while True:
        obs, reward, terminated, truncated, info = env.step({})
        for g in info.get("granted_windows", []):
            j = env._journeys.get(g.train_id)
            if j and not j.is_done:
                st = j.current_station
                actual[g.train_id].append((st, g.W_in[0], g.W_out[0]))
        if terminated or truncated:
            break

    # Sort each train's path by arrival time
    for tid in actual:
        actual[tid].sort(key=lambda x: x[1])
    return actual, env._step_idx


def collect_planned(cfg):
    """Extract planned schedule from timetable."""
    l1 = DispatchBounds(cfg)
    planned: dict[str, list[tuple[str, float, float]]] = {}
    for tid, train in cfg.trains.items():
        path = []
        stations = ["A", "B", "C", "D", "E"] if train.direction == "up" else ["E", "D", "C", "B", "A"]
        for st in stations:
            b = l1.for_train(tid, st)
            if b.t_arr_plan != 0 or b.t_dep_plan != 0:
                path.append((st, b.t_arr_plan, b.t_dep_plan))
        if path:
            path.sort(key=lambda x: x[1])
            planned[tid] = path
    return planned


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_running_chart(ax, planned, actual, cfg):
    """Main time-space diagram: y=stations (km), x=time (s)"""
    # Station positions (cumulative corridor distance)
    corridor_lengths = {"A_B": 1.2, "B_C": 2.4, "C_D": 1.2, "D_E": 1.2}
    station_km = {"A": 0.0}
    cumulative = 0.0
    for st in ["B", "C", "D", "E"]:
        prev = ["A", "B", "C", "D"][["B", "C", "D", "E"].index(st)]
        cumulative += corridor_lengths.get(f"{prev}_{st}", 1.2)
        station_km[st] = cumulative

    # Colors per train
    train_colors = {
        "K641": "#e41a1c", "D133": "#377eb8", "G5050": "#4daf4a",
        "K667": "#984ea3", "K999": "#ff7f00",
        "G1408": "#a65628", "G5060": "#f781bf", "Z386": "#999999",
        "K1666": "#66c2a5",
    }

    # Draw planned (dashed thin lines)
    for tid, path in planned.items():
        if not path:
            continue
        times = []
        kms = []
        for st, arr, dep in path:
            km = station_km.get(st, 0)
            # Arrival point
            times.append(arr)
            kms.append(km)
            # Departure point (horizontal segment for dwell)
            times.append(dep)
            kms.append(km)

        color = train_colors.get(tid, "#000000")
        ax.plot(times, kms, "--", color=color, linewidth=0.8, alpha=0.4)

    # Draw actual (solid thick lines)
    handles = []
    for tid, path in actual.items():
        if not path:
            continue
        times = []
        kms = []
        for st, arr, dep in path:
            km = station_km.get(st, 0)
            times.append(arr)
            kms.append(km)
            times.append(dep)
            kms.append(km)

        color = train_colors.get(tid, "#000000")
        line, = ax.plot(times, kms, "-", color=color, linewidth=1.8, alpha=0.9)
        handles.append(line)

        # Label at first station
        if path:
            first_st, first_arr, _ = path[0]
            ax.annotate(tid, (first_arr, station_km[first_st]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=6, color=color, fontweight="bold")

    # Station horizontal lines
    for st, km in station_km.items():
        ax.axhline(y=km, color="gray", linewidth=0.3, linestyle=":")
        ax.text(-80, km, f"{st} ({len(cfg.stations[st].platforms)}p)",
                fontsize=7, verticalalignment="center", color="gray")

    # D station bottleneck highlight
    d_km = station_km["D"]
    ax.axhspan(d_km - 0.3, d_km + 0.3, facecolor="red", alpha=0.05)

    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Distance from A (km)", fontsize=10)
    ax.set_title("Train Running Chart — Planned (dashed) vs Heuristic Actual (solid)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(-100, 1600)
    ax.set_ylim(-0.2, max(station_km.values()) + 0.3)

    # Legend
    ax.legend(handles, [h.get_label() for h in handles] if False else
              [tid for tid in actual.keys() if actual[tid]],
              fontsize=6, loc="upper left", ncol=2)

    # D station annotation
    ax.annotate("BOTTLENECK\n(2 platforms)", xy=(750, d_km), fontsize=8,
                color="red", ha="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))


def plot_delay_bars(ax, planned, actual, cfg):
    """Horizontal bar chart: delay per train per station."""
    l1 = DispatchBounds(cfg)
    train_ids = sorted(actual.keys())

    data = []
    labels = []
    colors_bar = []
    train_colors = {
        "K641": "#e41a1c", "D133": "#377eb8", "G5050": "#4daf4a",
        "K667": "#984ea3", "K999": "#ff7f00",
        "G1408": "#a65628", "G5060": "#f781bf", "Z386": "#999999",
        "K1666": "#66c2a5",
    }

    for tid in train_ids:
        path = actual.get(tid, [])
        for st, arr, dep in path:
            b = l1.for_train(tid, st)
            if b.t_dep_plan > 0:
                dep_delay = max(0, dep - b.t_dep_plan)
                arr_delay = max(0, arr - b.t_arr_plan)
                if dep_delay > 10 or arr_delay > 10:
                    data.append(dep_delay)
                    labels.append(f"{tid}@{st}")
                    colors_bar.append(train_colors.get(tid, "#888"))

    if not data:
        ax.text(0.5, 0.5, "No significant delays", ha="center", va="center",
                transform=ax.transAxes)
        return

    y_pos = range(len(data))
    ax.barh(y_pos, data, color=colors_bar, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Departure delay (s)", fontsize=9)
    ax.set_title("Station Departure Delays (>10s)", fontsize=11, fontweight="bold")
    ax.invert_yaxis()


def plot_cumulative_delay(ax, planned, actual, cfg):
    """Cumulative delay per train over stations visited."""
    l1 = DispatchBounds(cfg)
    train_ids = sorted(actual.keys())
    train_colors = {
        "K641": "#e41a1c", "D133": "#377eb8", "G5050": "#4daf4a",
        "K667": "#984ea3", "K999": "#ff7f00",
        "G1408": "#a65628", "G5060": "#f781bf", "Z386": "#999999",
        "K1666": "#66c2a5",
    }

    for tid in train_ids:
        path = actual.get(tid, [])
        if not path:
            continue
        cum_delay = 0.0
        x = [0]
        y = [0]
        station_idx = 0
        for st, arr, dep in path:
            b = l1.for_train(tid, st)
            if b.t_dep_plan > 0:
                cum_delay += max(0, dep - b.t_dep_plan)
            station_idx += 1
            x.append(station_idx)
            y.append(cum_delay)

        color = train_colors.get(tid, "#000")
        ax.plot(x, y, "-o", color=color, linewidth=1.5, markersize=4, label=tid)

    ax.set_xlabel("Station index in route", fontsize=9)
    ax.set_ylabel("Cumulative delay (s)", fontsize=9)
    ax.set_title("Cumulative Delay Along Route", fontsize=11, fontweight="bold")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)


def plot_speed_profile(ax, planned, actual, cfg):
    """Average speed between stations (planned vs actual)."""
    corridor_lengths = {"A_B": 1.2, "B_C": 2.4, "C_D": 1.2, "D_E": 1.2}

    train_ids = sorted(actual.keys())
    # Pick 3 representative trains
    representative = ["K641", "D133", "G1408"]
    train_colors = {"K641": "#e41a1c", "D133": "#377eb8", "G1408": "#a65628"}

    x_offset = 0
    all_labels = []
    all_positions = []

    for tid in representative:
        path_act = actual.get(tid, [])
        path_plan = planned.get(tid, [])
        if len(path_act) < 2:
            continue

        for i in range(len(path_act) - 1):
            st_cur = path_act[i][0]
            st_nxt = path_act[i + 1][0]
            corr_key = f"{st_cur}_{st_nxt}"
            if corr_key not in corridor_lengths:
                corr_key = f"{st_nxt}_{st_cur}"
            dist = corridor_lengths.get(corr_key, 1.2) * 1000  # meters

            dep_act = path_act[i][2]
            arr_act = path_act[i + 1][1]
            travel_act = max(1, arr_act - dep_act)
            speed_act = dist / travel_act  # m/s

            # Find planned segment
            speed_plan = 0
            for j in range(len(path_plan) - 1):
                if path_plan[j][0] == st_cur and path_plan[j + 1][0] == st_nxt:
                    dep_plan = path_plan[j][2]
                    arr_plan = path_plan[j + 1][1]
                    travel_plan = max(1, arr_plan - dep_plan)
                    speed_plan = dist / travel_plan
                    break

            ax.bar(x_offset + 0.1, speed_act, 0.35,
                   color=train_colors.get(tid, "#888"), alpha=0.8)
            ax.bar(x_offset + 0.45, speed_plan, 0.35,
                   color="gray", alpha=0.4)
            all_labels.append(f"{tid}\n{st_cur}→{st_nxt}")
            all_positions.append(x_offset + 0.3)
            x_offset += 1

    ax.set_xticks(all_positions)
    ax.set_xticklabels(all_labels, fontsize=5, rotation=45, ha="right")
    ax.set_ylabel("Speed (m/s)", fontsize=9)
    ax.set_title("Corridor Speed — Actual vs Planned", fontsize=11, fontweight="bold")
    ax.axhline(y=30, color="green", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.text(0, 30.5, "max 30 m/s", fontsize=6, color="green")

    # Legend
    p1 = mpatches.Patch(color="#377eb8", alpha=0.8, label="Actual")
    p2 = mpatches.Patch(color="gray", alpha=0.4, label="Planned")
    ax.legend(handles=[p1, p2], fontsize=7)


def plot_station_dwell(ax, planned, actual, cfg):
    """Planned vs actual dwell time per station."""
    l1 = DispatchBounds(cfg)

    stations = ["A", "B", "C", "D", "E"]
    x_offset = 0
    labels = []
    positions = []

    for tid in ["K641", "D133", "G1408"]:
        path_act = actual.get(tid, [])
        path_plan = planned.get(tid, [])
        plan_dict = {st: (arr, dep) for st, arr, dep in path_plan}
        act_dict = {st: (arr, dep) for st, arr, dep in path_act}

        for st in stations:
            if st in act_dict and st in plan_dict:
                _, dep_a = act_dict[st]
                arr_a, _ = act_dict[st]
                _, dep_p = plan_dict[st]
                arr_p, _ = plan_dict[st]

                dwell_act = max(0, dep_a - arr_a)
                dwell_plan = max(0, dep_p - arr_p)

                ax.bar(x_offset + 0.1, dwell_act, 0.35,
                       color="#377eb8", alpha=0.8)
                ax.bar(x_offset + 0.45, dwell_plan, 0.35,
                       color="gray", alpha=0.4)
                labels.append(f"{tid}@{st}")
                positions.append(x_offset + 0.3)
                x_offset += 1

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=5, rotation=45, ha="right")
    ax.set_ylabel("Dwell time (s)", fontsize=9)
    ax.set_title("Station Dwell — Actual vs Planned", fontsize=11, fontweight="bold")
    p1 = mpatches.Patch(color="#377eb8", alpha=0.8, label="Actual")
    p2 = mpatches.Patch(color="gray", alpha=0.4, label="Planned")
    ax.legend(handles=[p1, p2], fontsize=7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_multistation_cfg()
    corr = CorridorDisturbance("A_B_up", 200, 600, 0.5)
    initial_delays = {"K999": 60.0}

    actual, steps = collect_actual(cfg, [corr], initial_delays)
    planned = collect_planned(cfg)

    # Compute statistics
    l1 = DispatchBounds(cfg)
    total_delay = 0
    for tid, path in actual.items():
        for st, arr, dep in path:
            b = l1.for_train(tid, st)
            if b.t_dep_plan > 0:
                total_delay += max(0, dep - b.t_dep_plan)

    # ---- Figure ----
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Multi-Station Heuristic Baseline\n"
        f"9 trains, 5 stations, D bottleneck (2 platforms), "
        f"Corridor disturbance A→B [200-600s]\n"
        f"Total delay: {total_delay:.0f}s | Episode steps: {steps}",
        fontsize=11, fontweight="bold", y=0.98)

    # (1) Main running chart
    ax1 = fig.add_subplot(2, 3, (1, 3))  # span top row, 3 cols
    plot_running_chart(ax1, planned, actual, cfg)

    # (2) Delay bars
    ax2 = fig.add_subplot(2, 3, 4)
    plot_delay_bars(ax2, planned, actual, cfg)

    # (3) Cumulative delay
    ax3 = fig.add_subplot(2, 3, 5)
    plot_cumulative_delay(ax3, planned, actual, cfg)

    # (4) Speed profile
    ax4 = fig.add_subplot(2, 3, 6)
    plot_station_dwell(ax4, planned, actual, cfg)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "heuristic_running_chart.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    # Also save a higher-res D station detail
    fig2, ax = plt.subplots(figsize=(14, 5))
    # D station zoom
    station_km = {"A": 0.0, "B": 1.2, "C": 3.6, "D": 4.8, "E": 6.0}
    train_colors = {
        "K641": "#e41a1c", "D133": "#377eb8", "G5050": "#4daf4a",
        "K667": "#984ea3", "K999": "#ff7f00",
        "G1408": "#a65628", "G5060": "#f781bf", "Z386": "#999999",
        "K1666": "#66c2a5",
    }

    for tid, path in actual.items():
        if not path:
            continue
        # Only show segments around D station (C-D-E)
        st_filtered = [(s, a, d) for s, a, d in path if s in ("C", "D", "E")]
        if len(st_filtered) < 2:
            continue
        times, kms = [], []
        for s, a, d in st_filtered:
            km = station_km[s]
            times.extend([a, d])
            kms.extend([km, km])
        ax.plot(times, kms, "-o", color=train_colors.get(tid, "#000"),
                linewidth=2, markersize=5, label=tid)

    for s, km in station_km.items():
        if s in ("C", "D", "E"):
            ax.axhline(y=km, color="gray", linewidth=0.3, linestyle=":")
            ax.text(-80, km, f"{s}", fontsize=9, va="center")

    d_km = station_km["D"]
    ax.axhspan(d_km - 0.3, d_km + 0.3, facecolor="red", alpha=0.08)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance from A (km)")
    ax.set_title("D Station Bottleneck Detail — C-D-E Segment", fontweight="bold")
    ax.legend(fontsize=7, ncol=3)
    ax.set_xlim(200, 1300)
    ax.set_ylim(3.2, 6.4)

    detail_path = output_dir / "station_D_detail.png"
    fig2.savefig(detail_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {detail_path}")

    plt.close("all")


if __name__ == "__main__":
    main()
