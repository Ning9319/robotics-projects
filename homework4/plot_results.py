#!/usr/bin/env python3
"""Plot the two HW4 report execution figures from final CSV + console logs.

Run from the robotics-projects repository root:

    python3 homework4/plot_results.py

Default output directory: homework4/output/
    shortest_executed.png
    safety_executed.png


This script is offline-only: it does not import ROS, access sensors, or drive motors.
Required: matplotlib, PyYAML; uses existing homework4/geometry.py and planning.py.

IMPORTANT: the terminal stop (black X) is parsed from the GOAL REACHED
line of the corresponding .log, never substituted with the final CSV sample.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # render on headless rover, no desktop needed
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import yaml

from geometry import expanded_rect
from planning import World, max_clearance_grid, shortest_visibility

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "output"

# Final submissions matched to HW4 report.
FINAL_RUNS = {
    "shortest": "shortest_drive_20261123_010014",
    "safety": "safety_drive_20261123_004538",
}
# Guard only: the position is always READ FROM .log; these coordinates check
# that the log matches the result (to its console's 1 mm precision).
REPORT_TERMINAL_POSITIONS = {
    "shortest": (0.460, 2.033),
    "safety": (0.462, 2.031),
}
STOP_RE = re.compile(
    r"GOAL REACHED\s*\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*"
    r"([+-]?\d+(?:\.\d+)?)\s*\)"
)

COLOR = {
    "field": "#36404b",
    "obstacle": "#e7a18c",
    "obstacle_edge": "#9e5149",
    "cspace": "#c15b55",
    "online": "#b19b6d",
    "graph": "#c4cfd9",
    "shortest": "#2a65a3",
    "safety": "#bc6d12",
    "executed": "#208581",
    "start": "#188455",
    "goal": "#783e98",
    "stop": "#242b35",
}


def load_samples(stem: str) -> list[tuple[float, float]]:
    path = DATA_DIR / (stem + ".csv")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 2:
        raise ValueError(f"Not enough trajectory records in {path}")
    return [(float(row["x_m"]), float(row["y_m"])) for row in rows]


def load_final_pose(stem: str, expected: tuple[float, float]) -> tuple[float, float]:
    log_path = DATA_DIR / (stem + ".log")
    contents = log_path.read_text(encoding="utf-8", errors="replace")
    matches = STOP_RE.findall(contents)
    if len(matches) != 1:
        raise ValueError(f"Expected one GOAL REACHED (x, y) in {log_path}; found {len(matches)}")
    xy = (float(matches[0][0]), float(matches[0][1]))
    if math.dist(xy, expected) > 0.00051:
        raise ValueError(
            f"{log_path.name}: console stop {xy} does not match final report "
            f"position {expected}. Check the run files before regenerating figures."
        )
    return xy


def add_base(ax, world: World, *, show_execution: bool) -> None:
    s = world.side
    ax.add_patch(Rectangle((0, 0), s, s, facecolor="none",
                           edgecolor=COLOR["field"], lw=2.6, zorder=1))
    x0, y0, x1, y1 = world.raw_obstacle
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                           facecolor=COLOR["obstacle"],
                           edgecolor=COLOR["obstacle_edge"], lw=1.8, zorder=2))
    x0, y0, x1, y1 = world.inflated
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                           fill=False, edgecolor=COLOR["cspace"],
                           lw=2.0, ls="--", zorder=3, label="C-space"))
    if show_execution:
        rect = expanded_rect(world.raw_obstacle, world.robot_radius + 0.02)
        x0, y0, x1, y1 = rect
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               fill=False, edgecolor=COLOR["online"],
                               lw=1.6, ls=(0, (1, 3)), zorder=2.8,
                               label="Execution limit"))
    ax.set_xlim(-0.08, s + 0.08)
    ax.set_ylim(-0.08, s + 0.08)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)", fontsize=13)
    ax.set_ylabel("y (m)", fontsize=13)
    ticks = [i * 0.4 for i in range(7)]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=12)
    ax.grid(True, lw=0.6, alpha=0.20, zorder=0)


def draw_one(name: str, world: World, output_dir: Path) -> Path:
    stem = FINAL_RUNS[name]
    points = load_samples(stem)
    terminal = load_final_pose(stem, REPORT_TERMINAL_POSITIONS[name])
    if name == "shortest":
        planned, info = shortest_visibility(world)
    else:
        planned, info = max_clearance_grid(world)
    path_len = sum(math.dist(a, b) for a, b in zip(planned, planned[1:]))

    fig, ax = plt.subplots(figsize=(7.4, 7.11), dpi=155)
    add_base(ax, world, show_execution=(name == "shortest"))

    if name == "shortest":
        
        for u, v in info["edges"]:
            ax.plot([u[0], v[0]], [u[1], v[1]], color=COLOR["graph"],
                    lw=1.15, zorder=1.8)

    px, py = zip(*planned)
    cx, cy = zip(*points)
    ax.plot(px, py, "--", color=COLOR[name], lw=2.8,
            label="Planned", zorder=4)
    if name == "safety":
        ax.scatter(px[1:-1], py[1:-1], s=32, color=COLOR[name], zorder=5)

    ax.plot(cx, cy, "-", color=COLOR["executed"], lw=2.7,
            label="CSV poses", zorder=6)
    ax.scatter(cx, cy, s=21, color=COLOR["executed"],
               alpha=0.36, edgecolors="none", zorder=5.9)
    ax.scatter([terminal[0]], [terminal[1]], marker="X", s=170,
               color=COLOR["stop"], zorder=9, label="Terminal stop")
    ax.scatter([world.start[0]], [world.start[1]], s=145,
               color=COLOR["start"], zorder=8)
    ax.scatter([world.goal[0]], [world.goal[1]], s=155, marker="*",
               color=COLOR["goal"], zorder=8)
    ax.annotate("START", world.start, xytext=(7, -22),
                textcoords="offset points", fontsize=12, fontweight="bold",
                color="#21644c")
    ax.annotate("GOAL", world.goal, xytext=(5, 8),
                textcoords="offset points", fontsize=12, fontweight="bold",
                color=COLOR["goal"])

    title = "Minimum distance" if name == "shortest" else "Maximum safety"
    ax.set_title(f"{title}  |  {path_len:.3f} m planned",
                 fontsize=17, fontweight="bold", pad=12)
    
    legend_handles = [
        Line2D([0], [0], color=COLOR[name], lw=2.8, ls="--", label="Planned"),
        Line2D([0], [0], color=COLOR["executed"], lw=2.7, label="CSV poses"),
        Line2D([0], [0], marker="X", color=COLOR["stop"], lw=0,
               markersize=9, label="Terminal stop"),
        Line2D([0], [0], color=COLOR["cspace"], lw=2.0, ls="--", label="C-space"),
    ]
    if name == "shortest":
        legend_handles.append(Line2D([0], [0], color=COLOR["online"],
                                     lw=1.6, ls=":", label="Execution limit"))
    ax.legend(handles=legend_handles, loc="lower left", framealpha=0.91,
              fontsize=11, borderpad=0.35, labelspacing=0.25)
    fig.tight_layout(pad=0.8)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{name}_executed.png"
    fig.savefig(destination, dpi=155)
    plt.close(fig)

    goal_error_cm = 100 * math.dist(terminal, world.goal)
    print(f"{destination}\n  planned={path_len:.4f} m; CSV poses={len(points)}; "
          f"console terminal={terminal}; goal error={goal_error_cm:.2f} cm")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DATA_DIR,
                        help="Image output directory; default homework4/output")
    args = parser.parse_args()
    with (HERE / "config.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    world = World(cfg)
    for mode in ("shortest", "safety"):
        draw_one(mode, world, args.out)


if __name__ == "__main__":
    main()
