#!/usr/bin/env python3
"""Plot the CSE 276A HW2 trajectory from visual_navigation.py's CSV.

Run on Rubik Pi:
  python3 homework2/plot_hw2_trajectory.py \
      homework2/output/trajectory_drive_20261028_064146.csv
Output (same directory as CSV):
  hw2_trajectory.png, hw2_trajectory.pdf

Does not control the robot. Requires matplotlib.
"""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WAYPOINTS = [
    ("Start / Finish", 0.0, 0.0),
    ("WP1", 0.5, 0.0),
    ("WP2", 0.5, 1.0),
    ("Finish", 0.0, 0.0),
]

def load_data(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"elapsed_s", "x_m", "y_m"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Invalid/empty CSV; required columns: {sorted(required)}")
    samples = []
    for r in rows:
        try:
            t = float(r["elapsed_s"])
            x = float(r["x_m"])
            y = float(r["y_m"])
        except (ValueError, TypeError, KeyError):
            continue
        # Samples logged before the arming timer began all have elapsed=0.
        if t > 0.0:
            samples.append((t, x, y))
    if len(samples) < 2:
        raise ValueError("Not enough running trajectory samples in CSV.")
    return samples

def make_plot(samples, output_stem, note="Robot pose estimated from AprilTag landmarks"):
    tt, xx, yy = zip(*samples)
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=160)

    # Ideal route, dashed and unobtrusive.
    refx = [w[1] for w in WAYPOINTS]
    refy = [w[2] for w in WAYPOINTS]
    ax.plot(refx, refy, "--", linewidth=1.8, label="Reference route")
    ax.plot(xx, yy, "-", linewidth=1.35, alpha=0.75, label="Estimated trajectory")
    # Dots are the actual sampled poses, approx. every 0.5 s.
    ax.scatter(xx, yy, s=19, alpha=0.85, label="Logged position (~0.5 s)", zorder=3)
    # Show ideal targets distinctly, but not every repeated finish label.
    targets = [(0, 0, "Start / Finish"), (0.5, 0, "WP1"), (0.5, 1, "WP2")]
    for x, y, label in targets:
        ax.scatter([x], [y], marker="x", s=100, linewidths=2.0, zorder=5)
        if label == "Start / Finish":
            offset = (-42, -30)
        elif label == "WP1":
            offset = (12, -24)
        else:
            offset = (12, 14)
        ax.annotate(label, xy=(x, y), xytext=offset,
                    textcoords="offset points", fontsize=10, weight="bold")
    ax.scatter([xx[-1]], [yy[-1]], marker="s", s=50, zorder=6,
               label="Last logged pose")

    ax.set_title("CSE 276A HW2 | Closed-loop Visual Navigation", fontsize=15, pad=14)
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(min(-0.16, min(xx)-0.08), max(0.68, max(xx)+0.08))
    ax.set_ylim(min(-0.18, min(yy)-0.08), max(1.18, max(yy)+0.08))
    ax.grid(True, alpha=0.23)
    ax.legend(loc="upper left", fontsize=8)
    ax.text(0.01, -0.15, note, transform=ax.transAxes,
            ha="left", va="top", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    output_stem = Path(output_stem)
    fig.savefig(str(output_stem) + ".png", dpi=220, bbox_inches="tight")
    fig.savefig(str(output_stem) + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return str(output_stem) + ".png", str(output_stem) + ".pdf"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="successful trajectory_drive_*.csv")
    args = parser.parse_args()
    path = Path(args.csv).expanduser().resolve()
    samples = load_data(path)
    output_stem = path.parent / "hw2_trajectory"
    files = make_plot(samples, output_stem,
                      note="Each dot: onboard AprilTag pose estimate, ~0.5 s logging interval")
    print(f"Read {len(samples)} logged positions, {samples[0][0]:.2f}–{samples[-1][0]:.2f} s")
    print("Saved:", *files, sep="\n  ")

if __name__ == "__main__":
    main()
