#!/usr/bin/env python3
"""Estimate total translation and rotation from visual_navigation.py's CSV.

Run:
  python3 homework2/total_distance.py \
      homework2/output/trajectory_drive_20261101_031124.csv

Translation: steps are summed while the rover drives forward. For each
in-place turn or stop, only the net displacement from its start to its end
is added, so tag-switch jumps during turns are not counted as motion.
Rotation: sum of absolute wrapped heading changes between samples.
"""
import argparse
import csv
import math


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        # Samples logged before the arming timer began all have elapsed=0.
        return [r for r in csv.DictReader(f) if float(r["elapsed_s"]) > 0.0]


def driving(row):
    return float(row["motor_left"]) > 0 and float(row["motor_right"]) > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="trajectory_drive_*.csv")
    parser.add_argument("--unit-m", type=float, default=0.5,
                        help="meters per homework unit (default: 0.5)")
    args = parser.parse_args()

    rows = load_rows(args.csv)
    xy = [(float(r["x_m"]), float(r["y_m"])) for r in rows]
    yaw = [math.radians(float(r["heading_deg"])) for r in rows]

    def step(i, j):
        return math.hypot(xy[j][0] - xy[i][0], xy[j][1] - xy[i][1])

    all_steps = sum(step(i, i + 1) for i in range(len(rows) - 1))
    drive = sum(step(i, i + 1) for i in range(len(rows) - 1)
                if driving(rows[i]) or driving(rows[i + 1]))

    # Each maximal run of non-driving samples is one in-place turn/stop.
    turn_net = 0.0
    i = 0
    while i < len(rows):
        if driving(rows[i]):
            i += 1
            continue
        j = i
        while j + 1 < len(rows) and not driving(rows[j + 1]):
            j += 1
        turn_net += step(i, j)
        i = j + 1

    rotation = sum(abs(wrap_angle(yaw[i + 1] - yaw[i])) for i in range(len(rows) - 1))
    total = drive + turn_net

    print(f"Samples: {len(rows)}")
    print(f"Translation: drive {drive:.3f} m + turn net {turn_net:.3f} m "
          f"= {total:.3f} m ({total / args.unit_m:.2f} units)")
    print(f"  (all steps summed: {all_steps:.3f} m)")
    print(f"Rotation: {rotation:.2f} rad ({math.degrees(rotation):.0f} deg)")


if __name__ == "__main__":
    main()
