#!/usr/bin/env python3
"""Effect of the tag viewing angle on range / bearing (justifies --max-view-deg).

Uses a calibrate_sensor.py CSV in which the robot stays fixed and only the tag
is rotated about its vertical axis (labels such as view0, view30, ...). Errors
are shown RELATIVE to the most head-on position, so robot placement and tape
errors cancel. Runs without ROS:

    python3 homework3/plot_view_angle.py homework3/output/sensor_calibration_20261105_043411.csv \
        --not-detected 70 --limit 60
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv')
    parser.add_argument('--prefix', default='view', help='only labels starting with this')
    parser.add_argument('--not-detected', type=float, nargs='*', default=[],
                        help='tag angles (deg) at which the tag was never detected')
    parser.add_argument('--limit', type=float, default=60.0, help='--max-view-deg used by the node')
    parser.add_argument('--figure', default=str(ROOT / 'homework3/output/figures/view_angle.png'))
    args = parser.parse_args()

    groups = defaultdict(list)
    with open(args.csv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['label'].startswith(args.prefix):
                groups[row['label']].append(row)
    if not groups:
        raise SystemExit(f'no labels starting with {args.prefix!r}')

    rows = []
    for label, data in groups.items():
        r = np.array([float(d['range_m']) for d in data])
        b = np.array([float(d['bearing_deg']) for d in data])
        v = np.array([float(d['view_deg']) for d in data])
        rows.append((v.mean(), label, r.mean(), r.std(ddof=1), b.mean(), b.std(ddof=1), len(r)))
    rows.sort()
    v0, _, r0, _, b0, _, _ = rows[0]

    print(f'reference: {rows[0][1]} (view {v0:.0f} deg)')
    print(f'{"label":10s} {"view":>6s} {"n":>3s} {"d_range":>9s} {"d_bearing":>10s}')
    for v, label, r, rs, b, bs, n in rows:
        print(f'{label:10s} {v:5.1f}° {n:3d} {100 * (r - r0):+8.2f}cm {b - b0:+9.2f}°')
    for a in args.not_detected:
        print(f'{"(" + str(int(a)) + " deg)":10s} not detected')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    views = [row[0] for row in rows]
    ax1.errorbar(views, [100 * (row[2] - r0) for row in rows], yerr=[100 * row[3] for row in rows],
                 fmt='o-', color='tab:blue', capsize=3)
    ax2.errorbar(views, [row[4] - b0 for row in rows], yerr=[row[5] for row in rows],
                 fmt='o-', color='tab:blue', capsize=3)
    for ax in (ax1, ax2):
        ax.axhline(0, color='k', lw=0.8)
        ax.axvspan(args.limit, 90, color='tab:red', alpha=0.12, label=f'rejected by node (> {args.limit:.0f}°)')
        for a in args.not_detected:
            ax.axvline(a, color='tab:red', ls='--', lw=1)
            ax.text(a, ax.get_ylim()[1], ' not\n detected', color='tab:red', va='top', fontsize=8)
        ax.set_xlim(0, 80)
        ax.set_xlabel('tag viewing angle (deg, estimated by AprilTag)')
    ax1.set(ylabel='range change vs head-on (cm)', title='Range vs viewing angle (1.5 m)')
    ax2.set(ylabel='bearing change vs head-on (deg)', title='Bearing vs viewing angle (1.5 m)')
    ax1.legend(loc='lower left', fontsize=8)
    fig.tight_layout()
    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=150)
    print(f'wrote {args.figure}')


if __name__ == '__main__':
    main()
