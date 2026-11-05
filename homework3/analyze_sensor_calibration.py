#!/usr/bin/env python3
"""Fit the AprilTag measurement correction and noise R from calibrate_sensor.py CSVs.

Runs without ROS (laptop is fine):
    python3 homework3/analyze_sensor_calibration.py homework3/output/sensor_calibration_*.csv

Outputs
  * a per-position table (bias and frame-to-frame std)
  * range correction  r_true = scale * r_measured + offset   (least squares)
  * bearing offset    (mean bearing error; camera yaw mounting error)
  * R parameters for ekf_slam.NoiseParams, from the errors AFTER correction:
        range_std(r) = range_std_base + range_std_per_m * r,   bearing_std
    These include the position-to-position error, not only the frame jitter,
    because that is what the filter actually sees while the robot moves.
  * homework3/config/sensor_calibration.yaml and a figure for the report.
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
MIN_RANGE_STD = 0.005              # m; tape measure resolution floor
MIN_BEARING_STD = math.radians(0.5)


def load(paths):
    groups = defaultdict(list)
    for path in paths:
        with open(path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                groups[(Path(path).name, row['label'], int(row['tag_id']))].append(row)
    out = []
    for (source, label, tag_id), rows in groups.items():
        out.append({
            'source': source, 'label': label, 'tag_id': tag_id,
            'true_r': float(rows[0]['true_range_m']),
            'true_b': math.radians(float(rows[0]['true_bearing_deg'])),
            'r': np.array([float(x['range_m']) for x in rows]),
            'b': np.radians([float(x['bearing_deg']) for x in rows]),
            'view': float(np.mean([float(x['view_deg']) for x in rows])),
        })
    return sorted(out, key=lambda g: (g['true_r'], g['true_b']))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', nargs='+')
    parser.add_argument('--out', default=str(ROOT / 'homework3/config/sensor_calibration.yaml'))
    parser.add_argument('--figure', default=str(ROOT / 'homework3/output/figures/sensor_calibration.png'))
    parser.add_argument('--max-view-deg', type=float, default=90.0,
                        help='ignore positions seen more obliquely than this when fitting')
    parser.add_argument('--exclude', nargs='+', default=[], metavar='LABEL',
                        help='drop these position labels entirely (e.g. a badly placed run that was redone)')
    args = parser.parse_args()

    groups = [g for g in load(args.csv) if g['label'] not in args.exclude]
    if not groups:
        raise SystemExit('no data')

    print(f'{"label":22s} {"tag":>4s} {"n":>3s} {"true r":>7s} {"r bias":>8s} {"r std":>7s} '
          f'{"true b":>7s} {"b bias":>7s} {"b std":>6s} {"view":>5s}')
    for g in groups:
        print(f'{g["label"]:22s} {g["tag_id"]:4d} {g["r"].size:3d} {g["true_r"]:6.3f}m '
              f'{100 * (g["r"].mean() - g["true_r"]):+7.2f}cm {100 * g["r"].std(ddof=1):6.2f}cm '
              f'{math.degrees(g["true_b"]):+6.1f}° {math.degrees(wrap(g["b"].mean() - g["true_b"])):+6.2f}° '
              f'{math.degrees(g["b"].std(ddof=1)):5.2f}° {g["view"]:4.0f}°')

    fit = [g for g in groups if g['view'] <= args.max_view_deg]
    if len(fit) < len(groups):
        print(f'\n{len(groups) - len(fit)} position(s) above {args.max_view_deg:.0f} deg view angle '
              'excluded from the fit')

    # --- range correction: r_true = scale * r_meas + offset
    mean_r = np.array([g['r'].mean() for g in fit])
    true_r = np.array([g['true_r'] for g in fit])
    if np.ptp(true_r) > 0.2:
        scale, offset = np.polyfit(mean_r, true_r, 1)
    else:
        scale, offset = 1.0, float(np.mean(true_r - mean_r))
        print('\nonly one distance: fitting a range offset only')

    # --- bearing offset (camera yaw): mean error over all frames
    b_err = np.concatenate([wrap(g['b'] - g['true_b']) for g in fit])
    bearing_offset = float(np.mean(b_err))

    def range_errors(g, corrected):
        r = scale * g['r'] + offset if corrected else g['r']
        return r - g['true_r']

    rms_before = math.sqrt(np.mean(np.concatenate([range_errors(g, False) for g in fit]) ** 2))
    rms_after = math.sqrt(np.mean(np.concatenate([range_errors(g, True) for g in fit]) ** 2))

    # --- R from errors AFTER correction (bias left at each position + jitter)
    pos_rms = np.array([math.sqrt(np.mean(range_errors(g, True) ** 2)) for g in fit])
    if np.ptp(true_r) > 0.2:
        per_m, base = np.polyfit(true_r, pos_rms, 1)
        if per_m < 0:   # noise should not shrink with distance
            per_m, base = 0.0, float(pos_rms.max())
    else:
        per_m, base = 0.0, float(pos_rms.max())
    base = max(base, MIN_RANGE_STD)
    bearing_std = max(math.sqrt(np.mean((b_err - bearing_offset) ** 2)), MIN_BEARING_STD)
    jitter_r = np.median([g['r'].std(ddof=1) for g in fit])
    jitter_b = np.median([g['b'].std(ddof=1) for g in fit])

    print('\n=== correction ===')
    print(f'range:   r_true = {scale:.4f} * r_meas {offset:+.4f} m   '
          f'(RMS error {100 * rms_before:.2f} cm -> {100 * rms_after:.2f} cm)')
    print(f'bearing: subtract {math.degrees(bearing_offset):+.2f} deg (camera yaw offset)')
    print('\n=== R for ekf_slam.NoiseParams (after correction) ===')
    print(f'range_std = {100 * base:.2f} cm + {100 * per_m:.2f} cm per m of range')
    print(f'bearing_std = {math.degrees(bearing_std):.2f} deg')
    print(f'(for comparison, median frame-to-frame jitter: range {100 * jitter_r:.2f} cm, '
          f'bearing {math.degrees(jitter_b):.2f} deg; R above also covers the error left '
          'between positions)')

    result = {
        'source_csv': [Path(p).name for p in args.csv],
        'range_scale': float(scale),
        'range_offset_m': float(offset),
        'bearing_offset_deg': math.degrees(bearing_offset),
        'range_std_base_m': float(base),
        'range_std_per_m': float(per_m),
        'bearing_std_deg': math.degrees(bearing_std),
        'max_calibrated_range_m': float(true_r.max()),
        'note': 'corrected = scale * measured + offset; bearing_corrected = measured - bearing_offset',
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        yaml.safe_dump(result, f, sort_keys=False)
    print(f'\nwrote {out}')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    # Without lateral positions every true bearing is 0: plot bearing error vs range instead.
    bearing_vs_range = np.ptp([g['true_b'] for g in groups]) < math.radians(1)
    for g in groups:
        before = 100 * (g['r'].mean() - g['true_r'])
        after = 100 * np.mean(range_errors(g, True))
        err = 100 * g['r'].std(ddof=1)
        ax1.errorbar(g['true_r'], before, yerr=err, fmt='o', color='tab:gray', capsize=3)
        ax1.errorbar(g['true_r'] + 0.02, after, yerr=err, fmt='s', color='tab:blue', capsize=3)
        ax2.errorbar(g['true_r'] if bearing_vs_range else math.degrees(g['true_b']),
                     math.degrees(wrap(g['b'].mean() - g['true_b'])),
                     yerr=math.degrees(g['b'].std(ddof=1)), fmt='o', color='tab:blue', capsize=3)
    ax1.axhline(0, color='k', lw=0.8)
    ax1.set(xlabel='tape-measured range (m)', ylabel='range error (cm)',
            title='Range error (gray: raw, blue: corrected)')
    ax2.axhline(0, color='k', lw=0.8)
    ax2.axhline(math.degrees(bearing_offset), color='tab:red', ls='--', lw=1,
                label=f'offset {math.degrees(bearing_offset):+.2f}°')
    ax2.set(xlabel='tape-measured range (m), tag straight ahead' if bearing_vs_range
            else 'tape-measured bearing (deg)', ylabel='bearing error (deg)',
            title='Bearing error (raw)')
    ax2.legend()
    fig.tight_layout()
    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=150)
    print(f'wrote {args.figure}')


if __name__ == '__main__':
    main()
