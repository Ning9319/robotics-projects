#!/usr/bin/env python3
"""Estimate the process noise Q from recorded field runs (no ROS).

Ground truth for the robot's motion is taken from the TAGS, not from the SLAM
map: when the stops before and after one motion both see >= 2 of the same tags,
the change of those tags' positions in the robot frame gives the robot's actual
motion (2D rigid fit, like visual odometry; mm / sub-degree level). Comparing it
with the motion the EKF PREDICTED from the wheel commands gives the motion error
of that segment. Errors are then pooled:

  drive : along-track, lateral and heading error per segment -> variance per meter
  turn  : heading error vs. |turn angle| -> var = per_turn + per_rad * |angle|,
          plus the position shift of the robot during an in-place turn

    python3 homework3/estimate_q.py                       # default runs
    python3 homework3/estimate_q.py square_final octagon_final
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from replay import motions

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'homework3/output/runs'
DEFAULT_RUNS = ['square_test4', 'square_run2', 'octagon_run4', 'square_final', 'octagon_final']


def stops(run_dir):
    """[(t, {tag: (x, y) in robot frame})], one entry per stop, from updates.csv (raw range)."""
    cfg = json.loads((run_dir / 'config.json').read_text())
    sc, off = cfg['sensor_calibration']['range_scale'], cfg['sensor_calibration']['range_offset_m']
    with open(run_dir / 'updates.csv', newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if not r['status'].startswith(('unconfirmed', 'deferred'))]
    out = []
    for r in rows:
        t = float(r['t_s'])
        rng = (float(r['range_m']) - off) / sc          # undo any range correction
        b = math.radians(float(r['bearing_deg']))
        if not out or t - out[-1][0] > 0.05:
            out.append((t, {}))
        out[-1][1][int(r['tag_id'])] = np.array([rng * math.cos(b), rng * math.sin(b)])
    return out


def rigid(prev, curr):
    """Pose of the robot at `curr` in the frame of `prev` from common tags (Procrustes)."""
    ids = sorted(set(prev) & set(curr))
    if len(ids) < 2:
        return None
    P = np.array([prev[i] for i in ids])
    C = np.array([curr[i] for i in ids])
    cp, cc = P.mean(0), C.mean(0)
    U, _, Vt = np.linalg.svd((C - cc).T @ (P - cp))
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        return None
    t = cp - R @ cc
    spread = np.linalg.norm(P - cp, axis=1).max()
    return t, math.atan2(R[1, 0], R[0, 0]), len(ids), spread


def segments(run_dir):
    st = stops(run_dir)
    with open(run_dir / 'trajectory.csv', newline='', encoding='utf-8') as f:
        traj = list(csv.DictReader(f))
    mot = motions(traj)
    starts = [float(r['t_s']) for p, r in zip(traj, traj[1:])
              if r['phase'] in ('TURN', 'DRIVE') and p['phase'] != r['phase']]
    out = []
    for i, ((t_end, d, a, turn), t_start) in enumerate(zip(mot, starts)):
        before = [s for s in st if s[0] < t_start]
        after = [s for s in st if s[0] > t_end]
        if not before or not after:
            continue
        b, c = before[-1], after[0]
        prev_end = mot[i - 1][0] if i > 0 else -1.0
        next_start = starts[i + 1] if i + 1 < len(starts) else float('inf')
        if b[0] < prev_end or c[0] > next_start:   # the two stops must bracket ONLY this motion
            continue
        fit = rigid(b[1], c[1])
        if fit is None or fit[3] < 0.25:            # >= 2 common tags, spread >= 0.25 m
            continue
        t, th, n, _ = fit
        mid = a / 2
        pred = np.array([d * math.cos(mid), d * math.sin(mid)])
        err = t - pred
        along = err @ np.array([math.cos(mid), math.sin(mid)])
        lateral = err @ np.array([-math.sin(mid), math.cos(mid)])
        dth = math.atan2(math.sin(th - a), math.cos(th - a))
        out.append({'turn': turn, 'd': d, 'a': a, 'along': along, 'lateral': lateral,
                    'dth': dth, 'n_tags': n})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='*', default=DEFAULT_RUNS)
    args = p.parse_args()
    segs = []
    for name in args.runs:
        s = segments(RUNS / name)
        print(f'{name}: {len(s)} usable segments ({sum(x["turn"] for x in s)} turns)')
        segs += s
    drives = [s for s in segs if not s['turn'] and abs(s['d']) > 0.05]
    turns = [s for s in segs if s['turn']]

    D = np.array([abs(s['d']) for s in drives])
    al, la, hd = (np.array([s[k] for s in drives]) for k in ('along', 'lateral', 'dth'))
    print(f'\nDRIVE segments: n={len(drives)}, mean length {D.mean():.2f} m')
    print(f'  along-track error : mean {100 * al.mean():+.2f} cm, std {100 * al.std():.2f} cm')
    print(f'  lateral error     : mean {100 * la.mean():+.2f} cm, std {100 * la.std():.2f} cm')
    print(f'  heading error     : mean {math.degrees(hd.mean()):+.2f} deg, std {math.degrees(hd.std()):.2f} deg')
    # variance per meter (errors accumulate ~ linearly with distance); bias included (RMS)
    q = {'var_forward_per_m': float(np.sum(al ** 2) / D.sum()),
         'var_lateral_per_m': float(np.sum(la ** 2) / D.sum()),
         'var_heading_per_m': float(np.sum(hd ** 2) / D.sum())}

    A = np.array([abs(s['a']) for s in turns])
    th = np.array([s['dth'] for s in turns])
    pos = np.array([math.hypot(s['along'], s['lateral']) for s in turns])
    print(f'\nTURN segments: n={len(turns)}, |angle| {math.degrees(A.min()):.0f}-{math.degrees(A.max()):.0f} deg '
          f'(mean {math.degrees(A.mean()):.0f})')
    print(f'  heading error     : mean {math.degrees(th.mean()):+.2f} deg, std {math.degrees(th.std()):.2f} deg, '
          f'max |err| {math.degrees(np.abs(th).max()):.1f} deg')
    print(f'  position shift    : mean {100 * pos.mean():.2f} cm, RMS {100 * math.sqrt((pos ** 2).mean()):.2f} cm')
    X = np.column_stack([np.ones_like(A), A])
    c, *_ = np.linalg.lstsq(X, th ** 2, rcond=None)
    c = np.maximum(c, 0)
    q['turn_std_deg'] = math.degrees(math.sqrt(c[0]))
    q['var_heading_per_rad'] = float(c[1])
    q['turn_pos_std_m'] = float(math.sqrt((pos ** 2).mean() / 2))   # per axis
    print(f'  fit var(heading)  = ({q["turn_std_deg"]:.2f} deg)^2 per turn + {c[1]:.5f} rad^2 per rad turned')

    print('\nData-driven Q (RMS, bias included):')
    for k, v in q.items():
        print(f'  {k:22s} {v:.6g}')
    out = ROOT / 'homework3/output/q_estimate.json'
    out.write_text(json.dumps({'runs': args.runs, 'n_drive': len(drives), 'n_turn': len(turns), 'q': q},
                              indent=2))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
