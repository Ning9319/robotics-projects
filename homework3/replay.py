#!/usr/bin/env python3
"""Offline re-processing of a recorded HW3 run with different filter settings (no ROS).

It re-runs the EKF on the SAME recorded data as slam_node.py:
  * motion: the commanded motion of every TURN / DRIVE, recovered from the
    logged predicted poses (trajectory.csv: pose at motion start -> pose after
    the prediction, before the camera update),
  * measurements: the per-stop averaged (range, bearing) of every tag
    (updates.csv), optionally with a different range correction.
Only the filter is re-run; the robot's driving cannot change. Use it for
sensitivity analysis (range correction, Q / R scale), not as a new experiment.

    python3 homework3/replay.py homework3/output/runs/square_run2                 # reproduce
    python3 homework3/replay.py homework3/output/runs/square_run2 --range-scale 1 --range-offset 0
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from ekf_slam import EKFSLAM, NoiseParams, wrap_angle


def load_rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def motions(traj):
    """[(t_end, d, a, is_turn)] from consecutive (TURN|DRIVE start row, SETTLE row)."""
    out, start = [], None
    for prev, row in zip(traj, traj[1:]):
        if row['phase'] in ('TURN', 'DRIVE') and prev['phase'] != row['phase']:
            start = row
        elif row['phase'] == 'SETTLE' and prev['phase'] in ('TURN', 'DRIVE') and start is not None:
            x0, y0, th0 = float(start['x_m']), float(start['y_m']), math.radians(float(start['yaw_deg']))
            x1, y1, th1 = float(row['x_m']), float(row['y_m']), math.radians(float(row['yaw_deg']))
            a = wrap_angle(th1 - th0)
            mid = th0 + a / 2
            d = (x1 - x0) * math.cos(mid) + (y1 - y0) * math.sin(mid)
            out.append((float(row['t_s']), d, a, start['phase'] == 'TURN'))
            start = None
    return out


def replay(run_dir, range_scale=None, range_offset=None, r_scale=None, q_scale=None,
           new_max_yaw_std_deg=10.0, min_new_frames=3, q_override=None, turn_std_deg=None,
           turn_pos_std_m=0.0):
    """q_override: dict of NoiseParams motion fields (absolute values, before q_scale).
    turn_std_deg: per-turn heading std (default: the run's turn_coast_std_deg).
    turn_pos_std_m: per-turn position std per axis (in-place spins also shift the robot)."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / 'config.json').read_text())
    old_sc = cfg['sensor_calibration']['range_scale']
    old_off = cfg['sensor_calibration']['range_offset_m']
    sc = old_sc if range_scale is None else range_scale
    off = old_off if range_offset is None else range_offset

    n = dict(cfg['noise'])
    if q_override:
        n.update(q_override)
    if q_scale is not None:
        for k in ('var_forward_per_m', 'var_lateral_per_m', 'var_heading_per_rad', 'var_heading_per_m'):
            n[k] *= q_scale
    if r_scale is not None:
        f = r_scale / cfg['args']['r_scale']
        for k in ('range_std_base', 'range_std_per_m', 'bearing_std'):
            n[k] *= f
    slam = EKFSLAM(NoiseParams(**n))
    turn_var = math.radians(cfg['turn_coast_std_deg'] if turn_std_deg is None else turn_std_deg) ** 2
    turn_pos_var = turn_pos_std_m ** 2

    events = [(t, 0, ('motion', d, a, turn)) for t, d, a, turn in motions(load_rows(run_dir / 'trajectory.csv'))]
    for k, r in enumerate(load_rows(run_dir / 'updates.csv')):
        events.append((float(r['t_s']), 1, ('update', k, r)))
    sides = 4 if cfg['args']['path'] == 'square' else 8
    for r in load_rows(run_dir / 'vertices.csv'):
        if int(r['side']) == sides:
            events.append((float(r['t_s']), 2, ('lap', int(r['lap']))))
    events.sort(key=lambda e: (e[0], e[1], e[2][1] if e[2][0] == 'update' else 0))

    snapshots, nis = {}, []
    for _, _, ev in events:
        if ev[0] == 'motion':
            _, d, a, turn = ev
            slam.predict(d, a)
            if turn:
                slam.add_robot_noise(np.diag([turn_pos_var, turn_pos_var, turn_var]))
        elif ev[0] == 'update':
            r = ev[2]
            if r['status'].startswith('unconfirmed') or int(r['n_frames']) < 1:
                continue
            tag = int(r['tag_id'])
            rng = (float(r['range_m']) - old_off) / old_sc * sc + off
            brg = math.radians(float(r['bearing_deg']))
            if tag not in slam.index:
                if int(r['n_frames']) < min_new_frames or \
                        math.sqrt(slam.robot_cov[2, 2]) > math.radians(new_max_yaw_std_deg):
                    continue
            slam.last_nis = float('nan')
            slam.update(tag, rng, brg)
            if not math.isnan(slam.last_nis):
                nis.append(slam.last_nis)
        else:
            snapshots[ev[1]] = slam.snapshot()
    return snapshots, np.array(nis), cfg


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dir')
    p.add_argument('--range-scale', type=float, default=None)
    p.add_argument('--range-offset', type=float, default=None)
    p.add_argument('--r-scale', type=float, default=None)
    p.add_argument('--q-scale', type=float, default=None)
    a = p.parse_args()
    snaps, nis, _ = replay(a.run_dir, a.range_scale, a.range_offset, a.r_scale, a.q_scale)
    for lap, s in sorted(snaps.items()):
        print(f'lap {lap}: {len(s["landmark_ids"])} landmarks')
    print(f'NIS mean {nis.mean():.2f} over {nis.size} updates')


if __name__ == '__main__':
    main()
