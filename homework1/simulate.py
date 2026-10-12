#!/usr/bin/env python3
"""Offline simulation of the waypoint controller (no ROS needed).

The controller believes in `model`; the simulated robot moves according to `truth`,
which can differ from it to show the effect of calibration errors.

  python3 homework1/simulate.py
  python3 homework1/simulate.py --speed-error 0.1 --base-error -0.1 --right-trim 1.05
  python3 homework1/simulate.py --no-imu --allow-reverse
"""

import argparse
import math
import os
import random
import sys
from collections import deque

import matplotlib
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'robot_control'))

from robot_control.waypoint_controller import (  # noqa: E402
    ControllerParams, DiffDriveModel, PoseEstimator, WaypointController,
    integrate_pose, load_waypoints, wrap_angle,
)

matplotlib.use('Agg')

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3dd'


def simulate(waypoints, model, truth, params, use_imu, imu_noise, dt, max_time, seed, delay):
    rng = random.Random(seed)
    true_pose = waypoints[0]
    estimator = PoseEstimator(waypoints[0])
    reached = []

    def on_reached(index, goal, est_pose):
        reached.append((index, goal, est_pose, true_pose))

    controller = WaypointController(waypoints, params, on_reached=on_reached)
    log = {'t': [], 'true': [], 'est': [], 'left': [], 'right': []}
    cmd = (0.0, 0.0)
    # The real rover reacts ~0.18 s after a command (measured in spin calibration).
    pending = deque([(0.0, 0.0)] * round(delay / dt))
    t = 0.0
    while not controller.done and t < max_time:
        # Same order as the ROS node: propagate with last command, read IMU, compute control.
        pending.append(cmd)
        true_pose = integrate_pose(true_pose, *truth.cmds_to_twist(*pending.popleft()), dt)
        estimator.predict(*model.cmds_to_twist(*cmd), dt)
        if use_imu:
            estimator.set_heading(true_pose[2] + rng.gauss(0.0, imu_noise))
        v, w = controller.update(estimator.pose, dt)
        cmd = model.twist_to_cmds(v, w)

        log['t'].append(t)
        log['true'].append(true_pose)
        log['est'].append(estimator.pose)
        log['left'].append(cmd[0])
        log['right'].append(cmd[1])
        t += dt
    return log, reached, controller.done


def style_axes(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot(waypoints, log, reached, out_path, title):
    fig, (ax_path, ax_cmd) = plt.subplots(
        1, 2, figsize=(13, 5.5), gridspec_kw={'width_ratios': [1, 1.3]})

    style_axes(ax_path)
    tx, ty = zip(*[(p[0], p[1]) for p in log['true']])
    ex, ey = zip(*[(p[0], p[1]) for p in log['est']])
    ax_path.plot(ex, ey, color=ORANGE, linewidth=2, linestyle='--', label='estimated (dead reckoning)')
    ax_path.plot(tx, ty, color=BLUE, linewidth=2, label='true path')
    labels = {}
    for i, (x, y, theta) in enumerate(waypoints):
        ax_path.annotate('', xy=(x + 0.18 * math.cos(theta), y + 0.18 * math.sin(theta)),
                         xytext=(x, y), arrowprops=dict(arrowstyle='->', color=INK, lw=1.5))
        ax_path.plot(x, y, 'o', color=INK, markersize=8, markerfacecolor='white', zorder=3)
        labels.setdefault((round(x, 3), round(y, 3)), []).append(str(i))
    for (x, y), ids in labels.items():
        ax_path.annotate(','.join(ids), (x, y), textcoords='offset points', xytext=(-8, 8),
                         ha='right', color=MUTED, fontsize=9)
    for _, _, _, (x, y, _) in reached:
        ax_path.plot(x, y, 'o', color=AQUA, markersize=8, markeredgecolor='white',
                     markeredgewidth=2, zorder=4)
    ax_path.plot([], [], 'o', color=AQUA, markersize=8, label='true pose when "reached"')
    ax_path.set_aspect('equal')
    ax_path.set_xlabel('x (m)', color=MUTED)
    ax_path.set_ylabel('y (m)', color=MUTED)
    ax_path.set_title('Trajectory', color=INK, loc='left')
    ax_path.legend(loc='upper left', frameon=False, fontsize=9, labelcolor=INK)
    ax_path.margins(0.15)

    style_axes(ax_cmd)
    ax_cmd.plot(log['t'], log['left'], color=BLUE, linewidth=2, label='L')
    ax_cmd.plot(log['t'], log['right'], color=ORANGE, linewidth=2, label='R')
    ax_cmd.set_xlabel('time (s)', color=MUTED)
    ax_cmd.set_ylabel('motor command', color=MUTED)
    ax_cmd.set_title('Wheel commands', color=INK, loc='left')
    ax_cmd.legend(loc='upper right', frameon=False, labelcolor=INK)

    fig.suptitle(title, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--waypoints', default=os.path.join(HERE, 'waypoints.txt'))
    parser.add_argument('--out', default=os.path.join(HERE, 'output', 'simulation.png'))
    parser.add_argument('--speed-error', type=float, default=0.0,
                        help='true speed_per_cmd relative error, e.g. 0.1 = robot 10%% faster')
    parser.add_argument('--base-error', type=float, default=0.0,
                        help='true wheel_base relative error')
    parser.add_argument('--right-trim', type=float, default=1.0,
                        help='true right wheel speed multiplier (motor asymmetry)')
    parser.add_argument('--no-imu', action='store_true', help='heading from dead reckoning only')
    parser.add_argument('--imu-noise', type=float, default=0.01, help='IMU yaw noise std (rad)')
    parser.add_argument('--allow-reverse', action='store_true')
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--delay', type=float, default=0.18,
                        help='actuation delay of the simulated robot (s)')
    parser.add_argument('--max-time', type=float, default=180.0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    waypoints = load_waypoints(args.waypoints)
    model = DiffDriveModel()
    truth = DiffDriveModel(speed_per_cmd=model.speed_per_cmd * (1 + args.speed_error),
                           wheel_base=model.wheel_base * (1 + args.base_error),
                           right_trim=args.right_trim)
    params = ControllerParams(allow_reverse=args.allow_reverse)

    log, reached, finished = simulate(waypoints, model, truth, params, not args.no_imu,
                                      args.imu_noise, args.dt, args.max_time, args.seed,
                                      args.delay)

    print(f'{"finished" if finished else "TIMED OUT"} in {log["t"][-1]:.1f} s')
    print(f'{"wp":>3} {"goal (x, y, th)":>22} {"true (x, y, th)":>22} {"pos err":>8} {"th err":>7}')
    for index, goal, _, true_pose in reached:
        pos_err = math.hypot(true_pose[0] - goal[0], true_pose[1] - goal[1])
        th_err = wrap_angle(true_pose[2] - goal[2])
        print(f'{index:>3} ({goal[0]:5.2f}, {goal[1]:5.2f}, {goal[2]:5.2f}) '
              f'({true_pose[0]:5.2f}, {true_pose[1]:5.2f}, {true_pose[2]:5.2f}) '
              f'{pos_err:7.3f}m {th_err:6.3f}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    title = (f'speed err {args.speed_error:+.0%}, wheel base err {args.base_error:+.0%}, '
             f'right trim {args.right_trim:.2f}, IMU {"off" if args.no_imu else "on"}')
    plot(waypoints, log, reached, args.out, title)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
