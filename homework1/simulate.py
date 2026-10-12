#!/usr/bin/env python3
"""Run the open-loop plan on a simulated robot whose real parameters can be off.

  python3 simulate.py
  python3 simulate.py --speed-error 0.1 --spin-error -0.1 --drift 0.03
"""

import argparse
import math
import os
import sys
from collections import deque

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'robot_control'))
from robot_control.waypoint_controller import Model, load_waypoints, plan, wrap_angle  # noqa

BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#1f1f1e', '#6b6a64', '#e4e3dd'


def run(steps, twist, start, dt=0.02, delay=0.18):
    """Integrate the robot pose. The rover reacts about 0.18 s after each command."""
    x, y, th = start
    pending = deque([(0.0, 0.0)] * round(delay / dt))
    path, reached = [(x, y)], {}
    for step in steps:
        for _ in range(round(step.duration / dt)):
            pending.append((step.left, step.right))
            v, w = twist(*pending.popleft())
            x += v * math.cos(th) * dt
            y += v * math.sin(th) * dt
            th += w * dt
            path.append((x, y))
        if step.waypoint is not None:
            reached[step.waypoint] = (x, y, th)
    return path, reached


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--waypoints', default=os.path.join(HERE, 'waypoints.txt'))
    parser.add_argument('--out', default=os.path.join(HERE, 'output', 'simulation.png'))
    parser.add_argument('--speed-error', type=float, default=0.0)
    parser.add_argument('--spin-error', type=float, default=0.0)
    parser.add_argument('--drift', type=float, default=0.0,
                        help='right wheel faster by this fraction (robot curves left)')
    args = parser.parse_args()

    waypoints = load_waypoints(args.waypoints)
    model = Model()
    real = Model(speed_per_cmd=model.speed_per_cmd * (1 + args.speed_error),
                 wheel_base=model.wheel_base / (1 + args.spin_error))

    def real_twist(left, right):
        return real.twist(left * (1 - args.drift), right * (1 + args.drift))

    steps = plan(waypoints, model)
    planned, _ = run(steps, model.twist, waypoints[0], delay=0.0)
    actual, reached = run(steps, real_twist, waypoints[0])

    print(f'{sum(s.duration for s in steps):.1f} s')
    for i, (gx, gy, gth) in enumerate(waypoints):
        x, y, th = reached[i]
        print(f'wp{i} ({gx:.2f}, {gy:.2f})  error {math.hypot(x - gx, y - gy):.3f} m, '
              f'{math.degrees(wrap_angle(th - gth)):+.0f} deg')

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(*zip(*planned), color=ORANGE, lw=2, ls='--', label='planned')
    ax.plot(*zip(*actual), color=BLUE, lw=2, label='simulated robot')
    for gx, gy, gth in waypoints:
        ax.annotate('', xy=(gx + 0.2 * math.cos(gth), gy + 0.2 * math.sin(gth)), xytext=(gx, gy),
                    arrowprops=dict(arrowstyle='->', color=INK, lw=1.5))
        ax.plot(gx, gy, 'o', color=INK, mfc='white', ms=8)
    ax.set_aspect('equal')
    ax.margins(0.15)
    ax.grid(color=GRID)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.set_xlabel('x (m)', color=MUTED)
    ax.set_ylabel('y (m)', color=MUTED)
    ax.legend(frameon=False, loc='upper left')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
