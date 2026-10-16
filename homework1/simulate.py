#!/usr/bin/env python3
"""Run the open-loop plan on a simulated robot whose real parameters can be off.

  python3 simulate.py
  python3 simulate.py --speed-error 0.1 --spin-error -0.05 --drift 1
  python3 simulate.py --turn-noise 3 --trials 500
"""

import argparse
import math
import os
import random
import statistics
import sys

import matplotlib
import yaml
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'robot_control'))
from robot_control.waypoint_controller import Model, load_waypoints, plan, wrap_angle  # noqa

BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#1f1f1e', '#6b6a64', '#e4e3dd'


def load_model(path):
    with open(path) as f:
        params = yaml.safe_load(f)['velocity_control']['ros__parameters']
    return Model(**params)


def run(steps, model, start, speed_error=0.0, spin_error=0.0, drift=0.0, turn_noise=0.0,
        dt=0.01):
    """Integrate the pose. Each turn also coasts by its calibrated offset after it stops."""
    x, y, th = start
    path, reached = [(x, y)], {}
    for step in steps:
        v = w = extra = 0.0
        if step.left > 0 and step.right > 0:
            v = model.speed() * (1 + speed_error)
            w = math.radians(drift)
        elif step.left == -step.right != 0:
            cw = step.right < 0
            w = math.copysign(model.spin_rate(cw) * (1 + spin_error), step.right)
            offset = model.cw_offset if cw else model.spin_offset
            extra = math.copysign(math.radians(offset + random.gauss(0, turn_noise)), step.right)
        for _ in range(round(step.duration / dt)):
            x += v * math.cos(th) * dt
            y += v * math.sin(th) * dt
            th += w * dt
            path.append((x, y))
        th += extra
        if step.waypoint is not None:
            reached[step.waypoint] = (x, y, th)
    return path, reached


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--waypoints', default=os.path.join(HERE, 'waypoints.txt'))
    parser.add_argument('--config', default=os.path.join(HERE, 'config', 'ballroom.yaml'))
    parser.add_argument('--out', default=os.path.join(HERE, 'output', 'simulation.png'))
    parser.add_argument('--speed-error', type=float, default=0.0)
    parser.add_argument('--spin-error', type=float, default=0.0)
    parser.add_argument('--drift', type=float, default=0.0,
                        help='deg/s the robot curves left while driving straight')
    parser.add_argument('--turn-noise', type=float, default=0.0, help='std of each turn, deg')
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)

    waypoints = load_waypoints(args.waypoints)
    model = load_model(args.config)
    steps = plan(waypoints, model)
    errors = dict(speed_error=args.speed_error, spin_error=args.spin_error, drift=args.drift)
    path, reached = run(steps, model, waypoints[0], **errors)
    runs = [run(steps, model, waypoints[0], turn_noise=args.turn_noise, **errors)
            for _ in range(args.trials if args.turn_noise > 0 else 0)]

    print(f'{len(steps)} steps, {sum(s.duration for s in steps):.1f} s')
    for i, (gx, gy, gth) in enumerate(waypoints):
        x, y, th = reached[i]
        print(f'wp{i} ({gx:.2f}, {gy:.2f})  error {math.hypot(x - gx, y - gy):.3f} m, '
              f'{math.degrees(wrap_angle(th - gth)):+.0f} deg')
    if runs:
        last = len(waypoints) - 1
        gx, gy, _ = waypoints[last]
        final = sorted(math.hypot(r[last][0] - gx, r[last][1] - gy) for _, r in runs)
        print(f'{args.trials} trials, final error: median {statistics.median(final):.3f} m, '
              f'90% within {final[int(0.9 * len(final))]:.3f} m, '
              f'within 0.3 m: {sum(e <= 0.3 for e in final) / len(final):.0%}')

    fig, ax = plt.subplots(figsize=(6, 6))
    for i, (p, _) in enumerate(runs[:50]):
        ax.plot(*zip(*p), color=BLUE, lw=0.8, alpha=0.15,
                label=f'with {args.turn_noise:g} deg turn noise' if i == 0 else None)
    ax.plot(*zip(*[(w[0], w[1]) for w in waypoints]), color=ORANGE, lw=2, ls='--',
            label='planned')
    ax.plot(*zip(*path), color=BLUE, lw=2, label='simulated, no noise')
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
