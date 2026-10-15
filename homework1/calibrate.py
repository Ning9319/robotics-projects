#!/usr/bin/env python3
"""This file is used for calibration on different floors (apartment, grad housing, FAH)
Run fixed commands for fixed times, then type in what you measured.
motor_control must be running.

  python3 calibrate.py straight --cmd 0.2 --durations 2 2 4 4      # distance in m
  python3 calibrate.py spin --cmd 0.4 --durations 1 1 2 2          # angle in deg
"""

import argparse
import math
import time

import rclpy
from std_msgs.msg import Float32MultiArray


def fit_line(xs, ys):
    """Least-squares slope and intercept."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my / mx, 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return slope, my - slope * mx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['straight', 'spin'])
    parser.add_argument('--cmd', type=float, default=0.2)
    parser.add_argument('--durations', type=float, nargs='+', default=[2.0, 4.0])
    parser.add_argument('--speed-per-cmd', type=float, default=1.35)
    args = parser.parse_args()
    left, right = (args.cmd, args.cmd) if args.mode == 'straight' else (-args.cmd, args.cmd)

    rclpy.init()
    node = rclpy.create_node('calibrate')
    pub = node.create_publisher(Float32MultiArray, 'motor_commands', 10)

    def send(l, r, seconds):
        msg = Float32MultiArray()
        msg.data = [float(l), float(r)]
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            pub.publish(msg)
            time.sleep(0.01)

    while pub.get_subscription_count() == 0:
        print('waiting for motor_control...')
        time.sleep(1.0)

    unit = 'm' if args.mode == 'straight' else 'deg'
    results = []
    try:
        for i, duration in enumerate(args.durations, 1):
            input(f'\nrun {i}/{len(args.durations)}: {duration} s, place the robot, Enter')
            send(left, right, duration)
            send(0.0, 0.0, 1.0)
            results.append(float(input(f'  measured ({unit}): ')))
    finally:
        send(0.0, 0.0, 0.2)
        node.destroy_node()
        rclpy.shutdown()

    for duration, value in zip(args.durations, results):
        print(f'{duration:5.1f} s  {value:8.3f} {unit}')
    slope, offset = fit_line(args.durations, results)
    if args.mode == 'straight':
        print(f'speed {slope:.3f} m/s (offset {offset:+.3f} m), '
              f'speed_per_cmd = {slope / args.cmd:.3f}')
    else:
        rate = math.radians(slope)
        print(f'spin rate {slope:.1f} deg/s (offset {offset:+.1f} deg), '
              f'wheel_base = {2 * args.speed_per_cmd * abs(args.cmd) / rate:.3f}')


if __name__ == '__main__':
    main()
