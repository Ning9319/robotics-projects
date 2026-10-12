#!/usr/bin/env python3
"""Calibration runs for the rover. Requires motor_control to be running.

Publishes a fixed [L, R] command for exact durations and measures the rotation by
integrating the gyro z rate (the IMU's own yaw is unreliable on this rover, it is only
printed for comparison). Fitting a line through runs with different durations removes
constant offsets such as acceleration and coasting.

  # straight: reposition the robot before each run, then type the measured distance
  python3 calibrate.py straight --cmd 0.2 --durations 2 2 2 6 6 6

  # spin in place (counter-clockwise): no repositioning needed
  python3 calibrate.py spin --cmd 0.3 --durations 2 2 4 4 --speed-per-cmd 1.35

  # gyro check: no motor command, turn the robot by hand during the run
  python3 calibrate.py spin --cmd 0 --durations 6
"""

import argparse
import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class CalibrationNode(Node):
    def __init__(self, gyro_scale):
        super().__init__('calibration_node')
        self.cmd_pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        self.create_subscription(Imu, 'imu_data', self._imu_callback, 10)
        self.gyro_scale = gyro_scale  # angular_velocity.z -> rad/s
        self.gyro_bias = 0.0
        self.gyro_angle = 0.0  # integrated gyro rotation since start (rad)
        self.imu_yaw = 0.0     # unwrapped IMU orientation yaw, for comparison only (rad)
        self.last_yaw = None
        self.last_stamp = None
        self.samples = []      # raw gz values, used while measuring the bias
        self.msg_count = 0

    def _imu_callback(self, msg):
        q = msg.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        if self.last_yaw is not None:
            self.imu_yaw += wrap_angle(yaw - self.last_yaw)
        self.last_yaw = yaw

        gz = msg.angular_velocity.z
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_stamp is not None:
            self.gyro_angle += (gz - self.gyro_bias) * self.gyro_scale * (stamp - self.last_stamp)
        self.last_stamp = stamp
        self.samples.append(gz)
        self.msg_count += 1

    def publish(self, left, right):
        msg = Float32MultiArray()
        msg.data = [float(left), float(right)]
        self.cmd_pub.publish(msg)

    def spin_for(self, seconds, cmd=(0.0, 0.0), rate=20.0):
        """Publish cmd at a fixed rate for exactly `seconds`, processing IMU callbacks."""
        end = time.monotonic() + seconds
        next_pub = time.monotonic()
        while time.monotonic() < end:
            if time.monotonic() >= next_pub:
                self.publish(*cmd)
                next_pub += 1.0 / rate
            rclpy.spin_once(self, timeout_sec=0.005)

    def wait_ready(self, timeout=10.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.last_stamp is not None and self.cmd_pub.get_subscription_count() > 0:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def measure_bias(self, seconds):
        """Average gz while the robot stands still."""
        self.samples = []
        self.spin_for(seconds)
        self.gyro_bias = sum(self.samples) / len(self.samples)
        return self.gyro_bias

    def run(self, cmd, duration):
        """Execute one run; return gyro rotation (rad) including the coast to a stop."""
        self.spin_for(0.5)  # settle
        gyro_start, yaw_start = self.gyro_angle, self.imu_yaw
        self.msg_count = 0
        self.samples = []
        t0 = time.monotonic()
        self.spin_for(duration, cmd)
        gyro_at_stop = self.gyro_angle
        self.spin_for(1.5)  # robot stops (zero commands), keep integrating
        elapsed = time.monotonic() - t0
        print(f'  IMU: {self.msg_count} msgs ({self.msg_count / elapsed:.1f} Hz), '
              f'IMU yaw change {math.degrees(self.imu_yaw - yaw_start):+.1f} deg (unreliable)')
        # Rotation after the zero command was sent: the controller must stop this much early.
        print(f'  during command: {math.degrees(gyro_at_stop - gyro_start):+.1f} deg, '
              f'after stop (coast): {math.degrees(self.gyro_angle - gyro_at_stop):+.1f} deg')
        # Raw gz statistics: single huge values are glitches that corrupt the integral.
        n = len(self.samples)
        mean = sum(self.samples) / n
        std = math.sqrt(sum((s - mean) ** 2 for s in self.samples) / n)
        print(f'  gz: min {min(self.samples):+.1f}, max {max(self.samples):+.1f}, '
              f'std {std:.2f}, mean {mean:+.2f} (IMU units, {n} samples)')
        return self.gyro_angle - gyro_start


def fit_line(xs, ys):
    """Least-squares slope and intercept; falls back to y/x through the origin."""
    n = len(xs)
    if len(set(xs)) < 2:
        return sum(ys) / sum(xs), 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return slope, my - slope * mx


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=['straight', 'spin'])
    parser.add_argument('--cmd', type=float, default=0.2, help='motor command magnitude')
    parser.add_argument('--durations', type=float, nargs='+', default=[2.0, 6.0])
    parser.add_argument('--speed-per-cmd', type=float,
                        help='result of the straight calibration, used to compute wheel_base')
    parser.add_argument('--gyro-scale', type=float, default=math.pi / 180.0,
                        help='angular_velocity.z to rad/s (default: the IMU reports deg/s)')
    parser.add_argument('--bias-time', type=float, default=5.0)
    args = parser.parse_args()
    c = min(abs(args.cmd), 0.5)
    cmd = (c, c) if args.mode == 'straight' else (-c, c)

    rclpy.init()
    node = CalibrationNode(args.gyro_scale)
    try:
        if not node.wait_ready():
            print('No IMU data or no motor_control subscriber. Is motor_control running?')
            return
        print(f'Measuring gyro bias for {args.bias_time:.1f} s, keep the robot still...')
        print(f'  gyro bias = {node.measure_bias(args.bias_time):+.3f} (IMU units)')

        results = []  # (duration, rotation rad, distance m or None)
        for i, duration in enumerate(args.durations, 1):
            input(f'\nRun {i}/{len(args.durations)}: cmd={cmd}, {duration:.1f} s. '
                  f'Place the robot and press Enter...')
            rotation = node.run(cmd, duration)
            distance = None
            if args.mode == 'straight':
                distance = float(input('  Measured distance (m): '))
            print(f'  gyro rotation: {math.degrees(rotation):+.1f} deg')
            results.append((duration, rotation, distance))

        print('\n run  duration  rotation (deg)  distance (m)')
        for i, (duration, rotation, distance) in enumerate(results, 1):
            dist = f'{distance:12.3f}' if distance is not None else ''
            print(f'{i:4d}  {duration:7.1f}  {math.degrees(rotation):+13.1f}  {dist}')

        durations = [r[0] for r in results]
        if args.mode == 'straight':
            v, offset = fit_line(durations, [r[2] for r in results])
            print(f'\nspeed = {v:.3f} m/s at cmd {c} (offset {offset:+.3f} m)')
            print(f'speed_per_cmd = {v / c:.3f}')
            # Heading change per meter driven: curvature caused by left/right asymmetry.
            kappa = sum(r[1] for r in results) / sum(r[2] for r in results)
            print(f'curvature = {kappa:+.3f} rad/m (positive = drifts left)')
            print('  right_trim = 1 + curvature * wheel_base / 2, '
                  'left_trim = 1 - curvature * wheel_base / 2')
        elif c > 0:
            w, offset = fit_line(durations, [r[1] for r in results])
            print(f'\nangular rate = {w:+.3f} rad/s ({math.degrees(w):+.1f} deg/s) at cmd {c}'
                  f' (offset {math.degrees(offset):+.1f} deg)')
            print(f'imu_yaw_sign = {1.0 if w > 0 else -1.0}  '
                  '(the command turns counter-clockwise)')
            if args.speed_per_cmd and abs(w) > 1e-3:
                print(f'wheel_base = {2 * c * args.speed_per_cmd / abs(w):.3f} m')
            elif not args.speed_per_cmd:
                print('pass --speed-per-cmd to compute wheel_base')
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish(0.0, 0.0)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
