"""ROS2 wrapper around the ROS-independent waypoint algorithm.

SAFETY: enable_motion defaults to false, and real calibration is required.
Start on an unobstructed floor with direct access to power / emergency stop.
"""
import json
import math
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray

from .core import Calibration, DeadReckoner, WaypointNavigator, read_waypoints


class WaypointFollower(Node):
    def __init__(self):
        super().__init__('hw1_waypoint_follower')
        self.declare_parameter('enable_motion', False)
        self.declare_parameter('waypoints_file', '')
        self.declare_parameter('calibration_file', '')
        self.declare_parameter('imu_timeout_s', 0.4)
        self.declare_parameter('waypoint_timeout_s', 60.0)
        self.enabled = bool(self.get_parameter('enable_motion').value)
        self.active = self.enabled
        self.last_imu_time = None
        self.last_tick = time.monotonic()
        self.start_time = self.last_tick
        self.heading_ready_logged = False
        self.target_start_time = None
        self.last_command = (0.0, 0.0)
        self.last_stage = None
        self.estimator = None
        self.navigator = None

        self.pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        self.sub = self.create_subscription(Imu, 'imu_data', self.on_imu, 10)

        if self.enabled:
            wp_path = str(self.get_parameter('waypoints_file').value)
            cal_path = str(self.get_parameter('calibration_file').value)
            if not wp_path or not cal_path:
                raise ValueError('waypoints_file and calibration_file are required')
            settings = json.loads(Path(cal_path).read_text())
            calibration = Calibration(
                linear_gain=settings['linear_gain_mps_per_command'],
                yaw_gain=settings['yaw_gain_radps_per_command_difference'],
                max_command=settings.get('max_motor_command', 0.25),
                yaw_sign=settings.get('imu_yaw_sign', 1),
            )
            self.estimator = DeadReckoner(calibration)
            self.navigator = WaypointNavigator(
                read_waypoints(wp_path), calibration,
                position_tol=settings.get('position_tolerance_m', 0.10),
                heading_tol=settings.get('heading_tolerance_rad', 0.15),
            )
            self.get_logger().warn('MOTION ENABLED: 20 Hz motor commands; clear floor and supervise.')
        else:
            self.get_logger().info('DISARMED: enable_motion=false; no motor commands will be sent.')

        self.timer = self.create_timer(0.05, self.on_tick)  # 20 Hz

    def on_imu(self, msg: Imu) -> None:
        if not self.active or self.estimator is None:
            return
        try:
            # The supplied template publishes raw degrees/second despite
            # sensor_msgs/Imu specifying radians/second by convention.
            # Do not convert the sensor stream a second time here.
            now = time.monotonic()
            ready_before = self.estimator.has_heading
            self.estimator.observe_gyro(float(msg.angular_velocity.z), now)
            self.last_imu_time = now
            if self.estimator.has_heading and not ready_before:
                self.get_logger().info(
                    f'Gyro bias ready: {self.estimator.gyro_bias_dps:+.3f} deg/s; '
                    'relative heading initialized at 0 rad')
        except ValueError as exc:
            self.abort(f'Invalid gyro or failed stationary calibration: {exc}')

    def send(self, left: float, right: float) -> None:
        msg = Float32MultiArray()
        msg.data = [float(left), float(right)]
        self.pub.publish(msg)
        self.last_command = (left, right)

    def abort(self, why: str) -> None:
        if not self.active:
            return
        self.get_logger().error(f'HW1 ABORT: {why}')
        self.active = False
        self.send(0.0, 0.0)

    def on_tick(self) -> None:
        if not self.active:
            return
        now = time.monotonic()
        dt = now - self.last_tick
        self.last_tick = now
        if dt > 0.2:
            self.abort('Control loop delayed > 0.2 s')
            return
        if self.last_imu_time is None:
            if now - self.start_time > 8.0:
                self.abort('No IMU samples received in 8 seconds')
            else:
                self.send(0.0, 0.0)
            return
        if now - self.last_imu_time > float(self.get_parameter('imu_timeout_s').value):
            self.abort('IMU data stale')
            return
        if not self.estimator.has_heading:
            # Four seconds stationary bias calibration before ANY motion.
            self.send(0.0, 0.0)
            return
        self.estimator.integrate(*self.last_command, dt)
        result = self.navigator.update(self.estimator.pose)
        if result.reached_waypoint:
            self.get_logger().info(f'Completed waypoint {result.target_index}')
            self.target_start_time = now
        if self.target_start_time is None:
            self.target_start_time = now
        elif now - self.target_start_time > float(self.get_parameter('waypoint_timeout_s').value):
            self.abort(f'Waypoint {self.navigator.index} timed out')
            return
        if result.stage != self.last_stage:
            p = self.estimator.pose
            self.get_logger().info(
                f'Waypoint {self.navigator.index} | {result.stage} | '
                f'estimated pose=({p.x:.2f}, {p.y:.2f}, {p.theta:.2f})'
            )
            self.last_stage = result.stage
        self.send(result.left, result.right)
        if result.finished:
            self.get_logger().info('All waypoints completed (estimated pose; verify physical accuracy).')
            self.active = False

    def destroy_node(self):
        if self.enabled:
            self.send(0.0, 0.0)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = WaypointFollower()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
