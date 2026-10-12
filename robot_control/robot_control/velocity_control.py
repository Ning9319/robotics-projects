"""ROS 2 node that drives the rover through a list of waypoints.

Subscribes to imu_data (heading) and publishes [L, R] on motor_commands at a fixed rate.
Position is dead-reckoned from the executed commands. Heading comes from integrating the
gyro z rate by default; the IMU's own yaw was measured to be unreliable on this rover
(hand-turning 90 deg reported +24 / -40 deg, while the gyro integral gave 91 deg).

Example:
  ros2 run robot_control velocity_control --ros-args \
    -p waypoints_file:=/path/to/waypoints.txt -p speed_per_cmd:=0.8 -p wheel_base:=0.15
"""

import math
import os
from dataclasses import asdict

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray

from robot_control.waypoint_controller import (
    ControllerParams, DiffDriveModel, PoseEstimator, WaypointController,
    load_waypoints, wrap_angle,
)


def quaternion_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class VelocityControlNode(Node):
    def __init__(self):
        super().__init__('velocity_control_node')

        self.declare_parameter('waypoints_file', 'homework1/waypoints.txt')
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('use_imu', True)
        # 'gyro': integrate angular_velocity.z; 'imu_yaw': use the orientation quaternion.
        self.declare_parameter('heading_source', 'gyro')
        # Set to -1.0 if the heading increases when the robot turns clockwise.
        self.declare_parameter('imu_yaw_sign', 1.0)
        # Converts angular_velocity.z to rad/s. motor_control passes the raw value (deg/s).
        self.declare_parameter('gyro_scale', math.pi / 180.0)
        # Seconds of standing still at startup used to measure the gyro bias.
        self.declare_parameter('gyro_bias_time', 2.0)
        self.declare_parameter('imu_timeout', 0.5)
        # Model and controller parameters share their dataclass field names.
        for name, default in {**asdict(DiffDriveModel()), **asdict(ControllerParams())}.items():
            self.declare_parameter(name, default)

        path = os.path.expanduser(self.get_parameter('waypoints_file').value)
        waypoints = load_waypoints(path)
        self.use_imu = self.get_parameter('use_imu').value
        self.heading_source = self.get_parameter('heading_source').value
        self.imu_yaw_sign = self.get_parameter('imu_yaw_sign').value
        self.gyro_scale = self.get_parameter('gyro_scale').value
        self.gyro_bias_time = self.get_parameter('gyro_bias_time').value
        self.imu_timeout = self.get_parameter('imu_timeout').value
        if self.heading_source not in ('gyro', 'imu_yaw'):
            raise ValueError(f'heading_source must be gyro or imu_yaw, got {self.heading_source}')

        self.model = DiffDriveModel(**self._params_for(DiffDriveModel))
        params = ControllerParams(**self._params_for(ControllerParams))
        # The robot is assumed to start at the first waypoint.
        self.start_pose = waypoints[0]
        self.estimator = PoseEstimator(self.start_pose)
        self.controller = WaypointController(waypoints, params, on_reached=self._on_reached)

        self.heading = None  # becomes available once the IMU (and gyro bias) is ready
        self.yaw_offset = None
        self.imu_time = None
        self.last_stamp = None
        self.gyro_bias = None
        self.bias_samples = []
        self.bias_start = None
        self.last_cmd = (0.0, 0.0)
        self.finished_logged = False

        self.cmd_pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        self.create_subscription(Imu, 'imu_data', self._imu_callback, 10)
        self.last_tick = self.get_clock().now()
        self.timer = self.create_timer(1.0 / self.get_parameter('rate_hz').value, self._tick)

        self.get_logger().info(f'Loaded {len(waypoints)} waypoints from {path}')
        self.get_logger().info(f'Model: {self.model}')
        self.get_logger().info(f'Controller: {params}')

    def _params_for(self, cls):
        return {name: self.get_parameter(name).value for name in asdict(cls())}

    def _imu_callback(self, msg):
        self.imu_time = self.get_clock().now()
        if self.heading_source == 'imu_yaw':
            yaw = self.imu_yaw_sign * quaternion_to_yaw(msg.orientation)
            if self.yaw_offset is None:
                # Align the IMU frame with the waypoint frame using the start heading.
                self.yaw_offset = self.start_pose[2] - yaw
                self.get_logger().info(f'IMU initial yaw {math.degrees(yaw):.1f} deg')
            self.heading = wrap_angle(yaw + self.yaw_offset)
            return

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        gz = msg.angular_velocity.z
        if self.gyro_bias is None:
            # The robot is standing still: average the gyro reading to get its bias.
            if self.bias_start is None:
                self.bias_start = stamp
                self.get_logger().info(
                    f'Measuring gyro bias for {self.gyro_bias_time:.1f} s, keep the robot still')
            self.bias_samples.append(gz)
            if stamp - self.bias_start >= self.gyro_bias_time:
                self.gyro_bias = sum(self.bias_samples) / len(self.bias_samples)
                self.heading = self.start_pose[2]
                self.get_logger().info(f'Gyro bias {self.gyro_bias:+.3f} '
                                       f'({len(self.bias_samples)} samples)')
        else:
            rate = self.imu_yaw_sign * (gz - self.gyro_bias) * self.gyro_scale
            self.heading = wrap_angle(self.heading + rate * (stamp - self.last_stamp))
        self.last_stamp = stamp

    def _tick(self):
        now = self.get_clock().now()
        dt = (now - self.last_tick).nanoseconds * 1e-9
        self.last_tick = now

        # Propagate the pose with the command that was active during the last interval.
        self.estimator.predict(*self.model.cmds_to_twist(*self.last_cmd), dt)

        if self.use_imu:
            if self.heading is None:
                self._publish(0.0, 0.0)
                self.get_logger().info('Waiting for imu_data / gyro bias...',
                                       throttle_duration_sec=2.0)
                return
            if (now - self.imu_time).nanoseconds * 1e-9 > self.imu_timeout:
                self._publish(0.0, 0.0)
                self.get_logger().warn('IMU data is stale, holding still',
                                       throttle_duration_sec=1.0)
                return
            self.estimator.set_heading(self.heading)

        v, w = self.controller.update(self.estimator.pose, dt)
        self._publish(*self.model.twist_to_cmds(v, w))

        if self.controller.done:
            if not self.finished_logged:
                self.get_logger().info('All waypoints reached')
                self.finished_logged = True
            return
        x, y, theta = self.estimator.pose
        self.get_logger().info(
            f'[{self.controller.index}] {self.controller.phase.value:15s} '
            f'pose=({x:.2f}, {y:.2f}, {theta:.2f}) cmd=({self.last_cmd[0]:.2f}, '
            f'{self.last_cmd[1]:.2f})', throttle_duration_sec=1.0)

    def _on_reached(self, index, goal, pose):
        self.get_logger().info(
            f'Reached waypoint {index}: goal=({goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f}) '
            f'estimated=({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f})')

    def _publish(self, left, right):
        self.last_cmd = (left, right)
        msg = Float32MultiArray()
        msg.data = [float(left), float(right)]
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VelocityControlNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # motor_control also stops on its own 0.15 s command timeout.
        try:
            node._publish(0.0, 0.0)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
