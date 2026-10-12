"""Drive through the waypoints open loop.

  ros2 run robot_control velocity_control --ros-args -p waypoints_file:=/path/to/waypoints.txt
"""

import os

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from robot_control.waypoint_controller import Model, load_waypoints, plan


class VelocityControl(Node):
    def __init__(self):
        super().__init__('velocity_control')
        self.declare_parameter('waypoints_file', 'homework1/waypoints.txt')
        model = Model()
        for name, value in vars(model).items():
            self.declare_parameter(name, value)
            setattr(model, name, self.get_parameter(name).value)

        path = os.path.expanduser(self.get_parameter('waypoints_file').value)
        self.waypoints = load_waypoints(path)
        self.steps = plan(self.waypoints, model)
        self.ends = []
        t = 0.0
        for step in self.steps:
            t += step.duration
            self.ends.append(t)

        self.index = 0
        self.start = None
        self.done = False
        self.pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        self.create_timer(0.05, self.tick)
        self.get_logger().info(f'{model}, {len(self.steps)} steps, {t:.1f} s')

    def tick(self):
        now = self.get_clock().now()
        if self.start is None:
            # Wait for motor_control, otherwise the first command gets lost.
            if self.pub.get_subscription_count() == 0:
                return
            self.start = now
        t = (now - self.start).nanoseconds * 1e-9

        while self.index < len(self.steps) and t >= self.ends[self.index]:
            self.index += 1
            if self.index < len(self.steps) and self.steps[self.index].waypoint is not None:
                i = self.steps[self.index].waypoint
                self.get_logger().info(f'waypoint {i}: {self.waypoints[i]}')

        if self.index == len(self.steps):
            self.publish(0.0, 0.0)
            self.done = True
        else:
            self.publish(self.steps[self.index].left, self.steps[self.index].right)

    def publish(self, left, right):
        msg = Float32MultiArray()
        msg.data = [float(left), float(right)]
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VelocityControl()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
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
