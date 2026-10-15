import math
from collections import namedtuple
from dataclasses import dataclass


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def load_waypoints(path):
    waypoints = []
    with open(path) as f:
        for line in f:
            if line.strip():
                waypoints.append(tuple(float(v) for v in line.split(',')))
    return waypoints


@dataclass
class Model:
    """Differential drive: v = k (L + R) / 2,  w = k (R - L) / W."""

    # Manual Calibration:
    speed_per_cmd: float = 1.35  # k, m/s per unit command
    wheel_base: float = 0.64     # W, effective value for in-place turns
    drive_cmd: float = 0.2
    spin_cmd: float = 0.4
    trim: float = 0.0            # > 0 steers right when driving straight
    spin_offset: float = 0.0     # deg the robot keeps turning after a spin stops
    # Clockwise spins are calibrated separately (the two sides are not symmetric).
    cw_wheel_base: float = 0.0   # 0 = same as wheel_base
    cw_offset: float = 0.0       # deg


    # Kinematic Model
    def twist(self, left, right):
        v = self.speed_per_cmd * (left + right) / 2
        w = self.speed_per_cmd * (right - left) / self.wheel_base
        return v, w

    def speed(self):
        return self.twist(self.drive_cmd, self.drive_cmd)[0]

    def spin_rate(self, clockwise=False):
        w = self.cw_wheel_base if clockwise and self.cw_wheel_base > 0 else self.wheel_base
        return 2 * self.speed_per_cmd * self.spin_cmd / w


# One constant motor command. waypoint is set on the pause after reaching that waypoint.
Step = namedtuple('Step', 'left right duration waypoint')


def plan(waypoints, model, settle=0.3, pause=1.0):
    """Turn toward each waypoint, drive straight to it, turn to its heading, pause."""
    steps = []
    x, y, theta = waypoints[0]

    def turn(angle):
        angle = wrap_angle(angle)
        cw = angle < 0
        offset = model.cw_offset if cw else model.spin_offset
        duration = (abs(angle) - math.radians(offset)) / model.spin_rate(cw)
        if duration > 0.05:
            c = math.copysign(model.spin_cmd, angle)
            steps.append(Step(-c, c, duration, None))
            steps.append(Step(0.0, 0.0, settle, None))

    for i, (gx, gy, gtheta) in enumerate(waypoints):
        dist = math.hypot(gx - x, gy - y)
        if dist > 0.01:
            heading = math.atan2(gy - y, gx - x)
            turn(heading - theta)
            theta = heading
            c = model.drive_cmd
            steps.append(Step(c * (1 + model.trim), c * (1 - model.trim),
                              dist / model.speed(), None))
            steps.append(Step(0.0, 0.0, settle, None))
        turn(gtheta - theta)
        x, y, theta = gx, gy, gtheta
        steps.append(Step(0.0, 0.0, pause, i))
    return steps
