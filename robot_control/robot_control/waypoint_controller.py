"""Waypoint following for a differential-drive robot (no ROS dependency).

Contents:
  - load_waypoints: parse "x, y, theta" lines (meters, radians)
  - DiffDriveModel: kinematic model, converts between (v, w) and wheel commands
  - PoseEstimator: dead reckoning, optionally with heading from the IMU
  - WaypointController: rotate -> drive -> rotate state machine
"""

import math
from dataclasses import dataclass
from enum import Enum


def wrap_angle(angle):
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def load_waypoints(path):
    """Read waypoints as a list of (x, y, theta) tuples."""
    waypoints = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) != 3:
                raise ValueError(f'{path}:{line_no}: expected "x, y, theta", got {line!r}')
            waypoints.append(tuple(float(p) for p in parts))
    if not waypoints:
        raise ValueError(f'{path}: no waypoints found')
    return waypoints


@dataclass
class DiffDriveModel:
    """Differential-drive kinematics.

    v = (v_R + v_L) / 2,  w = (v_R - v_L) / wheel_base
    x' = v cos(theta),    y' = v sin(theta),    theta' = w

    The default values are placeholders; replace them with calibrated ones.
    """

    # Wheel ground speed (m/s) produced by one unit of motor command.
    speed_per_cmd: float = 1.0
    # Effective track width (m). Calibrate from in-place turns, it absorbs wheel slip.
    wheel_base: float = 0.13
    # Multipliers that compensate for left/right motor asymmetry (>1 means that wheel is faster).
    left_trim: float = 1.0
    right_trim: float = 1.0
    # Smallest command magnitude that actually turns a wheel (motor deadband).
    min_cmd: float = 0.0
    # motor_control clips commands to +-0.5.
    max_cmd: float = 0.5

    def twist_to_cmds(self, v, w):
        """Convert body velocity (m/s, rad/s) to (left, right) motor commands."""
        left = (v - w * self.wheel_base / 2) / (self.speed_per_cmd * self.left_trim)
        right = (v + w * self.wheel_base / 2) / (self.speed_per_cmd * self.right_trim)
        # Scale both wheels together so saturation keeps the turning radius.
        peak = max(abs(left), abs(right))
        if peak > self.max_cmd:
            left *= self.max_cmd / peak
            right *= self.max_cmd / peak
        return self._apply_deadband(left), self._apply_deadband(right)

    def cmds_to_twist(self, left, right):
        """Convert (left, right) motor commands to body velocity (m/s, rad/s)."""
        v_left = left * self.speed_per_cmd * self.left_trim
        v_right = right * self.speed_per_cmd * self.right_trim
        return (v_right + v_left) / 2, (v_right - v_left) / self.wheel_base

    def _apply_deadband(self, cmd):
        # Commands inside the deadband would not move the wheel, so bump them up to min_cmd.
        if abs(cmd) < 1e-6:
            return 0.0
        if abs(cmd) < self.min_cmd:
            return math.copysign(self.min_cmd, cmd)
        return cmd


def integrate_pose(pose, v, w, dt):
    """Advance (x, y, theta) by constant (v, w) for dt seconds (exact arc integration)."""
    x, y, theta = pose
    if abs(w) < 1e-9:
        x += v * dt * math.cos(theta)
        y += v * dt * math.sin(theta)
    else:
        new_theta = theta + w * dt
        x += v / w * (math.sin(new_theta) - math.sin(theta))
        y -= v / w * (math.cos(new_theta) - math.cos(theta))
        theta = new_theta
    return x, y, wrap_angle(theta)


class PoseEstimator:
    """Dead reckoning from executed commands; heading can be overwritten by the IMU."""

    def __init__(self, initial_pose):
        self.pose = tuple(initial_pose)

    def predict(self, v, w, dt):
        self.pose = integrate_pose(self.pose, v, w, dt)

    def set_heading(self, theta):
        x, y, _ = self.pose
        self.pose = (x, y, wrap_angle(theta))


@dataclass
class ControllerParams:
    max_v: float = 0.2          # m/s, cruise speed
    min_v: float = 0.05         # m/s, slowest speed while approaching a waypoint
    max_w: float = 1.5          # rad/s
    k_rho: float = 1.0          # forward speed gain on remaining distance
    k_heading: float = 2.0      # heading correction gain while driving
    k_theta: float = 2.0        # rotation gain for in-place turns
    dist_tol: float = 0.02      # m
    angle_tol: float = 0.05     # rad
    pause_time: float = 0.5     # s, stop at each waypoint (makes the video easy to read)
    allow_reverse: bool = False  # drive backwards when it needs less rotation


class Phase(Enum):
    ROTATE_TO_PATH = 'rotate_to_path'
    DRIVE = 'drive'
    ROTATE_TO_GOAL = 'rotate_to_goal'
    PAUSE = 'pause'
    DONE = 'done'


class WaypointController:
    """Visit each waypoint by rotating toward it, driving straight, then turning to its theta.

    Segments shorter than dist_tol skip the rotate/drive phases, so the first waypoint
    can be the start pose itself.
    """

    def __init__(self, waypoints, params=None, on_reached=None):
        self.waypoints = list(waypoints)
        self.params = params or ControllerParams()
        # Called as on_reached(index, goal, pose) when a waypoint is reached.
        self.on_reached = on_reached
        self.index = 0
        self.phase = Phase.ROTATE_TO_PATH
        self._segment = None  # (travel heading, direction +1/-1) of the current segment
        self._pause_left = 0.0

    @property
    def done(self):
        return self.phase == Phase.DONE

    @property
    def goal(self):
        return self.waypoints[min(self.index, len(self.waypoints) - 1)]

    def update(self, pose, dt):
        """Return (v, w) for the current estimated pose."""
        p = self.params
        x, y, theta = pose
        # A single tick may pass through several phases (e.g. skipping a zero-length segment).
        while True:
            if self.phase == Phase.DONE:
                return 0.0, 0.0

            gx, gy, gtheta = self.goal
            dx, dy = gx - x, gy - y
            dist = math.hypot(dx, dy)

            if self.phase == Phase.ROTATE_TO_PATH:
                if dist < p.dist_tol:
                    self.phase = Phase.ROTATE_TO_GOAL
                    continue
                if self._segment is None:
                    self._segment = self._plan_segment(theta, math.atan2(dy, dx), gtheta)
                heading, _ = self._segment
                err = wrap_angle(heading - theta)
                if abs(err) < p.angle_tol:
                    self.phase = Phase.DRIVE
                    continue
                return 0.0, self._turn_rate(err)

            if self.phase == Phase.DRIVE:
                heading, direction = self._segment
                # Distance left along the segment; negative once we have passed the goal.
                travel = heading if direction > 0 else wrap_angle(heading + math.pi)
                along = dx * math.cos(travel) + dy * math.sin(travel)
                if along <= 0.0:
                    self.phase = Phase.ROTATE_TO_GOAL
                    continue
                # Aim at the goal while far away; near it the bearing gets noisy, hold the line.
                if dist > 0.1:
                    bearing = math.atan2(dy, dx)
                    target = bearing if direction > 0 else wrap_angle(bearing + math.pi)
                else:
                    target = heading
                err = wrap_angle(target - theta)
                speed = min(p.max_v, max(p.min_v, p.k_rho * along))
                speed *= max(0.0, math.cos(err))  # slow down when badly misaligned
                w = max(-p.max_w, min(p.max_w, p.k_heading * err))
                return direction * speed, w

            if self.phase == Phase.ROTATE_TO_GOAL:
                err = wrap_angle(gtheta - theta)
                if abs(err) < p.angle_tol:
                    if self.on_reached:
                        self.on_reached(self.index, self.goal, pose)
                    self._pause_left = p.pause_time
                    self.phase = Phase.PAUSE
                    continue
                return 0.0, self._turn_rate(err)

            if self.phase == Phase.PAUSE:
                if self._pause_left > 0.0:
                    self._pause_left -= dt
                    return 0.0, 0.0
                self.index += 1
                self._segment = None
                self.phase = Phase.DONE if self.index >= len(self.waypoints) \
                    else Phase.ROTATE_TO_PATH
                continue

    def _plan_segment(self, theta, bearing, gtheta):
        """Choose forward or reverse driving; return (robot heading while driving, direction)."""
        def rotation_cost(heading):
            return abs(wrap_angle(heading - theta)) + abs(wrap_angle(gtheta - heading))

        reverse_heading = wrap_angle(bearing + math.pi)
        if self.params.allow_reverse and rotation_cost(reverse_heading) < rotation_cost(bearing):
            return reverse_heading, -1
        return bearing, 1

    def _turn_rate(self, err):
        p = self.params
        return max(-p.max_w, min(p.max_w, p.k_theta * err))
