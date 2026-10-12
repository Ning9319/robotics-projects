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

    Defaults are calibrated on our rover on a wooden floor (homework1/calibrate.py).
    """

    # Wheel ground speed (m/s) produced by one unit of motor command.
    # Straight runs at cmd 0.2: 0.227 m in 2 s, 1.307 m in 6 s -> 0.27 m/s.
    speed_per_cmd: float = 1.35
    # Effective track width (m), absorbs the skid-steer slip. From in-place turns at
    # cmd 0.5: 120 deg/s = 2 * 0.5 * 1.35 / wheel_base.
    wheel_base: float = 0.64
    # Multipliers that compensate for left/right motor asymmetry (>1 means that wheel is faster).
    # Measured drift was 3 cm left over 1.3 m, small enough to leave at 1.0.
    left_trim: float = 1.0
    right_trim: float = 1.0
    # Smallest command magnitude that moves the robot when driving straight.
    min_cmd: float = 0.07
    # motor_control clips commands to +-0.5.
    max_cmd: float = 0.5
    # In-place turns stall below ~0.5 on our floor (cmd 0.4 turned 84 deg in 1 s, then
    # stuck), so a pure rotation always uses this command. 0 disables it.
    spin_cmd: float = 0.5

    def twist_to_cmds(self, v, w):
        """Convert body velocity (m/s, rad/s) to (left, right) motor commands."""
        if v == 0.0 and w != 0.0 and self.spin_cmd > 0.0:
            spin = math.copysign(self.spin_cmd, w)
            return -spin, spin
        left =(v - w * self.wheel_base / 2) / (self.speed_per_cmd * self.left_trim)
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
    # Slowest speed while approaching a waypoint. Kept at max_v: lower commands sit near the
    # 0.07 deadband where the robot moves slower than the linear model predicts, so it
    # stopped short. The 0.18 s actuation delay cancels out (late start, late stop).
    min_v: float = 0.2
    # Limits steering while driving so both wheels keep turning forward; a reversed wheel at
    # low command is a slow in-place turn, which stalls on our floor.
    max_curvature: float = 1.5  # 1/m
    max_w: float = 1.5          # rad/s
    k_rho: float = 1.0          # forward speed gain on remaining distance
    k_heading: float = 1.0      # heading correction gain while driving
    k_theta: float = 2.0        # rotation gain for in-place turns (only when spin_lead = 0)
    dist_tol: float = 0.02      # m
    angle_tol: float = 0.15     # rad; in-place turns can't do finer than ~15 deg steps
    pause_time: float = 0.5     # s, stop at each waypoint (makes the video easy to read)
    allow_reverse: bool = False  # drive backwards when it needs less rotation
    # In-place turns run at full spin_cmd and stop early: the rover reacts ~0.18 s late,
    # so it keeps turning ~23 deg after the stop command (measured 21-26 deg).
    # spin_lead = 0 switches back to a proportional turn.
    spin_lead: float = 0.40     # rad, stop spinning when this close to the target
    spin_settle: float = 0.5    # s, wait after a spin before checking the heading again
    spin_min_time: float = 0.1  # s, shortest spin (~15 deg) so a correction actually moves
    max_spin_tries: int = 5     # spins per turn before accepting the remaining error


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
        self._spin = self._new_spin()

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
                    self._enter(Phase.ROTATE_TO_GOAL)
                    continue
                if self._segment is None:
                    self._segment = self._plan_segment(theta, math.atan2(dy, dx), gtheta)
                heading, _ = self._segment
                w = self._rotate(wrap_angle(heading - theta), dt)
                if w is None:
                    self._enter(Phase.DRIVE)
                    continue
                return 0.0, w

            if self.phase == Phase.DRIVE:
                heading, direction = self._segment
                # Distance left along the segment; negative once we have passed the goal.
                travel = heading if direction > 0 else wrap_angle(heading + math.pi)
                along = dx * math.cos(travel) + dy * math.sin(travel)
                if along <= 0.0:
                    self._enter(Phase.ROTATE_TO_GOAL)
                    continue
                # Aim at the goal while far away; near it the bearing gets noisy, hold the line.
                if dist > 0.1:
                    bearing = math.atan2(dy, dx)
                    target = bearing if direction > 0 else wrap_angle(bearing + math.pi)
                else:
                    target = heading
                err = wrap_angle(target - theta)
                if math.cos(err) <= 0.0:
                    # Facing away from the goal: steering can't fix it, turn in place again.
                    self._segment = None
                    self._enter(Phase.ROTATE_TO_PATH)
                    continue
                speed = min(p.max_v, max(p.min_v, p.k_rho * along))
                speed *= max(0.0, math.cos(err))  # slow down when badly misaligned
                w_limit = min(p.max_w, p.max_curvature * speed)
                w = max(-w_limit, min(w_limit, p.k_heading * err))
                return direction * speed, w

            if self.phase == Phase.ROTATE_TO_GOAL:
                w = self._rotate(wrap_angle(gtheta - theta), dt)
                if w is None:
                    if self.on_reached:
                        self.on_reached(self.index, self.goal, pose)
                    self._pause_left = p.pause_time
                    self._enter(Phase.PAUSE)
                    continue
                return 0.0, w

            if self.phase == Phase.PAUSE:
                if self._pause_left > 0.0:
                    self._pause_left -= dt
                    return 0.0, 0.0
                self.index += 1
                self._segment = None
                self._enter(Phase.DONE if self.index >= len(self.waypoints)
                            else Phase.ROTATE_TO_PATH)
                continue

    def _enter(self, phase):
        self.phase = phase
        self._spin = self._new_spin()

    @staticmethod
    def _new_spin():
        # direction: 0 = not spinning, +1/-1 = spinning CCW/CW
        return {'direction': 0, 'elapsed': 0.0, 'settle': 0.0, 'tries': 0}

    def _rotate(self, err, dt):
        """Return w for an in-place turn that reduces err, or None once aligned."""
        p = self.params
        if p.spin_lead <= 0.0:
            return None if abs(err) < p.angle_tol else self._turn_rate(err)

        s = self._spin
        if s['settle'] > 0.0:
            s['settle'] -= dt
            return 0.0
        if s['direction'] == 0:
            if abs(err) < p.angle_tol or s['tries'] >= p.max_spin_tries:
                return None
            s.update(direction=1 if err > 0 else -1, elapsed=0.0, tries=s['tries'] + 1)
        # Full-speed spin (the model maps a pure rotation to spin_cmd), stopped spin_lead
        # early because the robot keeps turning after the stop command.
        s['elapsed'] += dt
        if s['direction'] * err <= p.spin_lead and s['elapsed'] >= p.spin_min_time:
            s.update(direction=0, settle=p.spin_settle)
            return 0.0
        return s['direction'] * p.max_w

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
