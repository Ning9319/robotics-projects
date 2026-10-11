"""ROS-independent differential-drive waypoint navigation for CSE 276A HW1.

Commands L/R are normalized motor requests, NOT measured wheel speeds.
Calibration maps them to approximate physical motion. No ROS or hardware imports.
"""
from dataclasses import dataclass
import math
from pathlib import Path


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clip(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    theta: float


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


def read_waypoints(filename: str) -> list[Waypoint]:
    points = []
    for lineno, line in enumerate(Path(filename).read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(',')
        if len(parts) != 3:
            raise ValueError(f'Waypoint line {lineno} must contain x,y,theta: {line!r}')
        values = [float(p.strip()) for p in parts]
        if not all(math.isfinite(z) for z in values):
            raise ValueError(f'Waypoint line {lineno} has non-finite values')
        points.append(Waypoint(*values))
    if not points:
        raise ValueError('Waypoint file is empty')
    return points


@dataclass(frozen=True)
class Calibration:
    # meters/s per normalized, symmetric wheel-command unit
    linear_gain: float
    # radians/s per (right_command - left_command)
    yaw_gain: float
    max_command: float = 0.25
    yaw_sign: int = 1

    def __post_init__(self):
        if not (math.isfinite(self.linear_gain) and self.linear_gain > 0):
            raise ValueError('linear_gain must be measured and positive')
        if not (math.isfinite(self.yaw_gain) and self.yaw_gain > 0):
            raise ValueError('yaw_gain must be measured and positive')
        if not (0 < self.max_command <= 0.5):
            raise ValueError('max_command must be in (0,0.5]')
        if self.yaw_sign not in (-1, 1):
            raise ValueError('yaw_sign must be -1 or 1')

    def body_velocity(self, left: float, right: float) -> tuple[float, float]:
        return (self.linear_gain * (left + right) / 2.0,
                self.yaw_gain * (right - left))

    def wheel_commands(self, v: float, omega: float) -> tuple[float, float]:
        left = v / self.linear_gain - omega / (2.0 * self.yaw_gain)
        right = v / self.linear_gain + omega / (2.0 * self.yaw_gain)
        # Scale both proportionally to preserve requested curvature.
        scale = max(1.0, abs(left) / self.max_command,
                    abs(right) / self.max_command)
        return left / scale, right / scale


class DeadReckoner:
    """Estimated XY from calibrated motor commands, heading from IMU yaw.

    Open-loop XY will drift if wheels slip or motors differ from calibration.
    The first IMU yaw defines heading zero; robot must start facing world +X.
    """
    def __init__(self, calib: Calibration):
        self.calib = calib
        self.pose = Pose()
        self._yaw_zero = None
        self._last_integrated_heading = 0.0

    @property
    def has_heading(self) -> bool:
        return self._yaw_zero is not None

    def observe_yaw(self, raw_yaw: float) -> None:
        if not math.isfinite(raw_yaw):
            raise ValueError('Invalid IMU yaw')
        if self._yaw_zero is None:
            self._yaw_zero = raw_yaw
        self.pose.theta = wrap(self.calib.yaw_sign * wrap(raw_yaw - self._yaw_zero))

    def integrate(self, left: float, right: float, dt: float) -> None:
        if not self.has_heading or dt <= 0:
            return
        if not (math.isfinite(dt) and dt <= 0.2):
            raise ValueError('Unexpected integration time step')
        v, _ = self.calib.body_velocity(left, right)
        delta = wrap(self.pose.theta - self._last_integrated_heading)
        midpoint = wrap(self._last_integrated_heading + delta / 2.0)
        self.pose.x += v * math.cos(midpoint) * dt
        self.pose.y += v * math.sin(midpoint) * dt
        self._last_integrated_heading = self.pose.theta


@dataclass(frozen=True)
class Command:
    left: float
    right: float
    stage: str
    target_index: int
    finished: bool = False
    reached_waypoint: bool = False


class WaypointNavigator:
    """ALIGN -> DRIVE -> FINAL_ALIGN; stop between all waypoints."""
    def __init__(self, waypoints: list[Waypoint], calib: Calibration,
                 position_tol: float = 0.10, heading_tol: float = 0.15):
        if not waypoints:
            raise ValueError('Need at least one waypoint')
        self.waypoints = waypoints
        self.calib = calib
        self.position_tol = position_tol
        self.heading_tol = heading_tol
        self.index = 0
        self.stage = 'ALIGN'
        self.kp_distance = 0.8  # [1/s], v = kp * distance
        self.kp_heading = 2.0   # [1/s], omega = kp * angular error
        self.align_tol = 0.18
        self.realign_threshold = 0.65

    def update(self, pose: Pose) -> Command:
        if self.index >= len(self.waypoints):
            return Command(0.0, 0.0, 'DONE', self.index, finished=True)
        goal = self.waypoints[self.index]
        dx, dy = goal.x - pose.x, goal.y - pose.y
        distance = math.hypot(dx, dy)
        final_error = wrap(goal.theta - pose.theta)
        if distance <= self.position_tol and abs(final_error) <= self.heading_tol:
            completed = self.index
            self.index += 1
            self.stage = 'ALIGN'
            return Command(0.0, 0.0, 'DONE' if self.index == len(self.waypoints)
                           else 'NEXT', completed,
                           finished=self.index == len(self.waypoints),
                           reached_waypoint=True)

        if distance <= self.position_tol:
            self.stage = 'FINAL_ALIGN'
            return self._turn(final_error)

        # If a rotation nudges us out of the position tolerance, drive back.
        if self.stage == 'FINAL_ALIGN':
            self.stage = 'ALIGN'
        bearing_error = wrap(math.atan2(dy, dx) - pose.theta)
        if self.stage == 'ALIGN':
            if abs(bearing_error) > self.align_tol:
                return self._turn(bearing_error)
            self.stage = 'DRIVE'
        if abs(bearing_error) > self.realign_threshold:
            self.stage = 'ALIGN'
            return self._turn(bearing_error)

        v = self.kp_distance * distance * max(0.0, math.cos(bearing_error))
        omega = self.kp_heading * bearing_error
        left, right = self.calib.wheel_commands(v, omega)
        return Command(left, right, 'DRIVE', self.index)

    def _turn(self, error: float) -> Command:
        omega = self.kp_heading * error
        left, right = self.calib.wheel_commands(0.0, omega)
        return Command(left, right, self.stage, self.index)
