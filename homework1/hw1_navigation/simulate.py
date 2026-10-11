"""Deterministic ideal differential-drive simulation; NO ROS and NO motors.

Run after build: ros2 run hw1_navigation simulate_hw1
Or directly: cd homework1 && python3 -m hw1_navigation.simulate
"""
import math
from pathlib import Path
from .core import Calibration, Pose, WaypointNavigator, read_waypoints, wrap


def simulate(path: str, dt: float = 0.05, max_seconds: float = 300.0):
    # These are synthetic SIMULATION gains; never use them as real calibrations.
    calib = Calibration(linear_gain=0.6, yaw_gain=2.0, max_command=0.25)
    navigator = WaypointNavigator(read_waypoints(path), calib)
    actual = Pose()
    record = []
    for i in range(int(max_seconds / dt)):
        cmd = navigator.update(actual)
        if cmd.reached_waypoint:
            record.append((cmd.target_index, i * dt, actual.x, actual.y, actual.theta))
        if cmd.finished:
            return actual, record
        v, omega = calib.body_velocity(cmd.left, cmd.right)
        mid = actual.theta + 0.5 * omega * dt
        actual.x += v * math.cos(mid) * dt
        actual.y += v * math.sin(mid) * dt
        actual.theta = wrap(actual.theta + omega * dt)
    raise RuntimeError(f'Simulation timed out after {max_seconds} s at waypoint {navigator.index}')


def main():
    try:
        from ament_index_python.packages import get_package_share_directory
        input_file = Path(get_package_share_directory('hw1_navigation')) / 'hw1' / 'waypoints.txt'
    except Exception:
        input_file = Path(__file__).resolve().parents[1] / 'waypoints.txt'
    pose, record = simulate(str(input_file))
    print('OFFLINE SIMULATION ONLY — does not control the Rover')
    for idx, t, x, y, yaw in record:
        print(f'Waypoint {idx}: t={t:6.2f}s pose=({x:.3f}, {y:.3f}, {yaw:.3f})')
    print(f'Complete. Final pose: {pose}')


if __name__ == '__main__':
    main()
