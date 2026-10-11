"""Offline unit tests; python3 -m unittest discover -s tests -v from homework1."""
import math
from pathlib import Path
import tempfile
import unittest

from hw1_navigation.core import (Calibration, DeadReckoner, Pose,
                                  WaypointNavigator, read_waypoints, wrap)
from hw1_navigation.simulate import simulate


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.calib = Calibration(0.6, 2.0, max_command=0.25)
        self.waypoints = str(Path(__file__).resolve().parents[1] / 'waypoints.txt')

    def test_wrap_across_pi(self):
        self.assertAlmostEqual(wrap(math.radians(-179) - math.radians(179)),
                               math.radians(2), places=6)

    def test_waypoint_reader(self):
        points = read_waypoints(self.waypoints)
        self.assertEqual(len(points), 7)
        self.assertEqual((points[4].x, points[4].y), (2.0, 2.0))
        with tempfile.NamedTemporaryFile(mode='w+', delete=True) as f:
            f.write('1, 2\n')
            f.flush()
            with self.assertRaises(ValueError):
                read_waypoints(f.name)

    def test_kinematics_straight_and_spin(self):
        v, w = self.calib.body_velocity(.2, .2)
        self.assertAlmostEqual(v, .12)
        self.assertAlmostEqual(w, 0)
        v, w = self.calib.body_velocity(-.2, .2)
        self.assertAlmostEqual(v, 0)
        self.assertAlmostEqual(w, .8)

    def test_command_limits(self):
        l, r = self.calib.wheel_commands(10, -10)
        self.assertLessEqual(max(abs(l), abs(r)), .25 + 1e-9)

    def test_gyro_bias_heading_and_dead_reckoning(self):
        est = DeadReckoner(self.calib)
        for i in range(41):
            est.observe_gyro(0.4, i * 0.1)
        self.assertTrue(est.has_heading)
        self.assertAlmostEqual(est.gyro_bias_dps, 0.4, places=8)
        for i in range(1, 11):
            est.observe_gyro(90.4, 4.0 + i * 0.1)
        self.assertAlmostEqual(est.pose.theta, math.pi / 2, places=6)
        est.integrate(.2, .2, .1)
        self.assertAlmostEqual(est.pose.x, .012 * math.cos(math.pi / 4), places=6)
        self.assertAlmostEqual(est.pose.y, .012 * math.sin(math.pi / 4), places=6)

    def test_gyro_calibration_rejects_motion(self):
        est = DeadReckoner(self.calib)
        for i in range(40):
            est.observe_gyro(30.0 if i % 2 else -30.0, i * 0.1)
        with self.assertRaises(ValueError):
            est.observe_gyro(0.0, 4.0)

    def test_gyro_rejects_stale_callback(self):
        est = DeadReckoner(self.calib)
        for i in range(41):
            est.observe_gyro(0.0, i * 0.1)
        with self.assertRaises(ValueError):
            est.observe_gyro(0.0, 4.4)

    def test_goal_position_and_heading_required(self):
        from hw1_navigation.core import Waypoint
        nav = WaypointNavigator([Waypoint(0, 0, math.pi / 2)], self.calib)
        c = nav.update(Pose())
        self.assertFalse(c.finished)
        self.assertEqual(c.stage, 'FINAL_ALIGN')
        c = nav.update(Pose(0, 0, math.pi / 2))
        self.assertTrue(c.finished)
        self.assertEqual((c.left, c.right), (0.0, 0.0))

    def test_all_seven_waypoints_in_ideal_simulation(self):
        pose, visits = simulate(self.waypoints)
        self.assertEqual([x[0] for x in visits], list(range(7)))
        self.assertLess(math.hypot(pose.x, pose.y), .11)
        self.assertLess(abs(wrap(pose.theta)), .16)


if __name__ == '__main__':
    unittest.main()
