#!/usr/bin/env python3
"""HW4: known-map AprilTag closed-loop rover navigation.

Reuses HW2 VisualNavigation's /tf -> robot pose transform, HW1 calibrated
motor model, pulse-based rotation and /motor_commands publisher. Does NOT
implement an AprilTag detector, ROS motor driver or SLAM.

Default: preview only (NO WHEEL MOTION). --drive enables wheel commands.
"""
import argparse
import csv
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from tf2_msgs.msg import TFMessage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'homework2'))
sys.path.insert(0, str(ROOT / 'robot_control'))

from visual_navigation import Pose, VisualNavigation    # HW2, reused
from robot_control.waypoint_controller import Model, wrap_angle  # HW1
from check_localization import NORMALS, UP                 # HW2 extrinsics already in on_tf
from geometry import point_rect_distance, wall_clearance, segment_clearances
from planning import World, max_clearance_grid, shortest_visibility, validate_path

HERE = Path(__file__).resolve().parent
VISION_TIMEOUT = 0.75             # No blind forward driving on stale camera poses.
FINAL_TOL = 0.045                # Instructor's ~4.5 cm guidance for 8 ft field.
TEST_WP_TOL = 0.055
PASS_WP_RADIUS = 0.15           # Intermediate points guide the path; not stop locations.
TRAVEL_HEADING_TOL = math.radians(25)
EXECUTION_BUFFER = 0.02         # In addition to actual robot radius, NOT planning margin.
MAX_RUN_SECONDS = 600           # Emergency run limit, not a corridor/deviation abort.
REPLAN_RETRY_SECONDS = 1.0
# Symmetric in-place sweep: left -> right -> return, pausing for camera frames.
# Each sweep is bounded, and failure to see tags leaves the process WAITING,
# not aborted. Repeated blind spinning can accumulate positional drift.
SEARCH = [(0., 0., .55),
          (-.22, +.22, .22), (0., 0., .50),
          (+.22, -.22, .44), (0., 0., .50),
          (-.22, +.22, .22), (0., 0., .50)]
SEARCH_SWEEPS = 3


class HW4Navigation(VisualNavigation):
    """Only HW4 state machine is new; inherit HW2 camera TF and motor code."""

    def __init__(self, mode, drive, stop_after_wp=None):
        # Deliberately bypass HW2's *arena-specific* initializer / WP2 hacks.
        # The inherited on_tf/send/spin_towards/pending_spin/close are reused.
        Node.__init__(self, 'hw4_navigation')
        self.drive = drive
        self.mode = mode
        self.world = World(yaml.safe_load((HERE / 'config.yaml').read_text()))
        if mode == 'shortest':
            path, _ = shortest_visibility(self.world)
        else:
            path, _ = max_clearance_grid(self.world)
        if validate_path(self.world, path):
            raise ValueError('Refusing an invalid planned route')
        self.original_path = list(path)  # Offline planning result stays unchanged.
        self.waypoints_xy = list(path)    # Execution route may recover/replan.
        self.stop_at_goal = stop_after_wp is None
        self.stop_after = stop_after_wp if stop_after_wp is not None else (len(path) - 1)
        if not (1 <= self.stop_after < len(path)):
            raise ValueError(f'--stop-after-wp must be 1..{len(path)-1}')

        self.model = Model()
        calibration_path = ROOT / 'homework1/config/ballroom.yaml'
        calibration = yaml.safe_load(calibration_path.read_text())['velocity_control']['ros__parameters']
        for key, value in calibration.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value)

        # HW4 known map: all eight tags, including IDs 8 and 9.
        self.tags = {}
        for tag_id, tag in self.world.cfg['landmarks'].items():
            normal = np.array(NORMALS[tag['facing']], dtype=float)
            r_world_tag = np.column_stack((np.cross(UP, normal), UP, normal))
            self.tags[int(tag_id)] = (np.array([tag['x'], tag['y'], 0.14]), r_world_tag)

        self.last_by_tag = {}
        self.pose = None
        self.pose_seq = 0
        self.last_fused_sample = 0.0
        self.target_index = 1
        self.phase = 'TRAVEL'
        self.armed = False
        self.ready_since = None
        self.start_at = None
        self.done = False
        self.last_motion_at = 0.0
        self.last_log = 0.0
        self.last_command = (0.0, 0.0)
        # Required by the inherited HW2 calibrated pulse spin.
        self.spin_command = None
        self.spin_pulse_until = 0.0
        self.spin_settle_until = 0.0
        self.spin_observed_seq = -1
        self.last_spin_direction = 0
        self.last_spin_at = 0.0
        self.lost_at = None
        self.search_step = 0
        self.search_step_at = 0.0
        self.search_sweeps = 0
        self.reacquire_frames = 0
        self.reacquire_seq = -1
        self.reacquire_pose = None
        self.last_replan_at = -1e9
        self.last_block_log = -1e9

        self.motor_pub = self.create_publisher(Float32MultiArray, '/motor_commands', 10)
        self.create_subscription(TFMessage, '/tf', self.on_tf, 10)
        self.create_timer(0.05, self.tick)
        folder = HERE / 'output'
        folder.mkdir(parents=True, exist_ok=True)
        label = 'drive' if drive else 'preview'
        log_path = folder / f'{mode}_{label}_{datetime.now():%Y%m%d_%H%M%S}.csv'
        self.log_file = log_path.open('w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.log_file)
        self.writer.writerow(['elapsed_s', 'x_m', 'y_m', 'yaw_deg', 'tag_ids',
                              'target_wp', 'state', 'motor_left', 'motor_right'])
        print(f'MODE={mode}, DRIVE={drive}; robot start={self.world.start}, '
              f'yaw={self.world.cfg["start_yaw_deg"]}deg; target={self.world.goal}', flush=True)
        print('Route:', [(round(x, 3), round(y, 3)) for x, y in path], flush=True)
        print('CSV:', log_path, flush=True)
        print('Preview only. Use --drive ONLY after dry-run localization and safe clearance checks.'
              if not drive else 'DRIVE ENABLED: Verify floor, pose, obstacle and emergency stop.', flush=True)

    def update_pose(self, now):
        # Reuse HW2's tested camera->world TF transforms via inherited on_tf;
        # only adjust its map-specific initial-pose gate and fusion.
        fresh = [(tag_id, *p) for tag_id, p in self.last_by_tag.items()
                 if now - p[4] < VISION_TIMEOUT]
        if not fresh:
            return
        latest = max(p[5] for p in fresh)
        if latest <= self.last_fused_sample:
            return
        self.last_fused_sample = latest
        fresh = [p for p in fresh if latest - p[5] < 0.25]
        accepted = []
        initial_yaw = math.radians(self.world.cfg['start_yaw_deg'])
        for tag_id, x, y, yaw, rng, stamp in fresh:
            if self.pose is None:
                if (math.hypot(x - self.world.start[0], y - self.world.start[1]) > 0.18
                    or abs(wrap_angle(yaw - initial_yaw)) > math.radians(25)):
                    continue
            else:
                dt = max(0., now - self.pose.stamp)
                position_jump = math.hypot(x - self.pose.x, y - self.pose.y)
                if self.lost_at is not None:
                    # We deliberately turned while searching. Do not reject a
                    # reacquired tag just because its heading differs from the
                    # old, now-stale orientation. Reacquire TWO consistent
                    # images before moving forward again.
                    if position_jump > 0.30:
                        continue
                elif (position_jump > min(0.60, .22 + .22*dt)
                      or abs(wrap_angle(yaw - self.pose.yaw)) > min(
                          math.radians(65), math.radians(30) + .25*dt)):
                    continue
            accepted.append((tag_id, x, y, yaw, rng))
        if not accepted:
            return
        anchor = min(accepted, key=lambda p: p[4])
        consistent = [p for p in accepted
                      if math.hypot(p[1]-anchor[1], p[2]-anchor[2]) < .18
                      and abs(wrap_angle(p[3]-anchor[3])) < math.radians(18)]
        weights = [1.0/max(p[4], .4)**2 for p in consistent]
        wsum = sum(weights)
        x = sum(w*p[1] for w, p in zip(weights,consistent))/wsum
        y = sum(w*p[2] for w, p in zip(weights,consistent))/wsum
        yaw = math.atan2(sum(w*math.sin(p[3]) for w,p in zip(weights,consistent)),
                         sum(w*math.cos(p[3]) for w,p in zip(weights,consistent)))
        if self.pose is not None:
            alpha = 1.0 if self.lost_at is not None else .7
            x = self.pose.x + alpha*(x-self.pose.x)
            y = self.pose.y + alpha*(y-self.pose.y)
            yaw = wrap_angle(self.pose.yaw + alpha*wrap_angle(yaw-self.pose.yaw))
        self.pose = Pose(x, y, yaw, now, ','.join(str(p[0]) for p in consistent))
        self.pose_seq += 1

    def segment_drivable(self, a, b):
        """Conservative physical collision envelope, independent of planning cost.

        Offline planning inflates by the robot radius only (no extra 10 cm).
        Online execution independently requires the circumscribed radius plus
        its existing 2 cm buffer, against the measured raw obstacle/walls.
        """
        obstacle_gap, wall_gap = segment_clearances(
            a, b, self.world.side, self.world.raw_obstacle)
        return min(obstacle_gap, wall_gap) >= self.world.robot_radius + EXECUTION_BUFFER

    def rotation_drivable(self, pos):
        # In-place rotation still occupies the robot's circumscribed disk.
        return self.segment_drivable(pos, pos)

    def replan_from_pose(self, now):
        """Only if a direct recovery to the next route point is physically blocked.

        Preserve the original offline plan for grading. For recovery, plan
        against at least the 2 cm online execution buffer, so a newly planned
        segment must pass the same physical-clearance check before driving.
        """
        if now - self.last_replan_at < REPLAN_RETRY_SECONDS:
            return False
        self.last_replan_at = now
        start = (self.pose.x, self.pose.y)
        destination = (self.waypoints_xy[self.stop_after] if not self.stop_at_goal
                       else self.world.goal)
        for margin in (max(self.world.base_margin, EXECUTION_BUFFER),):
            cfg = {**self.world.cfg,
                   'start_xy_m': list(start), 'goal_xy_m': list(destination),
                   'planning': {**self.world.cfg['planning'], 'base_margin_m': margin}}
            try:
                local_world = World(cfg)
                if self.mode == 'safety':
                    route, _ = max_clearance_grid(local_world)
                else:
                    route, _ = shortest_visibility(local_world)
                if validate_path(local_world, route):
                    continue
                if len(route) < 2 or not self.segment_drivable(start, route[1]):
                    continue
            except ValueError:
                continue
            self.waypoints_xy = route
            self.target_index = 1
            self.stop_after = len(route) - 1
            self.spin_command = None
            self.phase = 'TRAVEL'
            print('REROUTE from current pose, not ABORT: '
                  f'({start[0]:.3f}, {start[1]:.3f}), '
                  f'{len(route)-1} legs, margin={margin:.3f}m', flush=True)
            return True
        return False

    def command_for_waypoint(self, now):
        # Use the existing HW2 calibrated short-pulse rotation when a large
        # change of heading really is needed. No mandatory pauses at WP1/WP2.
        active_spin = self.pending_spin(now)
        if active_spin is not None:
            return active_spin
        pos = (self.pose.x, self.pose.y)

        # Intermediate waypoints are hints: pass within 15 cm when the next
        # leg is physically clear. Never require exact corner contact.
        while self.target_index < self.stop_after:
            target = self.waypoints_xy[self.target_index]
            next_target = self.waypoints_xy[self.target_index + 1]
            if (math.dist(pos, target) <= PASS_WP_RADIUS
                    and self.segment_drivable(pos, next_target)):
                print(f'PASS WP{self.target_index} without stopping; '
                      'continue toward the goal.', flush=True)
                self.target_index += 1
            else:
                break

        target = self.waypoints_xy[self.target_index]
        distance = math.dist(pos, target)
        arrival_tol = FINAL_TOL if self.stop_at_goal else TEST_WP_TOL
        if self.target_index == self.stop_after and distance <= arrival_tol:
            self.done = True
            self.phase = 'DONE'
            print(f'GOAL REACHED ({self.pose.x:.3f}, {self.pose.y:.3f}); '
                  'motors stopped.', flush=True)
            return 0., 0.

        # A deviated pose is NOT an abort. Continue to the waypoint if the
        # physical robot fits; otherwise ask the chosen planner for a fresh
        # feasible route from this measured pose to the destination.
        if not self.segment_drivable(pos, target):
            if not self.replan_from_pose(now):
                self.phase = 'WAITING_FOR_CLEAR_ROUTE'
                if now - self.last_block_log > 2.0:
                    self.last_block_log = now
                    print('Next leg blocked for the physical robot; '
                          'waiting for another pose/replan, not aborting.', flush=True)
                return 0., 0.
            target = self.waypoints_xy[self.target_index]
            distance = math.dist(pos, target)

        # Look past a nearby intermediate waypoint to avoid atan2 bearing
        # jitter. Final target is always the actual goal, without overshoot.
        aim = target
        if distance < .20 and self.target_index < self.stop_after:
            previous = self.waypoints_xy[self.target_index - 1]
            yaw = math.atan2(target[1] - previous[1], target[0] - previous[0])
            aim = (target[0] + .15 * math.cos(yaw),
                   target[1] + .15 * math.sin(yaw))
        desired_yaw = math.atan2(aim[1] - pos[1], aim[0] - pos[0])
        heading_error = wrap_angle(desired_yaw - self.pose.yaw)
        self.phase = 'TRAVEL'
        if abs(heading_error) > TRAVEL_HEADING_TOL:
            return self.spin_towards(heading_error, now)

        # Steer continuously for small heading corrections instead of making
        # the robot pause and pulse-spin every time the estimate changes.
        speed = .10 if distance < .20 else .14
        if abs(heading_error) > math.radians(15):
            speed = min(speed, .10)
        steer = max(-.055, min(.055, .17 * heading_error))
        return (speed * (1 + self.model.trim) - steer,
                speed * (1 - self.model.trim) + steer)

    def lost_vision_command(self, now):
        # Immediately stop forward motion on stale AprilTag poses, but try a
        # small left/right camera sweep. Do not quit merely for lost sight.
        if self.lost_at is None:
            self.lost_at = now
            self.search_step = 0
            self.search_step_at = now
            self.search_sweeps = 0
            self.spin_command = None
            self.reacquire_frames = 0
            self.reacquire_seq = -1
            self.reacquire_pose = None
            print('VISION LOST — bounded left/right search for AprilTags.', flush=True)
        if not self.rotation_drivable((self.pose.x, self.pose.y)):
            self.phase = 'WAITING_FOR_TAG_NO_SPIN'
            return 0., 0.

        while (self.search_sweeps < SEARCH_SWEEPS and
               now - self.search_step_at >= SEARCH[self.search_step][2]):
            self.search_step_at += SEARCH[self.search_step][2]
            self.search_step += 1
            if self.search_step >= len(SEARCH):
                self.search_sweeps += 1
                self.search_step = 0
        if self.search_sweeps >= SEARCH_SWEEPS:
            # Remain alive, listen for TF, and auto-resume if tags reappear.
            # No endless unobserved spinning or blind motion.
            self.phase = 'WAITING_FOR_TAG'
            return 0., 0.
        self.phase = 'SEARCHING_TAGS'
        return SEARCH[self.search_step][:2]

    def tick(self):
        now = time.monotonic()
        self.update_pose(now)
        left = right = 0.0
        fresh = self.pose is not None and now - self.pose.stamp <= VISION_TIMEOUT
        if self.done:
            pass
        elif self.armed and now - self.start_at > MAX_RUN_SECONDS:
            # One hard overall limit in case the program is left running.
            self.done, self.phase = True, 'RUN_LIMIT_STOP'
        elif not self.armed:
            motor_ready = not self.drive or self.motor_pub.get_subscription_count() > 0
            if fresh and motor_ready:
                near = (math.dist((self.pose.x, self.pose.y), self.world.start) < .12
                        and abs(wrap_angle(self.pose.yaw - math.radians(
                            self.world.cfg['start_yaw_deg']))) < math.radians(18))
                if near:
                    self.armed, self.start_at = True, now
                    self.phase = 'TRAVEL'
                    print('START — current pose and motor node ready.', flush=True)
                else:
                    self.phase = 'WAITING_AT_START'
            else:
                self.phase = 'WAITING_FOR_MOTOR' if not motor_ready else 'WAITING_FOR_TAG'
        elif self.drive and self.motor_pub.get_subscription_count() == 0:
            # Pause and automatically resume on reconnection; no terminal abort.
            self.phase = 'WAITING_FOR_MOTOR'
        elif not fresh:
            left, right = self.lost_vision_command(now)
        elif self.lost_at is not None:
            # Two NEW, mutually consistent detections; not timer ticks or a
            # requirement to return to the original corridor.
            if self.pose_seq != self.reacquire_seq:
                self.reacquire_seq = self.pose_seq
                candidate = (self.pose.x, self.pose.y, self.pose.yaw)
                previous = self.reacquire_pose
                if (previous is not None and
                    math.hypot(candidate[0] - previous[0],
                               candidate[1] - previous[1]) < 0.10 and
                    abs(wrap_angle(candidate[2] - previous[2])) < math.radians(22)):
                    self.reacquire_frames += 1
                else:
                    self.reacquire_frames = 1
                self.reacquire_pose = candidate
            if self.reacquire_frames >= 2:
                self.lost_at = None
                self.reacquire_pose = None
                self.spin_command = None
                self.phase = 'TRAVEL'
                print('VISION REACQUIRED — continuing from current pose.', flush=True)
            else:
                self.phase = 'REACQUIRING_TAGS'
        elif not self.rotation_drivable((self.pose.x, self.pose.y)):
            # Physical collision envelope only; no abort for crossing the
            # chosen offline C-space boundary or being off the original route.
            self.phase = 'WAITING_FOR_CLEAR_POSE'
        else:
            left, right = self.command_for_waypoint(now)
        self.send(left, right)
        if now - self.last_log > .5:
            self.last_log = now
            if self.pose:
                elapsed = now - self.start_at if self.start_at is not None else 0.
                self.writer.writerow([f'{elapsed:.2f}', f'{self.pose.x:.4f}',
                                      f'{self.pose.y:.4f}', f'{math.degrees(self.pose.yaw):.1f}',
                                      self.pose.ids, self.target_index, self.phase,
                                      f'{self.last_command[0]:.3f}', f'{self.last_command[1]:.3f}'])
                self.log_file.flush()
                print(f'WP{self.target_index} {self.phase} '
                      f'({self.pose.x:.2f},{self.pose.y:.2f},'
                      f'{math.degrees(self.pose.yaw):+.0f}deg) tags={self.pose.ids} '
                      f'motor={self.last_command}', flush=True)
            else:
                print(self.phase, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('shortest','safety'), required=True)
    p.add_argument('--drive', action='store_true', help='Enable wheel commands, OFF by default')
    p.add_argument('--stop-after-wp', type=int, help='Test only first 1,2,.. waypoint')
    a = p.parse_args()
    rclpy.init()
    node = None
    try:
        node = HW4Navigation(a.mode, a.drive, a.stop_after_wp)
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=.1)
    except KeyboardInterrupt:
        print('STOP requested by keyboard.', flush=True)
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
