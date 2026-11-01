#!/usr/bin/env python3
"""HW2 AprilTag closed-loop waypoint navigation for Waveshare Rover / ROS 2 Jazzy.
Version 4: safer WP2 Tag 3 -> 4/5 handoff, stable-heading reacquisition.

Standalone script: python3 homework2/visual_navigation.py [--drive]
Dry-run by default. With --drive, publishes Float32MultiArray to /motor_commands.
Requires the existing camera, image rectification, AprilTag detector, and
robot_control/motor_control nodes; never run alongside HW1 velocity_control.
"""

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from tf2_msgs.msg import TFMessage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'robot_control'))
from robot_control.waypoint_controller import Model, wrap_angle  # Reuse HW1
from check_localization import (                        # Reuse validated extrinsics/math
    NORMALS, P_ROBOT_CAMERA, R_ROBOT_CAMERA, UP, quaternion_matrix,
)

# All physical coordinates and tolerances below are in meters or radians.
POSITION_TOL = 0.07                   # arrival if within 7 cm
FINAL_HEADING_TOL = math.radians(8)   # arrival heading within 8 degrees
TRAVEL_HEADING_TOL = math.radians(12)
VISION_TIMEOUT = 0.75                 # stop if no accepted camera pose for 0.75 s
MAX_RUN_SECONDS = 180.0               # safety cutoff after starting
DRIVE_FAST = 0.15                     # conservative vs HW1's 0.20
DRIVE_SLOW = 0.115                    # used near a waypoint
# One short HW1-speed spin pulse, then stop and wait for NEW visual feedback.
SPIN_MAX_PULSE_S = 0.18
SPIN_MIN_PULSE_S = 0.10
SPIN_SETTLE_S = 0.25
WP_HOLD_SECONDS = 2.0  # Stop for 2 seconds after reaching each waypoint
# Avoid in-place corrections caused by a fluctuating bearing to a very close
# waypoint (particularly WP2 after the 1 m northward segment).
APPROACH_DISTANCE_M = 0.23
APPROACH_LOOKAHEAD_M = 0.22
# Before commanding a reversal during the final 180-degree heading alignment,
# require several NEW visual observations after the previous CCW turn.
WP2_REVERSE_CONFIRM_FRAMES = 3
# During WP2 final alignment, Tag 3 has produced a reversed/ambiguous yaw.
# Prefer mapped Tags 4/5, which are visible during the westward camera handoff.
WP2_ALIGN_TAGS = {4, 5}
SYNC_WINDOW_S = 0.25             # avoid mixing observations taken far apart
REACQUIRE_FRAMES = 3             # large heading changes require 3 distinct images
REACQUIRE_MAX_POSITION_M = 0.30  # only re-anchor when still close to WP2
REACQUIRE_JUMP_LIMIT_DEG = 125  # turn from +90 to +180 can have been missed

# HW2 waypoints expressed in homework coordinate UNITS; unit_m from landmarks.yaml
WAYPOINTS_UNITS = [(0, 0, 0), (1, 0, 0), (1, 2, math.pi), (0, 0, 0)]


@dataclass
class Pose:
    x: float
    y: float
    yaw: float
    stamp: float
    ids: str


class VisualNavigation(Node):
    def __init__(self, drive, stop_after):
        super().__init__('hw2_visual_navigation')
        self.drive = drive
        self.stop_after = stop_after
        with open(ROOT / 'homework2/config/landmarks.yaml', encoding='utf-8') as f:
            world = yaml.safe_load(f)
        unit = float(world['unit_m'])
        self.waypoints = [(x * unit, y * unit, yaw) for x, y, yaw in WAYPOINTS_UNITS]

        # Read HW1's actual calibrated parameters instead of duplicating them.
        with open(ROOT / 'homework1/config/ballroom.yaml', encoding='utf-8') as f:
            calibration = yaml.safe_load(f)['velocity_control']['ros__parameters']
        self.model = Model()
        for key, value in calibration.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value)

        self.tags = {}
        for raw_id, config in world['tags'].items():
            normal = np.array(NORMALS[config['facing']], dtype=float)
            r_world_tag = np.column_stack((np.cross(UP, normal), UP, normal))
            self.tags[int(raw_id)] = (
                np.array(config['position_m'], dtype=float), r_world_tag
            )

        self.last_by_tag = {}           # tag ID -> (x,y,yaw,range,receive_time)
        self.pose = None
        self.pose_seq = 0
        self.last_fused_sample = 0.0
        self.target_index = 1          # start point is already waypoint 0
        self.phase = 'TRAVEL'           # TRAVEL -> ALIGN -> HOLD -> next waypoint
        self.hold_until = 0.0
        self.armed = False
        self.ready_since = None
        self.start_at = None
        self.last_motion_at = 0.0
        self.last_log = 0.0
        self.spin_command = None      # None or (left, right) while pulse/settling
        self.spin_pulse_until = 0.0
        self.spin_settle_until = 0.0
        self.spin_observed_seq = -1
        self.done = False
        self.wp2_reverse_frames = 0
        self.wp2_reverse_last_seq = -1
        self.last_command = (0.0, 0.0)
        self.last_spin_direction = 0
        self.last_spin_at = 0.0
        self.reacquire_by_tag = {}      # id -> (x,y,yaw,recv_time,consecutive_frames)
        self.last_reject_log = 0.0


        self.motor_pub = self.create_publisher(Float32MultiArray, '/motor_commands', 10)
        self.create_subscription(TFMessage, '/tf', self.on_tf, 10)
        self.create_timer(0.05, self.tick)  # 20 Hz, covers motor driver's 0.15s watchdog

        folder = ROOT / 'homework2/output'
        folder.mkdir(exist_ok=True, parents=True)
        label = 'drive' if drive else 'preview'
        path = folder / f'trajectory_{label}_{datetime.now():%Y%m%d_%H%M%S}.csv'
        self.log_file = open(path, 'w', encoding='utf-8', newline='')
        self.writer = csv.writer(self.log_file)
        self.writer.writerow(['elapsed_s', 'x_m', 'y_m', 'heading_deg',
                              'tag_ids', 'target_waypoint', 'phase', 'motor_left', 'motor_right'])
        self.log_file.flush()

        print(f'HW1 calibration: spin_cmd={self.model.spin_cmd:.2f}, '
              f'speed_per_cmd={self.model.speed_per_cmd:.2f}, '
              f'wheel_base={self.model.wheel_base:.3f}, '
              f'ccw_offset={self.model.spin_offset:.1f} deg, '
              f'cw_offset={self.model.cw_offset:.1f} deg', flush=True)
        print(f'Waypoints (meters): {self.waypoints}', flush=True)
        print(f'Trajectory CSV: {path}', flush=True)
        print('MODE: DRIVE ENABLED' if drive else 'MODE: PREVIEW ONLY (motors disabled)', flush=True)
        print(f'Stop after WP{stop_after}; waiting for visual pose near Start (0,0,0)...', flush=True)

    def on_tf(self, message):
        now = time.monotonic()
        for transform in message.transforms:
            if transform.header.frame_id.lstrip('/') != 'camera_frame':
                continue
            child = transform.child_frame_id.lstrip('/')
            if not child.startswith('tag_'):
                continue
            try:
                tag_id = int(child[4:])
            except ValueError:
                continue
            if tag_id not in self.tags:
                continue

            p_world_tag, r_world_tag = self.tags[tag_id]
            t = transform.transform.translation
            p_camera_tag = np.array([t.x, t.y, t.z], dtype=float)
            q = transform.transform.rotation
            r_camera_tag = quaternion_matrix((q.x, q.y, q.z, q.w))

            # Same tested camera-to-world calculation as check_localization.py
            r_world_camera = r_world_tag @ r_camera_tag.T
            r_world_robot = r_world_camera @ R_ROBOT_CAMERA.T
            p_world_camera = p_world_tag - r_world_camera @ p_camera_tag
            p_world_robot = p_world_camera - r_world_robot @ P_ROBOT_CAMERA
            yaw = math.atan2(r_world_robot[1, 0], r_world_robot[0, 0])
            if np.isfinite(p_world_robot).all() and math.isfinite(yaw):
                self.last_by_tag[tag_id] = (
                    float(p_world_robot[0]), float(p_world_robot[1]), yaw,
                    float(np.linalg.norm(p_camera_tag)), now
                )

    def update_pose(self, now):
        # Fuse only fresh, nearly simultaneous TF samples. A still-fresh Tag 3
        # estimate from before a turn must not be combined with a new Tag 5.
        fresh = [(tag_id, *rec) for tag_id, rec in self.last_by_tag.items()
                 if now - rec[4] < VISION_TIMEOUT]
        if not fresh:
            return
        latest = max(p[5] for p in fresh)
        if latest <= self.last_fused_sample:
            return
        self.last_fused_sample = latest
        fresh = [p for p in fresh if latest - p[5] <= SYNC_WINDOW_S]

        wp2_align = self.target_index == 2 and self.phase == 'ALIGN'
        if wp2_align:
            # In field runs, north-facing Tag 3 gave angles that moved BACKWARD
            # during CCW turns (90 -> 70 -> 69 deg). While aligning WP2, only
            # trust the 4/5 corridor; with neither visible, STOP rather than
            # issue extra motor pulses on the basis of Tag 3 alone.
            fresh = [p for p in fresh if p[0] in WP2_ALIGN_TAGS]

        accepted = []
        rejected = []
        verified_reanchor = False
        for tag_id, x, y, yaw, distance, recv in fresh:
            if self.pose is None:
                if math.hypot(x, y) > 0.18 or abs(wrap_angle(yaw)) > math.radians(20):
                    rejected.append((tag_id, 'not near Start'))
                    continue
            else:
                dt = now - self.pose.stamp
                position_jump = math.hypot(x - self.pose.x, y - self.pose.y)
                yaw_delta = wrap_angle(yaw - self.pose.yaw)
                max_position_jump = min(0.60, 0.22 + 0.22 * dt)
                max_yaw_jump = min(math.radians(60), math.radians(30) + 0.25 * dt)
                if position_jump > max_position_jump:
                    rejected.append((tag_id, 'position jump'))
                    continue

                if wp2_align:
                    # A +X/+Y floor-plan heading near +180 can be reported as
                    # -180; wrapped delta handles that. Keep progress CCW:
                    # an implausible backwards estimate must not cause the
                    # controller to keep turning on a false +69 deg reading.
                    if yaw_delta < -math.radians(10):
                        # Allow a bounded clockwise correction only if the
                        # controller actually commanded a clockwise spin.
                        clockwise_correction = (
                            self.last_spin_direction == -1
                            and now - self.last_spin_at < 3.0
                            and abs(yaw_delta) <= max_yaw_jump
                        )

                        if not clockwise_correction:
                            rejected.append((tag_id, 'unexplained reverse yaw during WP2 align'))
                            continue
                    if math.hypot(x - self.waypoints[2][0],
                                  y - self.waypoints[2][1]) > REACQUIRE_MAX_POSITION_M:
                        rejected.append((tag_id, 'outside WP2 vicinity'))
                        continue
                    # The camera can switch from Tag 4 to Tag 5 after a spin,
                    # making the true pose suddenly jump >30deg relative to an
                    # old/incorrect vision estimate. Never accept it on ONE
                    # observation; require 3 consecutive NEW, consistent frames.
                    if yaw_delta > max_yaw_jump:
                        if (self.last_spin_direction <= 0 or
                                yaw_delta > math.radians(REACQUIRE_JUMP_LIMIT_DEG)):
                            rejected.append((tag_id, 'large unexplained yaw jump'))
                            continue
                        prior = self.reacquire_by_tag.get(tag_id)
                        if (prior is not None and recv > prior[3] and
                                recv - prior[3] < 1.2 and
                                math.hypot(x - prior[0], y - prior[1]) < 0.09 and
                                abs(wrap_angle(yaw - prior[2])) < math.radians(12)):
                            count = prior[4] + 1
                        else:
                            count = 1
                        self.reacquire_by_tag[tag_id] = (x, y, yaw, recv, count)
                        if count < REACQUIRE_FRAMES:
                            rejected.append((tag_id, f'reconfirming heading {count}/{REACQUIRE_FRAMES}'))
                            continue
                        verified_reanchor = True
                        print(f'WP2 verified re-localization from Tag {tag_id}: '
                              f'heading {math.degrees(yaw):+.1f} deg '
                              f'({count} fresh frames)', flush=True)
                elif abs(yaw_delta) > max_yaw_jump:
                    rejected.append((tag_id, 'yaw jump'))
                    continue
            accepted.append((tag_id, x, y, yaw, distance))

        if not accepted:
            if now - self.last_reject_log > 2.0 and (wp2_align or rejected):
                self.last_reject_log = now
                print('VISION FILTER: no acceptable WP2 pose; holding STOP. '
                      + ('rejected=' + repr(rejected) if rejected else
                         'only Tag 3/other tags visible, waiting for Tag 4/5'),
                      flush=True)
            return

        # Use closest accepted tag as an anchor, but do not fuse contradictory
        # poses: an abrupt source-switch artifact should not bias robot yaw.
        anchor = min(accepted, key=lambda item: item[4])
        consistent = [p for p in accepted
                      if math.hypot(p[1] - anchor[1], p[2] - anchor[2]) < 0.18
                      and abs(wrap_angle(p[3] - anchor[3])) < math.radians(18)]
        weights = [1.0 / max(p[4], 0.4)**2 for p in consistent]
        total = sum(weights)
        x = sum(w * p[1] for p, w in zip(consistent, weights)) / total
        y = sum(w * p[2] for p, w in zip(consistent, weights)) / total
        yaw = math.atan2(sum(w * math.sin(p[3]) for p, w in zip(consistent, weights)),
                         sum(w * math.cos(p[3]) for p, w in zip(consistent, weights)))
        if self.pose is not None:
            # Smooth ordinary pose updates, but don't slowly blend a verified
            # 90deg reacquisition into a stale 69deg pose (causes lag/overspin).
            alpha = 1.0 if verified_reanchor else 0.7
            x = self.pose.x + alpha * (x - self.pose.x)
            y = self.pose.y + alpha * (y - self.pose.y)
            yaw = wrap_angle(self.pose.yaw + alpha * wrap_angle(yaw - self.pose.yaw))
        self.pose = Pose(x, y, yaw, now, ','.join(str(p[0]) for p in consistent))
        self.pose_seq += 1
        if verified_reanchor:
            self.reacquire_by_tag.clear()

    def send(self, left=0.0, right=0.0):
        if not self.drive or self.done:
            left = right = 0.0
        # Don't silently clip away HW1's calibrated spin_cmd=0.35.
        command_limit = max(self.model.drive_cmd, self.model.spin_cmd)
        left = max(-command_limit, min(command_limit, float(left)))
        right = max(-command_limit, min(command_limit, float(right)))
        msg = Float32MultiArray()
        msg.data = [left, right]
        self.motor_pub.publish(msg)
        self.last_command = (left, right)
        if abs(left) + abs(right) > 0.0001:
            self.last_motion_at = time.monotonic()

    def pending_spin(self, now):
        """Finish a short turn pulse and wait for *new* camera feedback.

        Camera is only about 2.4 fps: sending 0.35 continuously until the
        next frame can easily overshoot. Turning is still visual closed-loop.
        """
        if self.spin_command is None:
            return None
        if now < self.spin_pulse_until:
            return self.spin_command
        # Stop wheels and allow both inertial coast and camera frame latency.
        if (now < self.spin_settle_until
                or self.pose_seq <= self.spin_observed_seq
                or self.pose.stamp <= self.spin_pulse_until):
            return 0.0, 0.0
        self.spin_command = None
        return None

    def spin_towards(self, error, now):
        """Use the HW1 calibrated spin command in brief, observed segments."""
        clockwise = error < 0
        angular_rate = self.model.spin_rate(clockwise=clockwise)
        coast_deg = self.model.cw_offset if clockwise else self.model.spin_offset
        # HW1's offset estimates the continued turn after stopping, so plan
        # conservatively and use camera measurements between every short pulse.
        effective_angle = max(0.0, abs(error) - math.radians(coast_deg))
        duration = max(SPIN_MIN_PULSE_S,
                       min(SPIN_MAX_PULSE_S, effective_angle / angular_rate))
        direction = math.copysign(1.0, error)
        speed = self.model.spin_cmd     # 0.35 from homework1/config/ballroom.yaml
        self.spin_command = (-direction * speed, direction * speed)
        self.last_spin_direction = int(direction)
        self.last_spin_at = now
        self.spin_observed_seq = self.pose_seq
        self.spin_pulse_until = now + duration
        self.spin_settle_until = self.spin_pulse_until + SPIN_SETTLE_S
        print(f'SPIN pulse: error={math.degrees(error):+.1f} deg, '
              f'L={self.spin_command[0]:+.2f} R={self.spin_command[1]:+.2f}, '
              f'duration={duration:.2f} s (HW1 calibration)', flush=True)
        return self.spin_command

    def command_for_waypoint(self, now):
        # Hold the rover still for 2 seconds after each confirmed waypoint.
        if self.phase == 'HOLD':
            if now < self.hold_until:
                return 0.0, 0.0

            print(f'WP{self.target_index} HOLD COMPLETE: '
                f'{WP_HOLD_SECONDS:.1f} seconds elapsed.', flush=True)

            if self.target_index == self.stop_after:
                self.done = True
                self.phase = 'DONE'
                print('ROUTE COMPLETE — motors stopped.', flush=True)
            else:
                self.target_index += 1
                self.phase = 'TRAVEL'

            return 0.0, 0.0

        # Continue/finish any short turn before changing the movement phase.
        active_spin = self.pending_spin(now)
        if active_spin is not None:
            return active_spin
        goal_x, goal_y, goal_yaw = self.waypoints[self.target_index]
        distance = math.hypot(goal_x - self.pose.x, goal_y - self.pose.y)

        if self.phase == 'TRAVEL':
            if distance <= POSITION_TOL:
                self.phase = 'ALIGN'
                self.wp2_reverse_frames = 0
                self.wp2_reverse_last_seq = -1
                return 0.0, 0.0
            if distance < APPROACH_DISTANCE_M:
                # Aim slightly beyond the waypoint along the *incoming segment*.
                # Otherwise, centimeter-scale measurement noise near WP2 makes
                # atan2(goal_y-y, goal_x-x) swing by tens of degrees and trigger
                # the unwanted clockwise "correction" seen during the field run.
                prev_x, prev_y, _ = self.waypoints[self.target_index - 1]
                segment_yaw = math.atan2(goal_y - prev_y, goal_x - prev_x)
                aim_x = goal_x + APPROACH_LOOKAHEAD_M * math.cos(segment_yaw)
                aim_y = goal_y + APPROACH_LOOKAHEAD_M * math.sin(segment_yaw)
                desired_yaw = math.atan2(aim_y - self.pose.y, aim_x - self.pose.x)
            else:
                desired_yaw = math.atan2(goal_y - self.pose.y, goal_x - self.pose.x)
            heading_error = wrap_angle(desired_yaw - self.pose.yaw)
            if abs(heading_error) > TRAVEL_HEADING_TOL:
                return self.spin_towards(heading_error, now)
            # Steering and trim come from HW1's motor model/calibration.
            speed = DRIVE_SLOW if distance < 0.23 else DRIVE_FAST
            steer = max(-0.045, min(0.045, 0.13 * heading_error))
            left = speed * (1.0 + self.model.trim) - steer
            right = speed * (1.0 - self.model.trim) + steer
            return left, right

        heading_error = wrap_angle(goal_yaw - self.pose.yaw)
        if distance > POSITION_TOL + 0.035:
            # Recheck position after turning; don't accept a heading-only success.
            self.phase = 'TRAVEL'
            return 0.0, 0.0
        if abs(heading_error) > FINAL_HEADING_TOL:
            if self.target_index == 2 and heading_error < 0.0:
                # Goal 180 deg is exactly the +/- pi wrap boundary.  A single
                # noisy observation can change the preferred turn from CCW to
                # CW.  Pause until a genuine clockwise correction is confirmed
                # on multiple NEW camera frames; never turn blindly on stale TF.
                if self.pose_seq != self.wp2_reverse_last_seq:
                    self.wp2_reverse_last_seq = self.pose_seq
                    self.wp2_reverse_frames += 1
                if self.wp2_reverse_frames < WP2_REVERSE_CONFIRM_FRAMES:
                    return 0.0, 0.0
            else:
                self.wp2_reverse_frames = 0
                self.wp2_reverse_last_seq = -1
            return self.spin_towards(heading_error, now)
        if now - self.last_motion_at < 0.35:
            return 0.0, 0.0  # wait for the rover to settle

        print(f'REACHED WP{self.target_index}: x={self.pose.x:+.3f} m '
            f'y={self.pose.y:+.3f} m yaw={math.degrees(self.pose.yaw):+.1f} deg', flush=True)

        # Stop the rover for 2 seconds before proceeding to the next waypoint.
        self.phase = 'HOLD'
        self.hold_until = now + WP_HOLD_SECONDS

        print(f'WP{self.target_index} HOLD START: '
            f'motors stopped for {WP_HOLD_SECONDS:.1f} seconds.',
            flush=True)

        return 0.0, 0.0

    def tick(self):
        now = time.monotonic()
        self.update_pose(now)
        if self.done:
            self.send(0, 0)
            return

        left, right = 0.0, 0.0
        if self.pose is None or now - self.pose.stamp > VISION_TIMEOUT:
            status = 'WAITING/LOST VISION — STOP'
            self.ready_since = None if not self.armed else self.ready_since
        elif not self.armed:
            near_start = (math.hypot(self.pose.x, self.pose.y) < 0.15 and
                          abs(wrap_angle(self.pose.yaw)) < math.radians(18))
            motor_ready = not self.drive or self.motor_pub.get_subscription_count() > 0
            if near_start and motor_ready:
                if self.ready_since is None:
                    self.ready_since = now
                elif now - self.ready_since >= 2.0:
                    self.armed = True
                    self.start_at = now
                    print(f'START — navigating until WP{self.stop_after}', flush=True)
            else:
                self.ready_since = None
            status = 'ARMING / CHECK START AND MOTOR NODE'
        elif now - self.start_at > MAX_RUN_SECONDS:
            self.done = True
            self.phase = 'TIMEOUT'
            status = 'MAX RUN TIME — STOP'
        elif self.drive and self.motor_pub.get_subscription_count() == 0:
            status = 'MOTOR NODE DISCONNECTED — STOP'
        else:
            left, right = self.command_for_waypoint(now)
            status = self.phase

        self.send(left, right)
        if now - self.last_log >= 0.5:
            self.last_log = now
            if self.pose:
                elapsed = (now - self.start_at) if self.start_at is not None else 0.0
                self.writer.writerow([
                    f'{elapsed:.2f}', f'{self.pose.x:.4f}', f'{self.pose.y:.4f}',
                    f'{math.degrees(self.pose.yaw):.2f}', self.pose.ids,
                    self.target_index, status, f'{self.last_command[0]:.3f}',
                    f'{self.last_command[1]:.3f}'
                ])
                self.log_file.flush()
                print(f'WP{self.target_index} {status:30} '
                      f'pose=({self.pose.x:+.2f},{self.pose.y:+.2f},'
                      f'{math.degrees(self.pose.yaw):+.0f}°) tags={self.pose.ids} '
                      f'motor=({self.last_command[0]:+.2f},{self.last_command[1]:+.2f})',
                      flush=True)
            else:
                print(status, flush=True)

    def close(self):
        try:
            for _ in range(3):
                self.send(0, 0)
                time.sleep(0.05)
        except Exception:
            pass
        self.log_file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drive', action='store_true',
                        help='Enable real wheel commands; default is pose preview only')
    parser.add_argument('--stop-after-wp', type=int, choices=[1, 2, 3], default=3,
                        help='Stop the run after reaching WP1, WP2, or Finish (default: 3)')
    args = parser.parse_args()
    rclpy.init()
    node = VisualNavigation(args.drive, args.stop_after_wp)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        print('Interrupted — stopping motors.', flush=True)
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
