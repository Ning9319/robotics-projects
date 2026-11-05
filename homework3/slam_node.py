#!/usr/bin/env python3
"""HW3 EKF-SLAM node: drive a square / octagon with stop-and-observe and build a map.

Needs camera + image_proc rectify + apriltag (homework3/config/tags_hw3.yaml) and,
with --drive, robot_control motor_control. Never run with HW1/HW2 controllers.

    python3 homework3/slam_node.py                                   # preview: never moves
    python3 homework3/slam_node.py --path square --side 1.2 --laps 3 --drive
    python3 homework3/slam_node.py --path octagon --side 0.5 --laps 3 --drive

Each leg is split into short segments. One cycle:
  TURN    open-loop in-place spin toward the next waypoint (HW1 model), only if
          the heading error is large
  DRIVE   straight segment (open loop, HW1 speed); small heading errors are
          removed with a gentle differential steer during the segment
  SETTLE  wheels stopped, rover coasts to rest; camera frames are ignored
  OBSERVE wheels stopped, ~1 s of camera frames are averaged per tag, then ONE
          EKF update per tag is applied at the end of the stop
The EKF predicts from the commanded motion while moving and is corrected only
while the rover stands still (no motion blur, no camera-latency error).
Frames from one stop are averaged instead of used as separate updates: their
errors are almost fully correlated (Step 2: frame jitter 0.15 cm vs. 0.3+ cm
error between positions), and repeated updates of a just-initialized landmark
from the same pose make the EKF overconfident about the robot heading.
Waypoints are in the SLAM frame (start pose = origin), so the path is closed
loop on the SLAM estimate. The filter starts with NO landmarks and adds every
tag ID the first time it is seen; it never reads the ground-truth map.
"""

import argparse
import csv
import json
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

from ekf_slam import EKFSLAM, NoiseParams
from tag_measurement import measure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'robot_control'))
from robot_control.waypoint_controller import Model, wrap_angle  # noqa: E402  (reuse HW1)

TICK_S = 0.02                 # control / prediction period
LOG_PERIOD_S = 0.1            # trajectory.csv sampling
MIN_PULSE_S = 0.10            # shortest in-place spin command
MAX_STEER = 0.07              # max differential command while driving (~8 deg per 0.3 m segment)
MAX_ROBOT_STD_M = 0.5         # abort if the robot position estimate gets this uncertain


class Tee:
    """Write to the terminal and a log file at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            if not s.closed:
                s.write(text)

    def flush(self):
        for s in self.streams:
            if not s.closed:
                s.flush()


def make_waypoints(path, side, laps, direction):
    """Polygon vertices in the SLAM frame, starting at the origin facing +x."""
    sides, exterior = (4, math.pi / 2) if path == 'square' else (8, math.pi / 4)
    turn = exterior if direction == 'ccw' else -exterior
    x = y = heading = 0.0
    points = []
    for lap in range(laps):
        for k in range(sides):
            x += side * math.cos(heading)
            y += side * math.sin(heading)
            points.append((x, y, lap + 1, k + 1))
            heading += turn
    return points


class SlamNode(Node):
    def __init__(self, args):
        super().__init__('hw3_ekf_slam')
        self.args = args
        self.drive = args.drive

        # --- HW1 motion model and calibration (same floor as HW1/HW2)
        with open(ROOT / 'homework1/config/ballroom.yaml', encoding='utf-8') as f:
            hw1 = yaml.safe_load(f)['velocity_control']['ros__parameters']
        self.model = Model()
        for key, value in hw1.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value)
        # Optional per-floor overrides (e.g. a test on a different floor at home).
        # The HW3 field is on the HW1 floor, so the defaults need no override.
        if args.speed_per_cmd is not None:
            self.model.speed_per_cmd = args.speed_per_cmd
        if args.spin_offset_deg is not None:
            self.model.spin_offset = args.spin_offset_deg
        if args.cw_offset_deg is not None:
            self.model.cw_offset = args.cw_offset_deg
        if args.trim is not None:
            self.model.trim = args.trim
        self.drive_wheel_base = args.drive_wheel_base

        # --- sensor correction and R from Step 2 (homework3/config/sensor_calibration.yaml)
        with open(ROOT / 'homework3/config/sensor_calibration.yaml', encoding='utf-8') as f:
            sensor = yaml.safe_load(f)
        self.range_scale = float(sensor['range_scale'])
        self.range_offset = float(sensor['range_offset_m'])
        self.bearing_offset = math.radians(float(sensor['bearing_offset_deg']))
        q = args.q_scale
        default_q = NoiseParams()
        self.noise = NoiseParams(
            var_forward_per_m=q * default_q.var_forward_per_m,
            var_lateral_per_m=q * default_q.var_lateral_per_m,
            var_heading_per_rad=q * default_q.var_heading_per_rad,
            var_heading_per_m=q * default_q.var_heading_per_m,
            range_std_base=args.r_scale * float(sensor['range_std_base_m']),
            range_std_per_m=args.r_scale * float(sensor['range_std_per_m']),
            bearing_std=args.r_scale * math.radians(float(sensor['bearing_std_deg'])),
        )
        self.turn_coast_var = math.radians(args.turn_coast_std_deg) ** 2
        self.slam = EKFSLAM(self.noise)

        side = args.side if args.side is not None else (1.2 if args.path == 'square' else 0.5)
        self.side = side
        # Square corners (90 deg) -> 3 x 30 deg (limit 35 so 90-105 deg is 3 steps),
        # so a stop sees both walls; octagon corners (45 deg) stay one step (limit 50).
        self.max_turn_step_deg = args.max_turn_step_deg if args.max_turn_step_deg is not None \
            else (35.0 if args.path == 'square' else 50.0)
        self.waypoints = make_waypoints(args.path, side, args.laps, args.direction)
        self.wp_index = 0

        # --- state machine
        self.phase = 'INIT'
        self.phase_start = time.monotonic()
        self.phase_until = self.phase_start + args.init_observe
        self.started = self.phase_start
        self.command = (0.0, 0.0)
        self.command_target = (0.0, 0.0)  # wheel command of the current TURN / DRIVE
        self.motion = (0.0, 0.0)         # (v, omega) the EKF integrates while moving
        self.last_predict = None
        self.turn_coast = 0.0            # signed coast angle to add when a spin ends
        self.turn_tries = 0
        self.lost_stops = 0
        self.recoveries = 0
        self.searching = False
        self.search_steps = 0            # SEARCH turns in the current search
        self.search_dir = 1.0
        self.search_episodes = 0         # searches started for the current vertex
        self.searches = 0                # total SEARCH turns in the run
        self.last_turn_sign = 1.0
        self.last_motion = None
        self.stop_known = 0              # mapped tags successfully updated in this stop
        self.done = False
        self.finish_reason = None
        self.stop_buffer = {}            # tag_id -> [(range, bearing)] of the current stop
        self.accept_after = 0.0          # wall-clock time: ignore images captured earlier
        self.stop_frames = 0
        self.stop_tags = set()
        self.frame_queue = []
        self.frame_seq = 0
        self.last_log = 0.0
        self.last_preview_print = 0.0
        self.snapshots = []
        self.vertex_rows = []
        self.hold_until = 0.0

        # --- ROS
        self.motor_pub = self.create_publisher(Float32MultiArray, '/motor_commands', 10)
        self.create_subscription(TFMessage, '/tf', self.on_tf, 50)
        self.create_timer(TICK_S, self.tick)

        # --- logs
        name = args.label or f'{args.path}_{datetime.now():%Y%m%d_%H%M%S}'
        if not self.drive:
            name = 'preview_' + name
        self.run_dir = ROOT / 'homework3/output/runs' / name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Everything printed to the terminal is also saved to console.log.
        self.console = open(self.run_dir / 'console.log', 'w', encoding='utf-8')
        self.prev_stdout = sys.stdout
        self.tee = Tee(self.prev_stdout, self.console)
        sys.stdout = self.tee
        config = {
            'args': vars(args), 'side_m': side, 'waypoints': self.waypoints,
            'hw1_model': vars(self.model), 'sensor_calibration': sensor,
            'noise': vars(self.noise), 'turn_coast_std_deg': args.turn_coast_std_deg,
        }
        (self.run_dir / 'config.json').write_text(json.dumps(config, indent=2))
        self.traj_file = open(self.run_dir / 'trajectory.csv', 'w', newline='', encoding='utf-8')
        self.traj = csv.writer(self.traj_file)
        self.traj.writerow(['t_s', 'phase', 'lap', 'side', 'x_m', 'y_m', 'yaw_deg',
                            'var_xx', 'var_xy', 'var_yy', 'std_yaw_deg', 'n_landmarks',
                            'cmd_left', 'cmd_right'])
        self.obs_file = open(self.run_dir / 'observations.csv', 'w', newline='', encoding='utf-8')
        self.obs = csv.writer(self.obs_file)
        self.obs.writerow(['t_s', 'frame', 'phase', 'tag_id', 'raw_range_m', 'raw_bearing_deg',
                           'range_m', 'bearing_deg', 'view_deg', 'status', 'image_age_s'])
        self.upd_file = open(self.run_dir / 'updates.csv', 'w', newline='', encoding='utf-8')
        self.upd = csv.writer(self.upd_file)
        self.upd.writerow(['t_s', 'lap', 'side', 'tag_id', 'n_frames', 'range_m', 'bearing_deg',
                           'range_std_m', 'bearing_std_deg', 'status', 'nis',
                           'robot_x_m', 'robot_y_m', 'robot_yaw_deg', 'n_outliers'])

        m = self.model
        print(f'HW1 model: speed_per_cmd={m.speed_per_cmd}, drive_cmd={m.drive_cmd}, '
              f'spin_cmd={m.spin_cmd}, trim={m.trim}, CCW {math.degrees(m.spin_rate()):.0f} deg/s '
              f'+{m.spin_offset} deg coast, CW {math.degrees(m.spin_rate(True)):.0f} deg/s '
              f'+{m.cw_offset} deg coast, drive wheel base {self.drive_wheel_base} m', flush=True)
        print(f'Sensor correction: range*{self.range_scale:.4f}{self.range_offset:+.4f} m, '
              f'bearing -({math.degrees(self.bearing_offset):+.2f} deg); R x{args.r_scale}, '
              f'Q x{args.q_scale}', flush=True)
        print(f'Path: {args.path} side {side:.2f} m, {args.laps} lap(s), {args.direction}, '
              f'{len(self.waypoints)} waypoints', flush=True)
        print('MODE: DRIVE ENABLED' if self.drive else
              'MODE: PREVIEW (motors disabled, rover never moves)', flush=True)
        print(f'Logs: {self.run_dir}', flush=True)

    # ----------------------------------------------------------------- utils
    def now(self):
        return time.monotonic()

    def elapsed(self):
        return self.now() - self.started

    def set_phase(self, phase, duration=0.0):
        self.phase = phase
        self.phase_start = self.now()
        self.phase_until = self.phase_start + duration
        self.log_state(force=True)

    def send(self, left=0.0, right=0.0):
        if not self.drive or self.done:
            left = right = 0.0
        limit = max(self.model.drive_cmd * (1 + abs(self.model.trim)) + MAX_STEER, self.model.spin_cmd)
        left = max(-limit, min(limit, float(left)))
        right = max(-limit, min(limit, float(right)))
        msg = Float32MultiArray()
        msg.data = [left, right]
        self.motor_pub.publish(msg)
        self.command = (left, right)

    # -------------------------------------------------------------- camera
    def on_tf(self, message):
        observations = [o for o in (measure(t) for t in message.transforms) if o is not None]
        if not observations:
            return
        received = time.time()
        for o in observations:
            o['age'] = received - o['stamp']   # image capture -> arrival here (s)
        self.frame_seq += 1
        if self.phase in ('INIT', 'OBSERVE', 'HOLD'):   # rover standing still
            self.frame_queue.append((self.frame_seq, observations))
        else:
            for o in observations:
                self.log_observation(self.frame_seq, o, None, None, f'ignored_{self.phase.lower()}')

    def corrected(self, o):
        rng = self.range_scale * o['range'] + self.range_offset
        bearing = wrap_angle(o['bearing'] - self.bearing_offset)
        return rng, bearing

    def log_observation(self, frame, o, rng, bearing, status):
        self.obs.writerow([f'{self.elapsed():.3f}', frame, self.phase, o['tag_id'],
                           f'{o["range"]:.5f}', f'{math.degrees(o["bearing"]):.4f}',
                           '' if rng is None else f'{rng:.5f}',
                           '' if bearing is None else f'{math.degrees(bearing):.4f}',
                           f'{o["view_deg"]:.1f}', status, f'{o.get("age", float("nan")):.3f}'])

    def process_frames(self):
        """Buffer corrected observations of the current stop (no EKF update yet).

        Only images CAPTURED after the rover stopped (+ stamp_margin for the
        coast) are used. Detections can arrive late in bursts; an image taken
        before an in-place turn has the right range but a bearing off by the
        whole turn angle (seen in octagon_run1, 2026-11-05).
        """
        queue, self.frame_queue = self.frame_queue, []
        for frame, observations in queue:
            used = False
            for o in observations:
                rng, bearing = self.corrected(o)
                if o['stamp'] < self.accept_after:
                    self.log_observation(frame, o, rng, bearing, 'stale')
                elif rng > self.args.max_range:
                    self.log_observation(frame, o, rng, bearing, 'too_far')
                elif o['view_deg'] > self.args.max_view_deg:
                    self.log_observation(frame, o, rng, bearing, 'too_oblique')
                else:
                    self.stop_buffer.setdefault(o['tag_id'], []).append((rng, bearing))
                    self.log_observation(frame, o, rng, bearing, 'buffered')
                    used = True
            self.stop_frames += used

    def apply_stop_updates(self):
        """One EKF update per tag from the frames of this stop, then clear the buffer.

        Known landmarks first, so new ones are initialized from the corrected
        pose. A new tag ID needs min_new_frames frames in this stop (guards
        against a one-frame false detection creating a landmark).
        """
        buffer, self.stop_buffer = self.stop_buffer, {}
        averaged = {tag_id: self.average(samples) for tag_id, samples in buffer.items()}
        known = sorted(t for t in averaged if t in self.slam.index)
        new = sorted(t for t in averaged if t not in self.slam.index)

        statuses = {t: self.update_tag(t, averaged[t]) for t in known}
        if statuses and all(s == 'rejected' for s in statuses.values()):
            # Every re-observed landmark disagrees with the prediction: the
            # robot itself slipped / was bumped (motion model failed), not the
            # tags. Inflate the robot covariance and re-apply the observations,
            # otherwise the gate would reject them forever and the filter
            # could never recover.
            self.recoveries += 1
            pos, yaw = self.args.recover_std_m, math.radians(self.args.recover_std_deg)
            print(f'  RECOVER: all {len(known)} known tag(s) rejected -> inflate robot '
                  f'covariance ({100 * pos:.0f} cm, {self.args.recover_std_deg:.0f} deg) '
                  'and retry', flush=True)
            self.slam.add_robot_noise(np.diag([pos ** 2, pos ** 2, yaw ** 2]))
            for t in known:
                self.update_tag(t, averaged[t], retry=True)
        for t in new:
            self.update_tag(t, averaged[t])

    @staticmethod
    def average(samples, max_dev_bearing=math.radians(2.0), max_dev_range=0.05):
        """Mean of one stop's frames after dropping outliers around the median.

        The rover is still, so all frames should agree to ~0.1 deg / ~1 mm;
        anything farther than 2 deg / 5 cm from the median is a bad frame.
        """
        samples = np.array(samples)
        med_r = np.median(samples[:, 0])
        med_b = math.atan2(np.median(np.sin(samples[:, 1])), np.median(np.cos(samples[:, 1])))
        db = np.arctan2(np.sin(samples[:, 1] - med_b), np.cos(samples[:, 1] - med_b))
        keep = (np.abs(db) <= max_dev_bearing) & (np.abs(samples[:, 0] - med_r) <= max_dev_range)
        good = samples[keep] if keep.any() else samples
        rng = float(good[:, 0].mean())
        bearing = math.atan2(np.sin(good[:, 1]).mean(), np.cos(good[:, 1]).mean())
        d = good[:, 1] - bearing
        return (len(good), rng, bearing, float(good[:, 0].std()),
                float(np.degrees(np.arctan2(np.sin(d), np.cos(d)).std())), int((~keep).sum()))

    def update_tag(self, tag_id, averaged, retry=False):
        n, rng, bearing, r_std, b_std, n_out = averaged
        yaw_std = math.sqrt(self.slam.robot_cov[2, 2])
        if tag_id not in self.slam.index and n < self.args.min_new_frames:
            status, nis = 'unconfirmed_new', float('nan')
        elif tag_id not in self.slam.index and \
                yaw_std > math.radians(self.args.new_landmark_max_yaw_std_deg):
            # Do not map a landmark from a pose whose heading is very uncertain:
            # its position would inherit that error, and later re-observations
            # from the same spot would make the EKF falsely confident. It will
            # be added at a later stop, once a mapped tag has fixed the heading.
            status, nis = 'deferred_new', float('nan')
        else:
            self.slam.last_nis = float('nan')
            status = self.slam.update(tag_id, rng, bearing)
            nis = self.slam.last_nis
            self.stop_tags.add(tag_id)
            if status == 'updated':
                self.stop_known += 1
        x, y, yaw = self.slam.robot_pose
        lap, side_k = (self.waypoints[self.wp_index][2:] if self.wp_index < len(self.waypoints)
                       else ('', ''))
        self.upd.writerow([f'{self.elapsed():.3f}', lap, side_k, tag_id, n, f'{rng:.5f}',
                           f'{math.degrees(bearing):.4f}', f'{r_std:.5f}', f'{b_std:.4f}',
                           ('retry_' if retry else '') + status,
                           '' if math.isnan(nis) else f'{nis:.3f}',
                           f'{x:.4f}', f'{y:.4f}', f'{math.degrees(yaw):.3f}', n_out])
        if status == 'new':
            m = self.slam.landmarks()[tag_id][0]
            print(f'  NEW landmark {tag_id} at ({m[0]:+.2f}, {m[1]:+.2f}) m from {n} frames '
                  f'[{len(self.slam.index)} mapped]', flush=True)
        elif status == 'rejected':
            print(f'  tag {tag_id} {"still " if retry else ""}REJECTED by gate (NIS {nis:.1f})',
                  flush=True)
        return status

    # ------------------------------------------------------------ motion
    def integrate_motion(self, now):
        """EKF prediction with the commanded motion since the last call."""
        if self.last_predict is not None:
            dt = now - self.last_predict
            v, omega = self.motion
            if dt > 0 and (v or omega):
                self.slam.predict(v * dt, omega * dt)
        self.last_predict = now

    def start_turn(self, error, label='TURN', count_try=True):
        ccw = error > 0
        omega = self.model.spin_rate(clockwise=not ccw)
        coast = math.radians(self.model.spin_offset if ccw else self.model.cw_offset)
        duration = min(3.0, max(MIN_PULSE_S, (abs(error) - coast) / omega))
        sign = 1.0 if ccw else -1.0
        c = self.model.spin_cmd
        self.command_target = (-sign * c, sign * c)
        self.motion = (0.0, sign * omega)
        self.turn_coast = sign * coast
        self.last_predict = self.now()
        self.last_motion = 'TURN'
        if count_try:
            self.turn_tries += 1
            self.last_turn_sign = sign
        print(f'  {label} {math.degrees(error):+.1f} deg: {duration:.2f} s spin + '
              f'{math.degrees(coast):.1f} deg coast', flush=True)
        self.set_phase('TURN', duration)

    def start_drive(self, distance, heading_error):
        m = self.model
        speed = m.speed_per_cmd * m.drive_cmd
        duration = distance / speed
        # Remove a small heading error during the segment: omega = k (R-L) / W.
        # W here is the EFFECTIVE wheelbase while driving forward: a skid-steer
        # rover turns much less per wheel-speed difference while driving than
        # when spinning in place (field fit 2026-11-05: 0.34x the spin value).
        # Trim only cancels the rover's physical drift, so it is NOT part of the
        # predicted rotation.
        W = self.drive_wheel_base
        steer = heading_error / duration * W / (2 * m.speed_per_cmd)
        steer = max(-MAX_STEER, min(MAX_STEER, steer))
        left = m.drive_cmd * (1 + m.trim) - steer
        right = m.drive_cmd * (1 - m.trim) + steer
        self.command_target = (left, right)
        self.motion = (speed, 2 * m.speed_per_cmd * steer / W)
        self.last_predict = self.now()
        self.last_motion = 'DRIVE'
        self.set_phase('DRIVE', duration)

    def stop_motion(self, now):
        self.integrate_motion(now)
        if self.phase == 'TURN':
            # Coast after the spin stops (model value), plus a per-turn heading
            # uncertainty: short spins are dominated by static friction / coast.
            if self.turn_coast:
                self.slam.predict(0.0, self.turn_coast)
            self.slam.add_robot_noise(np.diag([0.0, 0.0, self.turn_coast_var]))
        self.motion = (0.0, 0.0)
        self.turn_coast = 0.0
        self.command_target = (0.0, 0.0)
        # Wheels stop now: only images captured after the coast are trusted.
        self.accept_after = time.time() + self.args.stamp_margin
        self.set_phase('SETTLE', self.args.settle)

    # ------------------------------------------------------------- planning
    def plan(self):
        if self.wp_index >= len(self.waypoints):
            self.finish('route complete')
            return
        if self.searching:
            if self.stop_known:
                print(f'  SEARCH done: mapped tag(s) {sorted(self.stop_tags)} in view', flush=True)
                self.searching = False
            elif self.search_steps >= self.args.max_search_steps:
                self.finish(f'search: no mapped tag after {self.search_steps} turns')
                return
            else:
                self.search_turn()
                return
        gx, gy, lap, side_k = self.waypoints[self.wp_index]
        x, y, yaw = self.slam.robot_pose
        dist = math.hypot(gx - x, gy - y)
        error = wrap_angle(math.atan2(gy - y, gx - x) - yaw)
        turn_tol = math.radians(self.args.turn_tol_deg)
        # Very close to the vertex the bearing is dominated by cm-level noise:
        # never spin in place for it (a -34 deg turn 6 cm before a vertex lost
        # all tags in the 2026-11-05 field test); count the vertex as reached.
        close = dist < self.args.approach_radius and abs(error) > turn_tol
        if dist < self.args.pos_tol or close:
            self.reach_vertex(lap, side_k, dist)
            return
        if abs(error) > turn_tol and self.turn_tries >= self.args.max_turn_tries and \
                abs(error) > math.radians(self.args.max_drive_error_deg):
            self.finish(f'could not align heading ({math.degrees(error):+.0f} deg left '
                        f'after {self.turn_tries} turns)')
            return
        if abs(error) > turn_tol and self.turn_tries < self.args.max_turn_tries:
            # Turn at most max_turn_step_deg at a time and observe in between, so
            # already-mapped tags and newly seen tags share a view (landmark
            # handoff): new tags are then initialized from a corrected heading
            # instead of inheriting the coast error of a full 90-degree spin.
            # Equal steps (e.g. 90 deg -> 3 x 30 deg), never a tiny leftover step.
            steps = math.ceil(abs(error) / math.radians(self.max_turn_step_deg) - 1e-9)
            self.start_turn(error / max(1, steps))
            return
        # About to DRIVE (turning in place never changes the position, driving
        # with a wrong heading does). If the last stop saw no mapped tag and the
        # heading is uncertain, look for a mapped tag first.
        yaw_std = math.sqrt(self.slam.robot_cov[2, 2])
        if not self.stop_known and yaw_std > math.radians(self.args.search_sigma_deg):
            self.search_episodes += 1
            if self.search_episodes > self.args.max_search_episodes:
                self.finish(f'search needed {self.search_episodes} times for one vertex')
                return
            self.searching = True
            self.search_steps = 0
            # Look back first: the mapped tags were on the side we turned away from.
            self.search_dir = -self.last_turn_sign
            print(f'  no mapped tag in view and yaw std {math.degrees(yaw_std):.1f} deg '
                  f'-> SEARCH before driving', flush=True)
            self.search_turn()
            return
        self.turn_tries = 0
        self.start_drive(min(self.args.segment, dist), error)

    def search_turn(self):
        self.search_steps += 1
        self.searches += 1
        self.start_turn(self.search_dir * math.radians(self.args.search_step_deg),
                        label=f'SEARCH {self.search_steps}/{self.args.max_search_steps}:',
                        count_try=False)

    def reach_vertex(self, lap, side_k, dist):
        x, y, yaw = self.slam.robot_pose
        cov = self.slam.robot_cov
        self.vertex_rows.append([lap, side_k, f'{self.elapsed():.2f}', f'{x:.4f}', f'{y:.4f}',
                                 f'{math.degrees(yaw):.2f}', f'{math.sqrt(cov[0, 0]):.4f}',
                                 f'{math.sqrt(cov[1, 1]):.4f}', f'{dist:.4f}'])
        print(f'VERTEX lap {lap} #{side_k}: est ({x:+.3f}, {y:+.3f}) m, '
              f'{math.degrees(yaw):+.1f} deg, {len(self.slam.index)} landmarks', flush=True)
        sides = 4 if self.args.path == 'square' else 8
        if side_k == sides:
            self.save_snapshot(f'end of lap {lap}', lap)
        self.wp_index += 1
        self.turn_tries = 0
        self.search_episodes = 0
        if self.args.vertex_hold > 0:
            self.set_phase('HOLD', self.args.vertex_hold)

    # ------------------------------------------------------------ main loop
    def tick(self):
        now = self.now()
        if self.done:
            self.send(0, 0)
            return
        if self.drive and self.phase not in ('INIT',) and self.motor_pub.get_subscription_count() == 0:
            self.finish('motor node disconnected')
            return
        if self.elapsed() > self.args.max_run_s:
            self.finish('max run time')
            return

        if self.phase == 'INIT':
            self.process_frames()
            if not self.drive:
                # Preview: one batched update every 2 s so the printed map stays current.
                if now - self.last_preview_print > 2.0:
                    self.last_preview_print = now
                    self.apply_stop_updates()
                    self.print_map('PREVIEW')
            elif now >= self.phase_until:
                self.apply_stop_updates()
                motor_ready = self.motor_pub.get_subscription_count() > 0
                if self.slam.index and motor_ready:
                    self.print_map('START')
                    self.save_snapshot('start', 0)
                    self.set_phase('PLAN')
                else:
                    print('WAITING: ' + ('no landmark confirmed yet' if not self.slam.index
                                         else 'motor_control not connected'), flush=True)
                    self.set_phase('INIT', self.args.init_observe)
        elif self.phase in ('TURN', 'DRIVE'):
            self.integrate_motion(now)
            if now >= self.phase_until:
                self.stop_motion(now)
        elif self.phase == 'SETTLE':
            self.frame_queue.clear()
            if now >= self.phase_until:
                self.stop_buffer = {}
                self.stop_frames = 0
                self.stop_tags = set()
                self.stop_known = 0
                self.set_phase('OBSERVE', self.args.observe)
        elif self.phase == 'OBSERVE':
            self.process_frames()
            if now >= self.phase_until:
                self.apply_stop_updates()
                self.end_observe()
        elif self.phase == 'HOLD':
            self.process_frames()
            if now >= self.phase_until:
                self.apply_stop_updates()
                self.set_phase('PLAN')
        if self.phase == 'PLAN':
            self.plan()

        left, right = self.command_target if self.phase in ('TURN', 'DRIVE') else (0.0, 0.0)
        self.send(left, right)
        self.log_state()

    def end_observe(self):
        x, y, yaw = self.slam.robot_pose
        std = math.sqrt(self.slam.robot_cov[0, 0] + self.slam.robot_cov[1, 1])
        tags = ','.join(str(t) for t in sorted(self.stop_tags)) or '-'
        yaw_std = math.sqrt(self.slam.robot_cov[2, 2])
        print(f'STOP wp {self.wp_index + 1}/{len(self.waypoints)}: '
              f'({x:+.3f}, {y:+.3f}) m {math.degrees(yaw):+.1f} deg, '
              f'pos std {100 * std:.1f} cm, yaw std {math.degrees(yaw_std):.1f} deg, '
              f'{self.stop_frames} frames, tags {tags}, {len(self.slam.index)} mapped', flush=True)
        # Limit blind DRIVING only: turning in place does not move the robot.
        if self.stop_tags:
            self.lost_stops = 0
        elif self.last_motion == 'DRIVE':
            self.lost_stops += 1
        if self.lost_stops >= self.args.lost_stops:
            self.finish(f'no landmark seen in {self.lost_stops} consecutive stops')
        elif std > MAX_ROBOT_STD_M:
            self.finish('robot position too uncertain')
        else:
            self.set_phase('PLAN')

    # ------------------------------------------------------------- output
    def log_state(self, force=False):
        now = self.now()
        if not force and now - self.last_log < LOG_PERIOD_S:
            return
        self.last_log = now
        x, y, yaw = self.slam.robot_pose
        P = self.slam.robot_cov
        lap, side_k = (self.waypoints[self.wp_index][2:] if self.wp_index < len(self.waypoints)
                       else ('', ''))
        self.traj.writerow([f'{self.elapsed():.3f}', self.phase, lap, side_k, f'{x:.4f}',
                            f'{y:.4f}', f'{math.degrees(yaw):.3f}', f'{P[0, 0]:.3e}',
                            f'{P[0, 1]:.3e}', f'{P[1, 1]:.3e}',
                            f'{math.degrees(math.sqrt(P[2, 2])):.3f}', len(self.slam.index),
                            f'{self.command[0]:.3f}', f'{self.command[1]:.3f}'])

    def print_map(self, title):
        x, y, yaw = self.slam.robot_pose
        print(f'{title}: robot ({x:+.3f}, {y:+.3f}) m {math.degrees(yaw):+.1f} deg, '
              f'{len(self.slam.index)} landmark(s)', flush=True)
        for tag_id, (m, C) in sorted(self.slam.landmarks().items()):
            st = self.slam.stats[tag_id]
            print(f'   tag {tag_id}: ({m[0]:+.3f}, {m[1]:+.3f}) m, std '
                  f'({100 * math.sqrt(C[0, 0]):.1f}, {100 * math.sqrt(C[1, 1]):.1f}) cm, '
                  f'{st["updates"]} updates, {st["rejected"]} rejected', flush=True)

    def save_snapshot(self, reason, lap):
        snap = self.slam.snapshot()
        snap.update({'reason': reason, 'lap': lap, 't_s': self.elapsed()})
        self.snapshots.append(snap)
        (self.run_dir / 'snapshots.json').write_text(json.dumps(self.snapshots))

    def finish(self, reason):
        if self.done:
            return
        self.finish_reason = reason
        self.done = True
        self.phase = 'DONE'
        self.send(0, 0)
        print(f'FINISHED: {reason} ({self.recoveries} covariance recoveries, '
              f'{self.searches} search turns)', flush=True)
        self.print_map('FINAL MAP')

    def close(self):
        try:
            for _ in range(3):
                self.done = True
                self.send(0, 0)
                time.sleep(0.05)
        except Exception:
            pass
        self.save_snapshot(f'final ({self.finish_reason or "interrupted"})', -1)
        with open(self.run_dir / 'vertices.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['lap', 'side', 't_s', 'x_m', 'y_m', 'yaw_deg', 'std_x_m', 'std_y_m',
                        'dist_to_target_m'])
            w.writerows(self.vertex_rows)
        self.traj_file.close()
        self.obs_file.close()
        self.upd_file.close()
        print(f'Saved run to {self.run_dir}', flush=True)
        if sys.stdout is self.tee:
            sys.stdout = self.prev_stdout
        self.console.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--drive', action='store_true', help='enable motors (default: preview only)')
    p.add_argument('--path', choices=['square', 'octagon'], default='square')
    p.add_argument('--side', type=float, default=None,
                   help='side length in m (default 1.2 square, 0.5 octagon)')
    p.add_argument('--laps', type=int, default=1)
    p.add_argument('--direction', choices=['ccw', 'cw'], default='ccw')
    p.add_argument('--label', default=None, help='name of the run folder')
    p.add_argument('--speed-per-cmd', type=float, default=None,
                   help='override HW1 speed_per_cmd (m/s per command) for another floor')
    p.add_argument('--spin-offset-deg', type=float, default=0.0,
                   help='CCW coast angle after a spin (deg). Default 0: on the field floor '
                        '(2026-11-05) short <=35 deg spins turned ~100 deg/s x duration with no '
                        "coast; HW1's 20.5 deg was tuned for long single turns")
    p.add_argument('--trim', type=float, default=0.04,
                   help='left/right drive trim. Default 0.04 from the 2026-11-05 field fit: '
                        "HW1's 0.10 over-corrected to a -4.4 deg/m right drift")
    p.add_argument('--drive-wheel-base', type=float, default=1.55,
                   help='effective wheel base (m) for steering while driving forward; field fit '
                        '2026-11-05 (in-place spins use the HW1 value 0.526)')
    p.add_argument('--cw-offset-deg', type=float, default=None,
                   help='override HW1 CW coast angle after a spin (deg) for another floor')
    p.add_argument('--segment', type=float, default=0.30, help='max straight segment between stops (m)')
    p.add_argument('--settle', type=float, default=0.5, help='wait after stopping before observing (s)')
    p.add_argument('--observe', type=float, default=1.0, help='observation time per stop (s)')
    p.add_argument('--stamp-margin', type=float, default=0.3,
                   help='use only images captured at least this long after the wheels stopped (s)')
    p.add_argument('--init-observe', type=float, default=3.0, help='observation time at the start (s)')
    p.add_argument('--vertex-hold', type=float, default=0.0,
                   help='extra stop at each vertex, e.g. to mark the real position (s)')
    p.add_argument('--pos-tol', type=float, default=0.05, help='vertex reached within (m)')
    p.add_argument('--turn-tol-deg', type=float, default=8.0,
                   help='spin in place above this heading error; steer while driving below it')
    p.add_argument('--max-turn-step-deg', type=float, default=None,
                   help='largest in-place turn between two observation stops '
                        '(default 35 for square, 50 for octagon)')
    p.add_argument('--max-turn-tries', type=int, default=10,
                   help='in-place turns per segment before driving anyway')
    p.add_argument('--max-drive-error-deg', type=float, default=20.0,
                   help='never drive with a larger heading error: stop the run instead')
    p.add_argument('--approach-radius', type=float, default=0.10,
                   help='within this distance (m) of a vertex, do not spin to re-aim: '
                        'count the vertex as reached')
    p.add_argument('--search-sigma-deg', type=float, default=5.0,
                   help='before driving: if the last stop saw no mapped tag and the heading '
                        'std exceeds this, search by turning in place instead of driving')
    p.add_argument('--search-step-deg', type=float, default=30.0)
    p.add_argument('--max-search-steps', type=int, default=12, help='12 x 30 deg = one full turn')
    p.add_argument('--max-search-episodes', type=int, default=6,
                   help='searches allowed per vertex before stopping the run')
    p.add_argument('--new-landmark-max-yaw-std-deg', type=float, default=10.0,
                   help='defer adding a new landmark while the heading std exceeds this')
    p.add_argument('--max-range', type=float, default=2.6, help='ignore tags farther than this (m)')
    p.add_argument('--max-view-deg', type=float, default=60.0, help='ignore tags seen more obliquely')
    p.add_argument('--min-new-frames', type=int, default=3,
                   help='frames of a new tag ID needed in one stop before it becomes a landmark')
    p.add_argument('--r-scale', type=float, default=2.0, help='multiply calibrated R std devs')
    p.add_argument('--q-scale', type=float, default=1.0, help='multiply Q variances')
    p.add_argument('--turn-coast-std-deg', type=float, default=8.0,
                   help='extra heading std per in-place turn (coast uncertainty)')
    p.add_argument('--recover-std-m', type=float, default=0.15,
                   help='robot position std added when every known tag at a stop is rejected')
    p.add_argument('--recover-std-deg', type=float, default=15.0,
                   help='robot heading std added in that recovery')
    p.add_argument('--lost-stops', type=int, default=3,
                   help='stop the run after this many stops without any landmark')
    p.add_argument('--max-run-s', type=float, default=900.0)
    return p.parse_args(argv)


def main():
    args = parse_args()
    rclpy.init()
    node = SlamNode(args)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        print('Interrupted -- stopping motors.', flush=True)
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
