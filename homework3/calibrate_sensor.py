#!/usr/bin/env python3
"""HW3 Step 2: record AprilTag range/bearing at tape-measured positions.

The robot stays still; motors are not used. Start the camera, rectify and
apriltag nodes as in HW2, then on the rover:

    python3 homework3/calibrate_sensor.py --frames 30

For each position, type the tape-measured FORWARD distance and LEFT offset
(right = negative) from the robot reference point (ground point 9 cm behind
the camera lens, as in HW2) to the tag CENTER. The script records N new camera
frames of that tag, prints mean / std / bias right away, and appends every raw
frame to homework3/output/sensor_calibration_<time>.csv.
Analyze the CSV with analyze_sensor_calibration.py (runs without ROS).
"""

import argparse
import csv
import math
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

from tag_measurement import measure

ROOT = Path(__file__).resolve().parents[1]
VISIBLE_WINDOW_S = 1.5   # a tag counts as "visible" if seen within this time
FIELDS = ['label', 'tag_id', 'true_forward_m', 'true_left_m', 'true_range_m',
          'true_bearing_deg', 'stamp_s', 'range_m', 'bearing_deg', 'view_deg',
          'x_robot_m', 'y_robot_m', 'z_robot_m']


class Recorder(Node):
    def __init__(self):
        super().__init__('hw3_sensor_calibration')
        self.lock = threading.Lock()
        self.samples = {}      # tag_id -> list of (receive_time, measurement)
        self.create_subscription(TFMessage, '/tf', self.on_tf, 50)

    def on_tf(self, message):
        now = time.monotonic()
        for transform in message.transforms:
            obs = measure(transform)
            if obs is None:
                continue
            with self.lock:
                history = self.samples.setdefault(obs['tag_id'], [])
                # One sample per camera image: skip repeated TF for the same stamp.
                if history and history[-1][1]['stamp'] == obs['stamp']:
                    continue
                history.append((now, obs))
                if len(history) > 2000:
                    del history[:1000]

    def visible(self):
        now = time.monotonic()
        with self.lock:
            return sorted(tag_id for tag_id, h in self.samples.items()
                          if h and now - h[-1][0] < VISIBLE_WINDOW_S)

    def new_samples(self, tag_id, since):
        with self.lock:
            return [obs for t, obs in self.samples.get(tag_id, []) if t > since]


def ask_float(prompt, default=None):
    while True:
        text = input(prompt).strip()
        if not text and default is not None:
            return default
        try:
            return float(text)
        except ValueError:
            print('  please type a number')


def choose_tag(recorder, fixed_tag):
    """Return the tag to record, waiting until it (or exactly one tag) is visible."""
    while True:
        visible = recorder.visible()
        if fixed_tag is not None:
            if fixed_tag in visible:
                return fixed_tag
            print(f'  tag {fixed_tag} not visible (visible: {visible or "none"}); '
                  'fix the setup, then press Enter')
        elif len(visible) == 1:
            return visible[0]
        elif not visible:
            print('  no tag visible; fix the setup, then press Enter')
        else:
            text = input(f'  several tags visible {visible}; which ID? ').strip()
            if text.isdigit() and int(text) in visible:
                return int(text)
            continue
        input()


def collect(recorder, tag_id, frames):
    start = time.monotonic()
    timeout = max(30.0, 3.0 * frames)
    last_count = -1
    while True:
        got = recorder.new_samples(tag_id, start)
        if len(got) >= frames:
            return got[:frames]
        if len(got) != last_count:
            print(f'\r  recording tag {tag_id}: {len(got)}/{frames} frames', end='', flush=True)
            last_count = len(got)
        if time.monotonic() - start > timeout:
            print(f'\n  timeout after {timeout:.0f} s: only {len(got)} frames')
            return got
        time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--frames', type=int, default=30, help='camera frames per position')
    parser.add_argument('--tag', type=int, default=None, help='only record this tag ID')
    args = parser.parse_args()

    folder = ROOT / 'homework3/output'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'sensor_calibration_{datetime.now():%Y%m%d_%H%M%S}.csv'

    rclpy.init()
    recorder = Recorder()
    threading.Thread(target=rclpy.spin, args=(recorder,), daemon=True).start()
    print(f'Raw data -> {path}')
    print('Robot frame: +x forward, +y LEFT. Distances to the tag CENTER, in meters.')

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        try:
            while True:
                label = input('\nlabel for this position (e.g. d2.0_left0.3, Enter = quit): ').strip()
                if not label:
                    break
                forward = ask_float('  forward distance to tag center (m): ')
                left = ask_float('  left offset of tag center (m, right = negative) [0]: ', 0.0)
                true_range = math.hypot(forward, left)
                true_bearing = math.degrees(math.atan2(left, forward))

                tag_id = choose_tag(recorder, args.tag)
                got = collect(recorder, tag_id, args.frames)
                if not got:
                    print('  nothing recorded for this position')
                    continue
                for obs in got:
                    p = obs['p_robot']
                    writer.writerow([label, tag_id, f'{forward:.4f}', f'{left:.4f}',
                                     f'{true_range:.4f}', f'{true_bearing:.3f}',
                                     f'{obs["stamp"]:.3f}', f'{obs["range"]:.5f}',
                                     f'{math.degrees(obs["bearing"]):.4f}',
                                     f'{obs["view_deg"]:.2f}',
                                     f'{p[0]:.5f}', f'{p[1]:.5f}', f'{p[2]:.5f}'])
                f.flush()

                r = np.array([o['range'] for o in got])
                b = np.degrees([o['bearing'] for o in got])
                view = np.mean([o['view_deg'] for o in got])
                print(f'\n  {len(got)} frames of tag {tag_id}, view angle {view:.0f} deg')
                print(f'  range   true {true_range:.3f} m | mean {r.mean():.4f} m, '
                      f'std {100 * r.std(ddof=1) if len(r) > 1 else 0:.2f} cm, '
                      f'bias {100 * (r.mean() - true_range):+.2f} cm')
                print(f'  bearing true {true_bearing:+.2f} deg | mean {b.mean():+.2f} deg, '
                      f'std {b.std(ddof=1) if len(b) > 1 else 0:.2f} deg, '
                      f'bias {b.mean() - true_bearing:+.2f} deg')
        except (KeyboardInterrupt, EOFError):
            print('\nstopped')
    recorder.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    print(f'Saved {path}')


if __name__ == '__main__':
    main()
