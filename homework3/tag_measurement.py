"""AprilTag TF detection -> (tag_id, range, bearing) in the ROBOT frame.

Shared by calibrate_sensor.py and the HW3 SLAM node. Reuses the HW2 camera
extrinsics and quaternion helper from homework2/check_localization.py
(which imports rclpy, so this module runs on the rover, not on a laptop
without ROS).

Robot frame: origin = ground point under the robot reference (9 cm behind the
camera lens, as in HW2), +x forward, +y left. Bearing > 0 means "to the left".
"""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'homework2'))
from check_localization import (  # noqa: E402  (reuse HW2 extrinsics + math)
    P_ROBOT_CAMERA, R_ROBOT_CAMERA, quaternion_matrix,
)

CAMERA_FRAME = 'camera_frame'


def parse_tag_id(child_frame_id):
    """'tag_7' (frame name from a tag list) or 'tag36h11:7' (apriltag_ros default) -> 7."""
    name = child_frame_id.lstrip('/')
    if name.startswith('tag_'):
        text = name[4:]
    elif ':' in name:
        text = name.rsplit(':', 1)[1]
    else:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def measure(transform):
    """One geometry_msgs/TransformStamped (camera_frame -> tag) -> dict, or None.

    range/bearing use only the horizontal (x, y) part of the tag position in
    the robot frame: the 2D SLAM state does not model height.
    view_deg is the angle between the tag's normal and the line of sight
    (0 = seen head-on), useful to study oblique detections.
    """
    if transform.header.frame_id.lstrip('/') != CAMERA_FRAME:
        return None
    tag_id = parse_tag_id(transform.child_frame_id)
    if tag_id is None:
        return None
    t = transform.transform.translation
    p_camera = np.array([t.x, t.y, t.z], dtype=float)
    q = transform.transform.rotation
    r_camera_tag = quaternion_matrix((q.x, q.y, q.z, q.w))
    if not np.isfinite(p_camera).all() or np.linalg.norm(p_camera) < 1e-6:
        return None

    p_robot = R_ROBOT_CAMERA @ p_camera + P_ROBOT_CAMERA
    rng = math.hypot(p_robot[0], p_robot[1])
    bearing = math.atan2(p_robot[1], p_robot[0])
    normal = r_camera_tag[:, 2]
    cos_view = abs(float(normal @ p_camera)) / float(np.linalg.norm(p_camera))
    return {
        'tag_id': tag_id,
        'stamp': transform.header.stamp.sec + 1e-9 * transform.header.stamp.nanosec,
        'range': rng,
        'bearing': bearing,
        'view_deg': math.degrees(math.acos(min(1.0, cos_view))),
        'p_robot': p_robot,
        'p_camera': p_camera,
    }
