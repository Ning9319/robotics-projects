"""Print robot world pose from one AprilTag, without moving the rover."""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

R_ROBOT_CAMERA = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
P_ROBOT_CAMERA = np.array([0.09, 0.0, 0.125])
NORMALS = {'+x': [1, 0, 0], '-x': [-1, 0, 0],
           '+y': [0, 1, 0], '-y': [0, -1, 0]}
UP = np.array([0.0, 0.0, 1.0])


def quaternion_matrix(q):
    x, y, z, w = q
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ])


class LocalizationCheck(Node):
    def __init__(self, tag_id):
        super().__init__('localization_check')
        path = Path(__file__).resolve().parent / 'config' / 'landmarks.yaml'
        with open(path, encoding='utf-8') as f:
            tags = yaml.safe_load(f)['tags']
        tag = tags[tag_id]
        self.tag_id = tag_id
        self.p_world_tag = np.array(tag['position_m'], dtype=float)
        normal = np.array(NORMALS[tag['facing']], dtype=float)
        self.R_world_tag = np.column_stack((np.cross(UP, normal), UP, normal))
        self.last_print = 0.0
        self.create_subscription(TFMessage, '/tf', self.on_tf, 10)
        print(f'Listening for camera_frame -> tag_{tag_id} ... (Ctrl+C to quit)', flush=True)

    def on_tf(self, message):
        for tf in message.transforms:
            if tf.header.frame_id.lstrip('/') != 'camera_frame' or tf.child_frame_id != f'tag_{self.tag_id}':
                continue
            if time.monotonic() - self.last_print < 0.5:
                continue
            self.last_print = time.monotonic()
            t = tf.transform.translation
            p_camera_tag = np.array([t.x, t.y, t.z], dtype=float)
            q = tf.transform.rotation
            R_camera_tag = quaternion_matrix((q.x, q.y, q.z, q.w))

            # World <- Camera <- Tag, then remove Robot <- Camera extrinsic.
            R_world_camera = self.R_world_tag @ R_camera_tag.T
            R_world_robot = R_world_camera @ R_ROBOT_CAMERA.T
            p_world_camera = self.p_world_tag - R_world_camera @ p_camera_tag
            p_world_robot = p_world_camera - R_world_robot @ P_ROBOT_CAMERA
            yaw = math.degrees(math.atan2(R_world_robot[1, 0], R_world_robot[0, 0]))
            print(f'tag {self.tag_id}: x={p_world_robot[0]:+.3f} m, '
                  f'y={p_world_robot[1]:+.3f} m, heading={yaw:+.1f} deg', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=int, default=1)
    args = parser.parse_args()
    rclpy.init()
    node = LocalizationCheck(args.tag)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
