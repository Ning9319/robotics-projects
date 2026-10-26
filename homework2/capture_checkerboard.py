
from pathlib import Path
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

OUT = Path("homework2/calibration/images")
PATTERN = (7, 9)  # Internal corners


class CameraCapture(Node):
    def __init__(self):
        super().__init__("hw2_camera_capture")
        self.latest = None
        self.sub = self.create_subscription(
            Image, "/camera/image_raw", self.on_image, 10
        )

    def on_image(self, msg):
        self.latest = msg


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = CameraCapture()
    count = len(list(OUT.glob("*.png")))

    print("Position checkerboard, then press Enter.")
    print("Type q to quit.")

    try:
        while True:
            if input("Capture [Enter/q]: ").strip().lower() == "q":
                break

            node.latest = None
            deadline = time.monotonic() + 5

            while node.latest is None and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.5)

            if node.latest is None:
                print("No camera frame received.")
                continue

            msg = node.latest
            rgb = np.ndarray(
                (msg.height, msg.width, 3),
                dtype=np.uint8,
                buffer=msg.data,
                strides=(msg.step, 3, 1),
            ).copy()

            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            found, _ = cv2.findChessboardCorners(gray, PATTERN)

            if not found:
                print("Checkerboard corners NOT detected. Reposition and retry.")
                continue

            count += 1
            path = OUT / f"checkerboard_{count:03d}.png"
            cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            print(f"Saved: {path} (corners detected)")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
