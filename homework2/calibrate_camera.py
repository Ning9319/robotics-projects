
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
IMAGES = sorted((ROOT / "calibration/images").glob("checkerboard_*.png"))
OUTPUT = ROOT / "calibration/camera_intrinsics.yaml"

PATTERN = (7, 9)       # Number of internal corners
SQUARE_SIZE = 0.020    # Meters; verify actual printed size

if len(IMAGES) < 10:
    raise RuntimeError("Not enough images for calibration.")

# Real-world checkerboard corner coordinates
obj = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
obj[:, :2] = (
    np.mgrid[0:PATTERN[0], 0:PATTERN[1]]
    .T.reshape(-1, 2) * SQUARE_SIZE
)

object_points = []
image_points = []
used_images = []
image_size = None

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)

for path in IMAGES:
    img = cv2.imread(str(path))
    if img is None:
        print(f"Skipped unreadable file: {path.name}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    current_size = (gray.shape[1], gray.shape[0])

    if image_size is None:
        image_size = current_size
    elif current_size != image_size:
        print(f"Skipped different resolution: {path.name}")
        continue

    found, corners = cv2.findChessboardCorners(gray, PATTERN)

    if not found:
        print(f"Skipped undetected corners: {path.name}")
        continue

    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1), criteria
    )

    object_points.append(obj.copy())
    image_points.append(corners)
    used_images.append(path.name)

if len(used_images) < 10:
    raise RuntimeError("Too few usable calibration images.")

rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
    object_points, image_points, image_size, None, None
)

print(f"\nImages used: {len(used_images)}")
print(f"Image resolution: {image_size}")
print(f"RMS reprojection error: {rms:.4f} pixels")
print("\nCamera matrix K:\n", K)
print("\nDistortion coefficients D:\n", D)

print("\nPer-image reprojection errors:")
for i, name in enumerate(used_images):
    projected, _ = cv2.projectPoints(
        object_points[i], rvecs[i], tvecs[i], K, D
    )
    diff = image_points[i].reshape(-1, 2) - projected.reshape(-1, 2)
    error = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    print(f"  {name}: {error:.3f} px")

# Match the format used by the existing robot_vision_camera node
P = np.hstack((K, np.zeros((3, 1))))

fs = cv2.FileStorage(str(OUTPUT), cv2.FILE_STORAGE_WRITE)
if not fs.isOpened():
    raise RuntimeError(f"Could not write {OUTPUT}")

fs.write("image_width", image_size[0])
fs.write("image_height", image_size[1])
fs.write("camera_name", "camera")
fs.write("camera_matrix", K)
fs.write("distortion_model", "plumb_bob")
fs.write("distortion_coefficients", D.reshape(1, -1))
fs.write("rectification_matrix", np.eye(3))
fs.write("projection_matrix", P)
fs.release()

print(f"\nCalibration saved to: {OUTPUT}")
