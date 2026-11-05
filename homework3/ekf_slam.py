#!/usr/bin/env python3
"""HW3 EKF-SLAM core (pure numpy, no ROS) for the Waveshare rover.

State   x = [x_r, y_r, theta_r, m1_x, m1_y, m2_x, m2_y, ...]^T   (m, m, rad, m, m, ...)
Measure z = [range, bearing] of ONE landmark in the robot frame  (m, rad)

* The start pose defines the SLAM frame: x_0|0 = [0, 0, 0], Sigma_0|0 ~ 0.
* The filter does NOT know how many landmarks exist. A landmark is appended to
  x and Sigma the first time its ID is observed (dynamic state augmentation).
* The rover has no wheel encoders and an imprecise IMU, so the motion input is
  the HW1 command model (left/right command -> v, omega). Its noise Q is large
  and scales with how far the robot moved, so a stopped robot gains no
  uncertainty.

The ROS node is responsible for turning AprilTag detections into
(tag_id, range, bearing) using the HW2 camera extrinsics, and for turning motor
commands into (distance, rotation) increments using the HW1 Model.
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'robot_control'))
from robot_control.waypoint_controller import wrap_angle  # noqa: E402  (reuse HW1)

ROBOT_DIM = 3   # x_r, y_r, theta_r
LM_DIM = 2      # m_x, m_y  (landmark orientation is not estimated)

CHI2_2DOF_95 = 5.991
CHI2_2DOF_99 = 9.210


@dataclass
class NoiseParams:
    """Process (Q) and measurement (R) noise.

    !!! PLACEHOLDER VALUES -- replace with the Step 2 calibration results. !!!

    Motion noise is a variance PER UNIT OF MOTION, so it adds up correctly no
    matter how often predict() is called. Calibrate it as:
        var_per_m   = (std of the measured error over a segment)^2 / segment_length
        var_per_rad = (std of the measured turn error)^2 / commanded_turn_angle
    """
    # Process noise Q (robot-frame: along-track, sideways, heading)
    var_forward_per_m: float = 0.10 ** 2            # m^2 per m driven
    var_lateral_per_m: float = 0.05 ** 2            # m^2 per m driven
    var_heading_per_rad: float = 0.25 ** 2          # rad^2 per rad turned
    var_heading_per_m: float = math.radians(8) ** 2  # rad^2 per m driven (drift)
    # Measurement noise R:  std_range = base + per_m * range
    range_std_base: float = 0.03                    # m
    range_std_per_m: float = 0.03                   # m per m of range
    bearing_std: float = math.radians(2.0)          # rad


def range_bearing_from_xy(px, py):
    """Landmark position in the ROBOT frame (x forward, y left) -> (range, bearing)."""
    return math.hypot(px, py), math.atan2(py, px)


def motion_model(pose, d, a):
    """f(x, u): move `d` meters along the mid-turn heading and rotate `a` radians.

    Using the mid-turn heading (theta + a/2) is exact for a pure drive (a = 0)
    and a pure in-place spin (d = 0), which are the two motions we command.
    Returns (new_pose, F_r, G, V):
        F_r = d f / d (x_r, y_r, theta)     3x3  system Jacobian (robot block)
        G   = d f / d (d, a)                3x2  control Jacobian
        V   = robot-frame -> SLAM-frame rotation used to map Q into the state
    """
    x, y, theta = pose
    mid = theta + 0.5 * a
    c, s = math.cos(mid), math.sin(mid)
    new_pose = np.array([x + d * c, y + d * s, wrap_angle(theta + a)])
    F_r = np.array([[1.0, 0.0, -d * s],
                    [0.0, 1.0, d * c],
                    [0.0, 0.0, 1.0]])
    G = np.array([[c, -0.5 * d * s],
                  [s, 0.5 * d * c],
                  [0.0, 1.0]])
    V = np.array([[c, -s, 0.0],
                  [s, c, 0.0],
                  [0.0, 0.0, 1.0]])
    return new_pose, F_r, G, V


def measurement_model(pose, landmark):
    """h(x): expected (range, bearing) of `landmark` seen from `pose`.

    Returns (z_hat, H_r, H_m): Jacobians w.r.t. the robot pose (2x3) and the
    landmark position (2x2). All other columns of H are zero.
    """
    dx = landmark[0] - pose[0]
    dy = landmark[1] - pose[1]
    q = dx * dx + dy * dy
    r = math.sqrt(q)
    z_hat = np.array([r, wrap_angle(math.atan2(dy, dx) - pose[2])])
    H_r = np.array([[-dx / r, -dy / r, 0.0],
                    [dy / q, -dx / q, -1.0]])
    H_m = np.array([[dx / r, dy / r],
                    [-dy / q, dx / q]])
    return z_hat, H_r, H_m


def inverse_measurement(pose, z):
    """g(x, z): landmark position from a pose and a (range, bearing) measurement.

    Returns (landmark, G_x, G_z) with G_x = dg/d(robot pose), G_z = dg/dz.
    """
    r, b = z
    angle = pose[2] + b
    c, s = math.cos(angle), math.sin(angle)
    landmark = np.array([pose[0] + r * c, pose[1] + r * s])
    G_x = np.array([[1.0, 0.0, -r * s],
                    [0.0, 1.0, r * c]])
    G_z = np.array([[c, -r * s],
                    [s, r * c]])
    return landmark, G_x, G_z


class EKFSLAM:
    """Landmark EKF-SLAM with an unknown, growing number of point landmarks."""

    def __init__(self, noise=None, gate_chi2=CHI2_2DOF_99, gate_min_updates=2,
                 init_pose_var=1e-9):
        self.noise = noise or NoiseParams()
        # Mahalanobis gate: reject an observation of a known landmark if
        # y^T S^-1 y > gate_chi2. Not applied until the landmark has been
        # updated `gate_min_updates` times, so one bad first sighting cannot
        # lock out every later (correct) observation of that landmark.
        self.gate_chi2 = gate_chi2
        self.gate_min_updates = gate_min_updates

        # x_0|0 = [0, 0, 0]: the start pose IS the SLAM origin, so its
        # uncertainty is (almost) zero. No landmarks are known yet.
        self.x = np.zeros(ROBOT_DIM)
        self.P = np.eye(ROBOT_DIM) * init_pose_var
        self.index = {}   # tag_id -> start index of (m_x, m_y) in x
        self.stats = {}   # tag_id -> {'seen', 'updates', 'rejected'}
        self.last_nis = float('nan')  # NIS of the latest known-landmark observation

    # ------------------------------------------------------------ prediction
    def predict(self, d, a):
        """Prediction step for a motion of `d` meters and `a` radians.

        x_t|t-1 = f(x_t-1|t-1, u),   Sigma_t|t-1 = F Sigma F^T + Q
        Landmarks are static, so F is the identity outside the robot block and
        only the robot rows/columns of Sigma need to be recomputed.
        """
        if d == 0.0 and a == 0.0:
            return  # stopped: F = I, Q = 0
        self.x[:ROBOT_DIM], F_r, _, V = motion_model(self.x[:ROBOT_DIM], d, a)
        n = self.noise
        M = np.diag([n.var_forward_per_m * abs(d),
                     n.var_lateral_per_m * abs(d),
                     n.var_heading_per_rad * abs(a) + n.var_heading_per_m * abs(d)])
        Q_r = V @ M @ V.T

        r = slice(0, ROBOT_DIM)
        m = slice(ROBOT_DIM, None)
        self.P[r, r] = F_r @ self.P[r, r] @ F_r.T + Q_r
        self.P[r, m] = F_r @ self.P[r, m]
        self.P[m, r] = self.P[r, m].T

    def add_robot_noise(self, Q_r):
        """Add extra process noise to the robot block only (e.g. per-turn coast error)."""
        self.P[:ROBOT_DIM, :ROBOT_DIM] += np.asarray(Q_r, dtype=float)

    def predict_velocity(self, v, omega, dt):
        """Convenience wrapper: HW1 Model.twist() gives (v, omega) per command."""
        self.predict(v * dt, omega * dt)

    # ---------------------------------------------------------------- update
    def measurement_noise(self, measured_range):
        n = self.noise
        range_std = n.range_std_base + n.range_std_per_m * measured_range
        return np.diag([range_std ** 2, n.bearing_std ** 2])

    def update_frame(self, observations):
        """Process every (tag_id, range, bearing) seen in ONE camera frame.

        Known landmarks are corrected first, so new landmarks are initialized
        from the best available robot pose. Landmarks that are not in
        `observations` (out of view) are simply left unchanged.
        Returns {tag_id: 'updated' | 'rejected' | 'new' | 'skipped'}.
        """
        known = [o for o in observations if o[0] in self.index]
        new = [o for o in observations if o[0] not in self.index]
        results = {}
        for tag_id, rng, bearing in known + new:
            results[tag_id] = self.update(tag_id, rng, bearing)
        return results

    def update(self, tag_id, rng, bearing):
        if not (math.isfinite(rng) and math.isfinite(bearing)) or rng < 1e-3:
            return 'skipped'
        R = self.measurement_noise(rng)
        if tag_id not in self.index:
            self._add_landmark(tag_id, np.array([rng, bearing]), R)
            return 'new'

        st = self.stats[tag_id]
        st['seen'] += 1
        j = self.index[tag_id]
        z_hat, H_r, H_m = measurement_model(self.x[:ROBOT_DIM], self.x[j:j + LM_DIM])
        H = np.zeros((2, self.x.size))
        H[:, :ROBOT_DIM] = H_r
        H[:, j:j + LM_DIM] = H_m

        y = np.array([rng - z_hat[0], wrap_angle(bearing - z_hat[1])])  # innovation
        S = H @ self.P @ H.T + R
        # NIS = y^T S^-1 y; averages ~2 (2 DoF) when Q and R are consistent.
        self.last_nis = float(y @ np.linalg.solve(S, y))
        if st['updates'] >= self.gate_min_updates and self.last_nis > self.gate_chi2:
            st['rejected'] += 1
            return 'rejected'

        K = np.linalg.solve(S, H @ self.P).T          # K = P H^T S^-1  (S symmetric)
        self.x = self.x + K @ y
        self.x[2] = wrap_angle(self.x[2])
        I_KH = np.eye(self.x.size) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form
        self.P = 0.5 * (self.P + self.P.T)
        st['updates'] += 1
        return 'updated'

    def _add_landmark(self, tag_id, z, R):
        """Augment x and Sigma with a newly observed landmark.

        m = g(x_r, z)
        Sigma_mm = G_x Sigma_rr G_x^T + G_z R G_z^T
        Sigma_mx = G_x Sigma_r,:      (correlation with robot AND old landmarks)
        """
        landmark, G_x, G_z = inverse_measurement(self.x[:ROBOT_DIM], z)
        P_rx = self.P[:ROBOT_DIM, :]
        P_mx = G_x @ P_rx
        P_mm = G_x @ self.P[:ROBOT_DIM, :ROBOT_DIM] @ G_x.T + G_z @ R @ G_z.T

        j = self.x.size
        self.x = np.concatenate([self.x, landmark])
        self.P = np.block([[self.P, P_mx.T],
                           [P_mx, P_mm]])
        self.P = 0.5 * (self.P + self.P.T)
        self.index[tag_id] = j
        self.stats[tag_id] = {'seen': 1, 'updates': 0, 'rejected': 0}

    # ------------------------------------------------------------- accessors
    @property
    def robot_pose(self):
        return self.x[:ROBOT_DIM].copy()

    @property
    def robot_cov(self):
        return self.P[:ROBOT_DIM, :ROBOT_DIM].copy()

    def landmarks(self):
        """{tag_id: (mean (2,), covariance (2, 2))} in the SLAM frame."""
        return {tag_id: (self.x[j:j + LM_DIM].copy(), self.P[j:j + LM_DIM, j:j + LM_DIM].copy())
                for tag_id, j in self.index.items()}

    def snapshot(self):
        """Plain-python copy of the full filter state, for logging to disk."""
        return {
            'x': self.x.tolist(),
            'P': self.P.tolist(),
            'landmark_ids': list(self.index),
            'landmark_index': dict(self.index),
            'stats': {k: dict(v) for k, v in self.stats.items()},
        }
