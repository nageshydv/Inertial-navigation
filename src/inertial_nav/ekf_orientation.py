"""
Extended Kalman Filter – Orientation Estimation
================================================
EKF-based fusion of accelerometer, gyroscope, and optional magnetometer data
for robust 3D orientation estimation.

Reference: "Accurate IMU-Based Orientation Estimation Under Conditions of
           Magnetic Distortion", Sensors, 2014. (Nagesh Yadav et al.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .dead_reckoning import Quaternion

logger = logging.getLogger(__name__)


@dataclass
class EKFConfig:
    """Tuning parameters for the orientation EKF."""
    # Process noise – gyroscope angle random walk (rad²/s)
    q_gyro: float = 1e-4
    # Process noise – gyroscope bias random walk (rad²/s³)
    q_bias: float = 1e-5
    # Measurement noise – accelerometer (m²/s⁴)
    r_acc: float = 1e-1
    # Measurement noise – magnetometer (µT²)
    r_mag: float = 1e0
    # Initial orientation covariance
    p0_orientation: float = 1e-2
    # Initial bias covariance
    p0_bias: float = 1e-4


class OrientationEKF:
    """
    Extended Kalman Filter for quaternion-based orientation estimation.

    State vector: [q_w, q_x, q_y, q_z, b_x, b_y, b_z]
        q   – unit quaternion (orientation)
        b   – gyroscope bias (rad/s)

    Observations:
        - Accelerometer (gravity direction in body frame)
        - Magnetometer  (optional, magnetic north in body frame)

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    cfg : EKFConfig
        Noise and initial covariance parameters.
    gravity_magnitude : float
        Expected gravitational acceleration in m/s².
    """

    STATE_DIM = 7   # [qw, qx, qy, qz, bx, by, bz]
    OBS_DIM_ACC = 3
    OBS_DIM_MAG = 3

    def __init__(
        self,
        fs: float = 50.0,
        cfg: EKFConfig | None = None,
        gravity_magnitude: float = 9.81,
    ) -> None:
        self.dt = 1.0 / fs
        self.cfg = cfg or EKFConfig()
        self.g_mag = gravity_magnitude

        # State: [q, bias]
        self._x = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Covariance
        self._P = np.diag([
            self.cfg.p0_orientation] * 4 + [self.cfg.p0_bias] * 3
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, q0: np.ndarray | None = None) -> None:
        """Reset filter to initial state."""
        self._x = np.zeros(7)
        self._x[0] = 1.0
        if q0 is not None:
            self._x[:4] = Quaternion.normalise(q0)
        self._P = np.diag(
            [self.cfg.p0_orientation] * 4 + [self.cfg.p0_bias] * 3
        )

    def update(
        self,
        gyro: np.ndarray,
        acc: np.ndarray,
        mag: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Process one IMU sample and return the updated orientation quaternion.

        Parameters
        ----------
        gyro : np.ndarray, shape (3,)
            Gyroscope measurement in rad/s.
        acc : np.ndarray, shape (3,)
            Accelerometer measurement in m/s².
        mag : np.ndarray or None, shape (3,)
            Optional magnetometer measurement in µT.

        Returns
        -------
        np.ndarray, shape (4,)
            Updated unit quaternion [w, x, y, z].
        """
        self._predict(gyro)
        self._correct_acc(acc)
        if mag is not None:
            self._correct_mag(mag)
        # Re-normalise quaternion component
        self._x[:4] = Quaternion.normalise(self._x[:4])
        return self._x[:4].copy()

    def run_batch(
        self,
        gyro_seq: np.ndarray,
        acc_seq: np.ndarray,
        mag_seq: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Process a full batch of IMU data.

        Parameters
        ----------
        gyro_seq : np.ndarray, shape (N, 3)
        acc_seq  : np.ndarray, shape (N, 3)
        mag_seq  : np.ndarray or None, shape (N, 3)

        Returns
        -------
        np.ndarray, shape (N, 4)  – quaternion at each timestep
        """
        N = len(gyro_seq)
        quaternions = np.zeros((N, 4))
        for i in range(N):
            mag = mag_seq[i] if mag_seq is not None else None
            quaternions[i] = self.update(gyro_seq[i], acc_seq[i], mag)
        euler = np.array([Quaternion.to_euler(q) for q in quaternions])
        logger.info(
            "EKF batch complete. Final Euler (RPY°): %s",
            np.round(euler[-1], 2),
        )
        return quaternions

    # ------------------------------------------------------------------
    # EKF internals
    # ------------------------------------------------------------------

    def _predict(self, gyro: np.ndarray) -> None:
        """Propagate state and covariance using gyro measurements."""
        dt = self.dt
        q = self._x[:4]
        bias = self._x[4:]

        # Bias-corrected angular rate
        omega = gyro - bias

        # Quaternion kinematics: dq/dt = 0.5 * Omega(omega) * q
        wx, wy, wz = omega
        Omega = 0.5 * np.array([
            [0,  -wx, -wy, -wz],
            [wx,   0,  wz, -wy],
            [wy, -wz,   0,  wx],
            [wz,  wy, -wx,   0],
        ])

        q_new = q + dt * Omega @ q
        self._x[:4] = Quaternion.normalise(q_new)

        # Jacobian of process model w.r.t. state
        F = np.eye(self.STATE_DIM)
        F[:4, :4] += dt * Omega
        F[:4, 4:] = -0.5 * dt * self._q_xi_matrix(q)

        # Process noise matrix
        Q = np.zeros((self.STATE_DIM, self.STATE_DIM))
        Q[:4, :4] = np.eye(4) * self.cfg.q_gyro * dt**2
        Q[4:, 4:] = np.eye(3) * self.cfg.q_bias * dt

        self._P = F @ self._P @ F.T + Q

    def _correct_acc(self, acc: np.ndarray) -> None:
        """Accelerometer update: gravity should align with [0,0,g] in world frame."""
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-6:
            return

        acc_unit = acc / acc_norm
        q = self._x[:4]

        # Expected gravity direction in body frame
        g_world = np.array([0.0, 0.0, 1.0])
        g_body_expected = Quaternion.rotate_vector(Quaternion.conjugate(q), g_world)

        innovation = acc_unit - g_body_expected

        # Jacobian H (3 x 7): d(g_body) / d(q)  (numerical)
        H = self._acc_jacobian(q)

        R = np.eye(3) * self.cfg.r_acc
        self._apply_measurement_update(H, R, innovation)

    def _correct_mag(self, mag: np.ndarray) -> None:
        """Magnetometer update: correct yaw using magnetic north reference."""
        mag_norm = np.linalg.norm(mag)
        if mag_norm < 1e-6:
            return

        mag_unit = mag / mag_norm
        q = self._x[:4]

        # Rotate measurement to world frame; flatten to horizontal plane
        mag_world = Quaternion.rotate_vector(q, mag_unit)
        mag_world_flat = np.array([mag_world[0], mag_world[1], 0.0])
        norm_flat = np.linalg.norm(mag_world_flat)
        if norm_flat < 1e-6:
            return
        mag_ref = mag_world_flat / norm_flat

        # Expected measurement in body frame
        mag_expected = Quaternion.rotate_vector(Quaternion.conjugate(q), mag_ref)
        innovation = mag_unit - mag_expected

        H = self._mag_jacobian(q, mag_ref)
        R = np.eye(3) * self.cfg.r_mag
        self._apply_measurement_update(H, R, innovation)

    def _apply_measurement_update(
        self, H: np.ndarray, R: np.ndarray, innovation: np.ndarray
    ) -> None:
        """Standard Kalman measurement update."""
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)
        self._x += K @ innovation
        self._P = (np.eye(self.STATE_DIM) - K @ H) @ self._P

    # ------------------------------------------------------------------
    # Jacobians (numerical)
    # ------------------------------------------------------------------

    def _acc_jacobian(self, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        g_world = np.array([0.0, 0.0, 1.0])
        H = np.zeros((3, self.STATE_DIM))
        for i in range(4):
            q_plus = q.copy(); q_plus[i] += eps
            q_plus = Quaternion.normalise(q_plus)
            q_minus = q.copy(); q_minus[i] -= eps
            q_minus = Quaternion.normalise(q_minus)
            f_plus  = Quaternion.rotate_vector(Quaternion.conjugate(q_plus),  g_world)
            f_minus = Quaternion.rotate_vector(Quaternion.conjugate(q_minus), g_world)
            H[:, i] = (f_plus - f_minus) / (2 * eps)
        return H

    def _mag_jacobian(
        self, q: np.ndarray, mag_ref: np.ndarray, eps: float = 1e-6
    ) -> np.ndarray:
        H = np.zeros((3, self.STATE_DIM))
        for i in range(4):
            q_plus = q.copy(); q_plus[i] += eps; q_plus = Quaternion.normalise(q_plus)
            q_minus = q.copy(); q_minus[i] -= eps; q_minus = Quaternion.normalise(q_minus)
            f_plus  = Quaternion.rotate_vector(Quaternion.conjugate(q_plus),  mag_ref)
            f_minus = Quaternion.rotate_vector(Quaternion.conjugate(q_minus), mag_ref)
            H[:, i] = (f_plus - f_minus) / (2 * eps)
        return H

    @staticmethod
    def _q_xi_matrix(q: np.ndarray) -> np.ndarray:
        """Xi matrix: maps body angular velocity to quaternion derivative."""
        w, x, y, z = q
        return np.array([
            [-x, -y, -z],
            [ w, -z,  y],
            [ z,  w, -x],
            [-y,  x,  w],
        ])
