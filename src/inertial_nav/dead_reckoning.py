"""
Dead Reckoning & Orientation Estimation
========================================
Quaternion-based inertial dead reckoning using accelerometer and gyroscope data.

Ported and modernised from original MATLAB implementation (Nagesh Yadav, 2013).
Reference: "Accurate IMU-Based Orientation Estimation Under Conditions of
           Magnetic Distortion", Sensors, 2014.
           "Two stage kalman filtering for position estimation using dual
           Inertial Measurement Units", IEEE Sensors, 2011.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IMUReading:
    """Single timestep IMU sample."""
    acc: np.ndarray    # shape (3,) in m/s²
    gyro: np.ndarray   # shape (3,) in rad/s


@dataclass
class NavigationState:
    """Full navigation state at a single timestep."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))   # metres
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))   # m/s
    orientation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0, 0]))  # quaternion [w,x,y,z]

    def copy(self) -> "NavigationState":
        return NavigationState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=self.orientation.copy(),
        )


# ---------------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------------

class Quaternion:
    """
    Lightweight quaternion arithmetic helpers.

    Convention: q = [w, x, y, z]
    """

    @staticmethod
    def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Hamilton product of two unit quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    @staticmethod
    def conjugate(q: np.ndarray) -> np.ndarray:
        """Quaternion conjugate (= inverse for unit quaternions)."""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def normalise(q: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(q)
        if norm < 1e-10:
            return np.array([1.0, 0, 0, 0])
        return q / norm

    @staticmethod
    def from_euler_increments(wx: float, wy: float, wz: float) -> np.ndarray:
        """
        Small-angle quaternion from incremental rotation angles (radians).
        Uses first-order approximation valid for small dt.
        """
        angle = np.sqrt(wx**2 + wy**2 + wz**2)
        if angle < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0])
        half = angle / 2.0
        s = np.sin(half) / angle
        return np.array([np.cos(half), wx * s, wy * s, wz * s])

    @staticmethod
    def rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate vector v by quaternion q: q * [0,v] * q_conj."""
        v_quat = np.array([0.0, v[0], v[1], v[2]])
        q_conj = Quaternion.conjugate(q)
        rotated = Quaternion.multiply(Quaternion.multiply(q, v_quat), q_conj)
        return rotated[1:]

    @staticmethod
    def to_euler(q: np.ndarray) -> np.ndarray:
        """Convert quaternion to Euler angles [roll, pitch, yaw] in degrees."""
        w, x, y, z = q
        roll  = np.degrees(np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2)))
        pitch = np.degrees(np.arcsin(np.clip(2*(w*y - z*x), -1, 1)))
        yaw   = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2)))
        return np.array([roll, pitch, yaw])


# ---------------------------------------------------------------------------
# Dead reckoning engine
# ---------------------------------------------------------------------------

class DeadReckoning:
    """
    Quaternion dead reckoning integrator.

    Integrates gyroscope measurements to propagate orientation,
    rotates accelerometer readings into the world frame, removes
    gravity, and double-integrates to estimate position.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    zero_velocity_threshold : float
        Acceleration magnitude threshold below which velocity is zeroed
        to prevent unbounded drift (m/s²). Default 0.22 m/s².
    initialisation_window : int
        Number of samples used to estimate the static gravity vector.
    """

    def __init__(
        self,
        fs: float = 50.0,
        zero_velocity_threshold: float = 0.22,
        initialisation_window: int = 10,
    ) -> None:
        self.fs = fs
        self.dt = 1.0 / fs
        self.zero_vel_thresh = zero_velocity_threshold
        self.init_window = initialisation_window

    def run(self, readings: list[IMUReading]) -> list[NavigationState]:
        """
        Execute dead reckoning on a sequence of IMU readings.

        Parameters
        ----------
        readings : list[IMUReading]
            Ordered list of IMU samples.

        Returns
        -------
        list[NavigationState]
            Navigation state at each timestep (same length as readings).
        """
        if len(readings) < self.init_window:
            raise ValueError(
                f"Need at least {self.init_window} samples for initialisation, "
                f"got {len(readings)}."
            )

        gravity = self._estimate_gravity(readings)
        logger.info("Estimated gravity vector: %s", np.round(gravity, 4))

        state = NavigationState()
        history: list[NavigationState] = []

        for reading in readings:
            state = self._step(state, reading, gravity)
            history.append(state.copy())

        logger.info(
            "Dead reckoning complete. Final position: %s m",
            np.round(history[-1].position, 3),
        )
        return history

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _estimate_gravity(self, readings: list[IMUReading]) -> np.ndarray:
        """Estimate gravity from the mean of the first N static samples."""
        acc_stack = np.array([r.acc for r in readings[:self.init_window]])
        return acc_stack.mean(axis=0)

    def _step(
        self,
        state: NavigationState,
        reading: IMUReading,
        gravity: np.ndarray,
    ) -> NavigationState:
        dt = self.dt

        # 1. Update orientation via gyro integration
        wx, wy, wz = reading.gyro * dt
        dq = Quaternion.from_euler_increments(wx, wy, wz)
        q_new = Quaternion.normalise(Quaternion.multiply(state.orientation, dq))

        # 2. Rotate accelerometer into world frame and subtract gravity
        q_inv = Quaternion.conjugate(q_new)
        acc_world = Quaternion.rotate_vector(q_inv, reading.acc) - gravity

        # 3. Zero-velocity update: suppress noise-induced drift
        acc_world = np.where(np.abs(acc_world) <= self.zero_vel_thresh, 0.0, acc_world)

        # 4. Integrate acceleration → velocity → position
        pos_new = state.position + state.velocity * dt + 0.5 * acc_world * dt**2
        vel_new = state.velocity + acc_world * dt

        return NavigationState(
            position=pos_new,
            velocity=vel_new,
            orientation=q_new,
        )


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_dead_reckoning(
    acc_xyz: np.ndarray,
    gyro_xyz: np.ndarray,
    fs: float = 50.0,
    zero_velocity_threshold: float = 0.22,
) -> dict[str, np.ndarray]:
    """
    High-level convenience function for dead reckoning.

    Parameters
    ----------
    acc_xyz : np.ndarray, shape (N, 3)
        Calibrated accelerometer data in m/s².
    gyro_xyz : np.ndarray, shape (N, 3)
        Calibrated gyroscope data in rad/s.
    fs : float
        Sampling frequency in Hz.
    zero_velocity_threshold : float
        Zero-velocity suppression threshold in m/s².

    Returns
    -------
    dict with keys 'position' (N,3), 'velocity' (N,3), 'euler' (N,3)
    """
    readings = [
        IMUReading(acc=acc_xyz[i], gyro=gyro_xyz[i])
        for i in range(len(acc_xyz))
    ]

    dr = DeadReckoning(
        fs=fs,
        zero_velocity_threshold=zero_velocity_threshold,
    )
    history = dr.run(readings)

    positions  = np.array([s.position for s in history])
    velocities = np.array([s.velocity for s in history])
    eulers     = np.array([Quaternion.to_euler(s.orientation) for s in history])

    return {"position": positions, "velocity": velocities, "euler": eulers}
