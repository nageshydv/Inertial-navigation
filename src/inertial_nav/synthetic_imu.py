"""
Synthetic IMU Data Generator
=============================
Generates realistic accelerometer and gyroscope signals for testing and
demonstrating the dead reckoning and EKF pipelines without physical hardware.
"""

from __future__ import annotations

import numpy as np


class SyntheticIMU:
    """
    Simulates IMU readings for a body moving through a parameterised trajectory.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    noise_acc : float
        Accelerometer white noise std dev in m/s².
    noise_gyro : float
        Gyroscope white noise std dev in rad/s.
    bias_gyro : np.ndarray or None
        Constant gyroscope bias in rad/s. Defaults to a small random bias.
    gravity : float
        Gravitational acceleration in m/s².
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        fs: float = 50.0,
        noise_acc: float = 0.05,
        noise_gyro: float = 0.002,
        bias_gyro: np.ndarray | None = None,
        gravity: float = 9.81,
        seed: int = 42,
    ) -> None:
        self.fs = fs
        self.dt = 1.0 / fs
        self.noise_acc = noise_acc
        self.noise_gyro = noise_gyro
        self.gravity = gravity
        self.rng = np.random.default_rng(seed)
        self.bias_gyro = (
            bias_gyro if bias_gyro is not None
            else self.rng.normal(0, 0.001, 3)
        )

    def generate_static(self, duration: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate a completely static IMU (sensor lying flat).

        Returns
        -------
        acc : np.ndarray, shape (N, 3)  – gravity + noise
        gyro : np.ndarray, shape (N, 3) – bias + noise
        """
        N = int(duration * self.fs)
        acc  = np.tile([0.0, 0.0, self.gravity], (N, 1))
        gyro = np.tile(self.bias_gyro, (N, 1))
        acc  += self.rng.normal(0, self.noise_acc,  (N, 3))
        gyro += self.rng.normal(0, self.noise_gyro, (N, 3))
        return acc, gyro

    def generate_rotation(
        self,
        axis: str = "z",
        rate_deg_s: float = 45.0,
        duration: float = 4.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate a constant-rate rotation about a body axis.

        Parameters
        ----------
        axis : 'x', 'y', or 'z'
        rate_deg_s : rotation rate in degrees/second
        duration : float in seconds
        """
        N = int(duration * self.fs)
        rate_rad = np.radians(rate_deg_s)
        axis_map = {"x": 0, "y": 1, "z": 2}
        idx = axis_map[axis]

        gyro = np.tile(self.bias_gyro, (N, 1))
        gyro[:, idx] += rate_rad
        gyro += self.rng.normal(0, self.noise_gyro, (N, 3))

        # Accelerometer sees gravity rotating in body frame
        acc = np.zeros((N, 3))
        angle = np.arange(N) * self.dt * rate_rad
        if axis == "z":
            acc[:, 0] = self.gravity * np.sin(angle)
            acc[:, 1] = self.gravity * np.cos(angle)
            acc[:, 2] = 0.0
        elif axis == "x":
            acc[:, 0] = 0.0
            acc[:, 1] = self.gravity * np.cos(angle)
            acc[:, 2] = self.gravity * np.sin(angle)
        else:  # y
            acc[:, 0] = self.gravity * np.sin(angle)
            acc[:, 1] = 0.0
            acc[:, 2] = self.gravity * np.cos(angle)

        acc += self.rng.normal(0, self.noise_acc, (N, 3))
        return acc, gyro

    def generate_walk(
        self,
        duration: float = 10.0,
        step_freq_hz: float = 1.8,
        step_amplitude: float = 1.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate periodic foot-strike accelerations mimicking walking.

        Returns sinusoidal acceleration bursts in the vertical axis
        to emulate gait, commonly used in pedestrian dead reckoning.
        """
        N = int(duration * self.fs)
        t = np.arange(N) * self.dt

        acc = np.zeros((N, 3))
        acc[:, 2] = self.gravity + step_amplitude * np.sin(2 * np.pi * step_freq_hz * t)
        acc[:, 0] = 0.3 * np.sin(2 * np.pi * step_freq_hz * t + np.pi / 4)
        acc += self.rng.normal(0, self.noise_acc, (N, 3))

        gyro = np.tile(self.bias_gyro, (N, 1))
        gyro[:, 2] += 0.05 * np.sin(2 * np.pi * step_freq_hz * 0.5 * t)
        gyro += self.rng.normal(0, self.noise_gyro, (N, 3))

        return acc, gyro
