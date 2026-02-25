"""
IMU Calibration Module
======================
Six-position static calibration for accelerometer and gyroscope axes.

Ported and modernised from original MATLAB implementation (Nagesh Yadav, 2013).
Reference: "Fast calibration of a 9-DOF IMU using a 3 DOF position tracker
           and a semi-random motion sequence", Elsevier Measurement, 2016.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CalibrationParams:
    """Stores calibration offsets and sensitivities for a 6-DOF IMU."""

    # Accelerometer
    offset_acc: np.ndarray = field(default_factory=lambda: np.zeros(3))   # [ofx, ofy, ofz]
    sensitivity_acc: np.ndarray = field(default_factory=lambda: np.ones(3))  # [sx, sy, sz]

    # Gyroscope
    offset_gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))
    sensitivity_gyro: np.ndarray = field(default_factory=lambda: np.ones(3))

    # Non-linearity bounds (positive/negative half-range means)
    acc_pos_limits: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acc_neg_limits: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_pos_limits: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_neg_limits: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Raw ADC midpoint (12-bit: 2048)
    ADC_MIDPOINT: int = 2048

    def apply_to_accel(self, raw: np.ndarray) -> np.ndarray:
        """Convert raw ADC accelerometer counts to m/s²."""
        return (raw - self.offset_acc - self.ADC_MIDPOINT) * self.sensitivity_acc

    def apply_to_gyro(self, raw: np.ndarray) -> np.ndarray:
        """Convert raw ADC gyroscope counts to °/s."""
        return (raw - self.offset_gyro - self.ADC_MIDPOINT) * self.sensitivity_gyro


class IMUCalibrator:
    """
    Six-position static calibration for a 9-DOF IMU.

    Each axis is calibrated by placing the sensor in +/- orientations
    relative to gravity, recording steady-state ADC values, and computing
    the offset and sensitivity scale factor.

    Parameters
    ----------
    gyro_range : float
        Full-scale gyroscope range in °/s (default 500).
    adc_midpoint : int
        ADC zero-g midpoint (default 2048 for 12-bit ADC).
    gravity : float
        Local gravitational acceleration in m/s² (default 9.81).
    """

    ACC_COLS = {
        "x": 4,   # 0-indexed column indices in the CSV files
        "y": 5,
        "z": 6,
    }
    GYRO_COLS = {"x": 7, "y": 8, "z": 9}

    def __init__(
        self,
        gyro_range: float = 500.0,
        adc_midpoint: int = 2048,
        gravity: float = 9.81,
    ) -> None:
        self.gyro_range = gyro_range
        self.adc_midpoint = adc_midpoint
        self.gravity = gravity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate_from_directory(self, data_dir: Path | str) -> CalibrationParams:
        """
        Run the full six-position calibration from a directory of CSV files.

        The directory must contain:
            plus_x_acc.csv, minus_x_acc.csv
            plus_y_acc.csv, minus_y_acc.csv
            plus_z_acc.csv, minus_z_acc.csv

        Parameters
        ----------
        data_dir : Path or str
            Directory containing the six orientation CSV files.

        Returns
        -------
        CalibrationParams
            Populated calibration parameters ready for use.
        """
        data_dir = Path(data_dir)
        logger.info("Loading calibration data from %s", data_dir)

        axes = ["x", "y", "z"]
        acc_col_map = [self.ACC_COLS["x"], self.ACC_COLS["y"], self.ACC_COLS["z"]]

        offsets = np.zeros(3)
        sensitivities = np.zeros(3)

        all_gyro_raw: list[np.ndarray] = []

        for i, axis in enumerate(axes):
            pos_data = self._load_csv(data_dir / f"plus_{axis}_acc.csv")
            neg_data = self._load_csv(data_dir / f"minus_{axis}_acc.csv")

            col = acc_col_map[i]
            max_val = pos_data[:, col].mean()
            min_val = neg_data[:, col].mean()

            midpoint = (max_val + min_val) / 2.0
            offsets[i] = midpoint - self.adc_midpoint

            tmp_pos = max_val - offsets[i] - self.adc_midpoint
            tmp_neg = min_val - offsets[i] - self.adc_midpoint
            sensitivities[i] = (self.gravity * 2.0) / (tmp_pos - tmp_neg)

            # Accumulate gyro data across all positions
            for data in (pos_data, neg_data):
                all_gyro_raw.append(data[:, [self.GYRO_COLS["x"],
                                              self.GYRO_COLS["y"],
                                              self.GYRO_COLS["z"]]])

        gyro_stacked = np.vstack(all_gyro_raw)
        gyro_offsets = gyro_stacked.mean(axis=0) - self.adc_midpoint
        gyro_sens = np.full(3, self.gyro_range / self.adc_midpoint)

        params = CalibrationParams(
            offset_acc=offsets,
            sensitivity_acc=sensitivities,
            offset_gyro=gyro_offsets,
            sensitivity_gyro=gyro_sens,
        )

        # Compute non-linearity limits for diagnostics
        params.acc_pos_limits, params.acc_neg_limits = self._compute_nonlinearity(
            data_dir, params
        )

        logger.info(
            "Calibration complete. Acc offsets=%s, sensitivities=%s",
            np.round(offsets, 4),
            np.round(sensitivities, 6),
        )
        return params

    def save_params(self, params: CalibrationParams, path: Path | str) -> None:
        """Persist calibration parameters to a NumPy .npz file."""
        path = Path(path)
        np.savez(
            path,
            offset_acc=params.offset_acc,
            sensitivity_acc=params.sensitivity_acc,
            offset_gyro=params.offset_gyro,
            sensitivity_gyro=params.sensitivity_gyro,
            acc_pos_limits=params.acc_pos_limits,
            acc_neg_limits=params.acc_neg_limits,
        )
        logger.info("Calibration params saved to %s", path)

    @staticmethod
    def load_params(path: Path | str) -> CalibrationParams:
        """Load calibration parameters from a .npz file."""
        data = np.load(Path(path))
        return CalibrationParams(
            offset_acc=data["offset_acc"],
            sensitivity_acc=data["sensitivity_acc"],
            offset_gyro=data["offset_gyro"],
            sensitivity_gyro=data["sensitivity_gyro"],
            acc_pos_limits=data["acc_pos_limits"],
            acc_neg_limits=data["acc_neg_limits"],
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_csv(path: Path) -> np.ndarray:
        df = pd.read_csv(path, header=0)
        return df.values.astype(float)

    def _compute_nonlinearity(
        self, data_dir: Path, params: CalibrationParams
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute mean positive/negative residuals for non-linearity diagnostics."""
        pos_limits = np.zeros(3)
        neg_limits = np.zeros(3)
        axes = ["x", "y", "z"]
        col_map = [self.ACC_COLS["x"], self.ACC_COLS["y"], self.ACC_COLS["z"]]

        for i, axis in enumerate(axes):
            raw = self._load_csv(data_dir / f"plus_{axis}_acc.csv")[:, col_map[i]]
            calibrated = params.apply_to_accel(
                np.column_stack([raw if j == i else np.zeros_like(raw) for j in range(3)])
            )[:, i]
            total = np.linalg.norm(calibrated.mean())
            angle = np.arccos(np.clip(calibrated.mean() / (total + 1e-9), -1, 1))
            residuals = calibrated - total * np.cos(angle)
            pos_limits[i] = residuals[residuals >= 0].mean() if (residuals >= 0).any() else 0.0
            neg_limits[i] = residuals[residuals < 0].mean() if (residuals < 0).any() else 0.0

        return pos_limits, neg_limits
