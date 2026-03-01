"""
inertial_nav
============
Python implementation of quaternion-based inertial navigation:
  - IMU calibration (six-position static method)
  - Dead reckoning via quaternion integration
  - Extended Kalman Filter for orientation under magnetic distortion
  - Synthetic IMU data generator

Original algorithms: Nagesh Yadav (PhD, UCD 2013)
References:
  Sensors 2014 | Elsevier Measurement 2016 | IEEE Sensors 2011
"""

from .calibration import CalibrationParams, IMUCalibrator
from .dead_reckoning import DeadReckoning, IMUReading, NavigationState, Quaternion, run_dead_reckoning
from .ekf_orientation import EKFConfig, OrientationEKF
from .synthetic_imu import SyntheticIMU

__all__ = [
    "CalibrationParams",
    "IMUCalibrator",
    "DeadReckoning",
    "IMUReading",
    "NavigationState",
    "Quaternion",
    "run_dead_reckoning",
    "EKFConfig",
    "OrientationEKF",
    "SyntheticIMU",
]
