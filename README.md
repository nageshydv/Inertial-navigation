# Inertial Navigation — IMU Dead Reckoning & Orientation Estimation

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](tests/)

A clean Python implementation of quaternion-based inertial navigation using data from 6-DOF/9-DOF Inertial Measurement Units (IMUs). Originally developed during PhD research at University College Dublin.

## Background

This project implements the algorithms published in:

- *"Accurate IMU-Based Orientation Estimation Under Conditions of Magnetic Distortion"*, **Sensors**, 2014
- *"Fast calibration of a 9-DOF IMU using a 3 DOF position tracker and a semi-random motion sequence"*, **Elsevier Measurement**, 2016
- *"Two stage Kalman filtering for position estimation using dual Inertial Measurement Units"*, **IEEE Sensors**, 2011

The original MATLAB prototype has been fully rewritten in modern Python with proper abstractions, type hints, unit tests and a demonstration notebook.

---

## Features

| Module | Description |
|---|---|
| `calibration.py` | Six-position static accelerometer and gyroscope calibration |
| `dead_reckoning.py` | Quaternion integration → velocity → position |
| `ekf_orientation.py` | Extended Kalman Filter with accelerometer + optional magnetometer fusion |
| `synthetic_imu.py` | Realistic IMU signal simulator for testing without hardware |

---

## Quickstart

```bash
git clone https://github.com/nagesh-yadav/inertial-navigation.git
cd inertial-navigation
pip install -e ".[dev]"
pytest
```

### Dead Reckoning from Sensor Data

```python
import numpy as np
from src.inertial_nav import run_dead_reckoning, SyntheticIMU

# Simulate walking motion
sim = SyntheticIMU(fs=50, seed=42)
acc, gyro = sim.generate_walk(duration=10.0)

# Run dead reckoning
result = run_dead_reckoning(acc, gyro, fs=50.0, zero_velocity_threshold=0.22)

print("Final position (m):", result["position"][-1].round(3))
print("Final Euler angles (°):", result["euler"][-1].round(2))
```

### EKF Orientation Estimation

```python
from src.inertial_nav import OrientationEKF, EKFConfig, Quaternion

cfg = EKFConfig(q_gyro=1e-4, r_acc=1e-1)
ekf = OrientationEKF(fs=50.0, cfg=cfg)

# Run on sequences of shape (N, 3)
quaternions = ekf.run_batch(gyro_seq=gyro, acc_seq=acc)

# Convert to Euler angles
import numpy as np
eulers = np.array([Quaternion.to_euler(q) for q in quaternions])
print("Roll, Pitch, Yaw (°):", eulers[-1].round(2))
```

### IMU Calibration

```python
from pathlib import Path
from src.inertial_nav import IMUCalibrator

calibrator = IMUCalibrator(gyro_range=500, gravity=9.81)
params = calibrator.calibrate_from_directory(Path("data/calib/sensor_001"))

calibrator.save_params(params, "data/calib/sensor_001_cal.npz")
print("Acc offsets:", params.offset_acc)
print("Acc sensitivities:", params.sensitivity_acc)
```

---

## Project Structure

```
inertial-navigation/
├── src/
│   └── inertial_nav/
│       ├── __init__.py
│       ├── calibration.py       # Six-position IMU calibration
│       ├── dead_reckoning.py    # Quaternion dead reckoning engine
│       ├── ekf_orientation.py   # Extended Kalman Filter
│       └── synthetic_imu.py     # Test signal generator
├── tests/
│   └── test_inertial_nav.py    # 20+ unit and integration tests
├── notebooks/
│   └── demo_dead_reckoning.ipynb
├── data/
│   └── sample/                  # Sample calibration CSVs
├── pyproject.toml
└── README.md
```

---

## Algorithm Overview

### Calibration
The six-position method places the sensor with each axis aligned with +g and -g. For each axis, the ADC midpoint (offset) and scale factor (sensitivity) are derived from the mean positive/negative readings. Gyroscope bias is extracted from the zero-rate mean across all positions.

### Dead Reckoning
1. **Gyro integration**: Angular velocity → quaternion increment via small-angle approximation
2. **Gravity subtraction**: Rotate measured acceleration to world frame; subtract static gravity estimate
3. **Zero-velocity update (ZVU)**: Suppress sub-threshold accelerations to control drift
4. **Double integration**: Acceleration → velocity → position

### Extended Kalman Filter
The EKF state is `[q_w, q_x, q_y, q_z, b_x, b_y, b_z]` (quaternion + gyro bias).
- **Predict**: Propagate orientation using bias-corrected gyro; propagate covariance with linearised kinematics
- **Correct (accel)**: Update using gravity direction alignment
- **Correct (mag, optional)**: Update yaw using horizontal magnetic north reference

---

## Running Tests

```bash
pytest tests/ -v --tb=short
# With coverage
pytest tests/ --cov=src/inertial_nav --cov-report=term-missing
```

---

## Contributing

Issues and pull requests welcome. See `pyproject.toml` for dev tooling (ruff, mypy, pytest).

---

## License

MIT © Nagesh Yadav
