"""
Unit and integration tests for inertial_nav package.
Run with: pytest tests/
"""

import numpy as np
import pytest

from src.inertial_nav import (
    CalibrationParams,
    DeadReckoning,
    EKFConfig,
    IMUReading,
    OrientationEKF,
    Quaternion,
    SyntheticIMU,
    run_dead_reckoning,
)


# ---------------------------------------------------------------------------
# Quaternion tests
# ---------------------------------------------------------------------------

class TestQuaternion:
    def test_normalise_unit(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(Quaternion.normalise(q), q)

    def test_normalise_arbitrary(self):
        q = np.array([2.0, 1.0, 0.5, 0.3])
        q_n = Quaternion.normalise(q)
        assert abs(np.linalg.norm(q_n) - 1.0) < 1e-10

    def test_conjugate(self):
        q = np.array([0.7071, 0.7071, 0.0, 0.0])
        qc = Quaternion.conjugate(q)
        assert qc[0] == q[0]
        np.testing.assert_allclose(qc[1:], -q[1:])

    def test_multiply_identity(self):
        q = Quaternion.normalise(np.array([1.0, 0.2, 0.3, 0.1]))
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        result = Quaternion.multiply(q, identity)
        np.testing.assert_allclose(result, q, atol=1e-10)

    def test_multiply_inverse_gives_identity(self):
        q = Quaternion.normalise(np.array([0.5, 0.5, 0.5, 0.5]))
        q_inv = Quaternion.conjugate(q)
        product = Quaternion.multiply(q, q_inv)
        np.testing.assert_allclose(product, [1, 0, 0, 0], atol=1e-10)

    def test_rotate_vector_no_rotation(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])  # identity
        v = np.array([1.0, 2.0, 3.0])
        rotated = Quaternion.rotate_vector(q, v)
        np.testing.assert_allclose(rotated, v, atol=1e-10)

    def test_rotate_90_degrees_z(self):
        """90° rotation around z-axis: x→y."""
        q = Quaternion.normalise(np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]))
        v = np.array([1.0, 0.0, 0.0])
        rotated = Quaternion.rotate_vector(q, v)
        np.testing.assert_allclose(rotated, [0, 1, 0], atol=1e-6)

    def test_euler_identity(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        euler = Quaternion.to_euler(q)
        np.testing.assert_allclose(euler, [0, 0, 0], atol=1e-10)


# ---------------------------------------------------------------------------
# Dead reckoning tests
# ---------------------------------------------------------------------------

class TestDeadReckoning:
    def setup_method(self):
        self.sim = SyntheticIMU(fs=50, seed=0)

    def test_static_position_near_zero(self):
        """Static sensor: position should stay near origin."""
        acc, gyro = self.sim.generate_static(duration=5.0)
        result = run_dead_reckoning(acc, gyro, fs=50.0, zero_velocity_threshold=0.22)
        final_pos = result["position"][-1]
        # With ZVU, drift should be minimal
        assert np.linalg.norm(final_pos) < 2.0, f"Too much drift: {final_pos}"

    def test_output_shapes(self):
        acc, gyro = self.sim.generate_static(duration=2.0)
        result = run_dead_reckoning(acc, gyro, fs=50.0)
        N = len(acc)
        assert result["position"].shape == (N, 3)
        assert result["velocity"].shape == (N, 3)
        assert result["euler"].shape == (N, 3)

    def test_orientation_remains_unit_quaternion(self):
        acc, gyro = self.sim.generate_rotation(axis="z", rate_deg_s=30.0, duration=3.0)
        readings = [IMUReading(acc=acc[i], gyro=gyro[i]) for i in range(len(acc))]
        dr = DeadReckoning(fs=50.0)
        history = dr.run(readings)
        for state in history:
            norm = np.linalg.norm(state.orientation)
            assert abs(norm - 1.0) < 1e-6, f"Quaternion not unit: norm={norm}"

    def test_insufficient_samples_raises(self):
        readings = [IMUReading(acc=np.zeros(3), gyro=np.zeros(3))] * 5
        dr = DeadReckoning(fs=50.0, initialisation_window=10)
        with pytest.raises(ValueError):
            dr.run(readings)


# ---------------------------------------------------------------------------
# EKF tests
# ---------------------------------------------------------------------------

class TestOrientationEKF:
    def setup_method(self):
        self.sim = SyntheticIMU(fs=50, seed=1)

    def test_static_converges_to_identity_orientation(self):
        """Static upright sensor: roll and pitch should converge near zero."""
        acc, gyro = self.sim.generate_static(duration=10.0)
        ekf = OrientationEKF(fs=50.0)
        quaternions = ekf.run_batch(gyro, acc)
        eulers = np.array([Quaternion.to_euler(q) for q in quaternions])
        # After 10 s, roll and pitch should be within ±5°
        final_rp = eulers[-1, :2]
        assert np.all(np.abs(final_rp) < 5.0), f"Roll/pitch too large: {final_rp}"

    def test_output_unit_quaternions(self):
        acc, gyro = self.sim.generate_rotation(duration=3.0)
        ekf = OrientationEKF(fs=50.0)
        quaternions = ekf.run_batch(gyro, acc)
        norms = np.linalg.norm(quaternions, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_batch_shape(self):
        acc, gyro = self.sim.generate_static(duration=2.0)
        ekf = OrientationEKF(fs=50.0)
        quaternions = ekf.run_batch(gyro, acc)
        assert quaternions.shape == (len(acc), 4)


# ---------------------------------------------------------------------------
# Calibration params tests
# ---------------------------------------------------------------------------

class TestCalibrationParams:
    def test_apply_to_accel_identity(self):
        params = CalibrationParams(
            offset_acc=np.zeros(3),
            sensitivity_acc=np.ones(3),
        )
        raw = np.array([2048.0, 2048.0, 2048.0 + 9.81])
        cal = params.apply_to_accel(raw)
        np.testing.assert_allclose(cal, [0, 0, 9.81], atol=1e-6)

    def test_apply_to_gyro_bias_removal(self):
        bias = np.array([10.0, -5.0, 3.0])
        params = CalibrationParams(
            offset_gyro=bias,
            sensitivity_gyro=np.ones(3),
        )
        raw = np.array([2048.0 + 10, 2048.0 - 5, 2048.0 + 3])
        cal = params.apply_to_gyro(raw)
        np.testing.assert_allclose(cal, [0, 0, 0], atol=1e-6)


# ---------------------------------------------------------------------------
# Synthetic IMU tests
# ---------------------------------------------------------------------------

class TestSyntheticIMU:
    def test_static_gravity_magnitude(self):
        sim = SyntheticIMU(fs=50, noise_acc=0.0, noise_gyro=0.0, seed=0)
        acc, _ = sim.generate_static(duration=1.0)
        mean_mag = np.linalg.norm(acc.mean(axis=0))
        assert abs(mean_mag - 9.81) < 0.01

    def test_walk_shape(self):
        sim = SyntheticIMU(fs=50)
        acc, gyro = sim.generate_walk(duration=5.0)
        assert acc.shape == (250, 3)
        assert gyro.shape == (250, 3)

    def test_rotation_gyro_axis(self):
        sim = SyntheticIMU(fs=50, noise_gyro=0.0, seed=0)
        sim.bias_gyro = np.zeros(3)
        _, gyro = sim.generate_rotation(axis="z", rate_deg_s=90.0, duration=1.0)
        mean_rate = gyro.mean(axis=0)
        expected_z = np.radians(90.0)
        assert abs(mean_rate[2] - expected_z) < 0.01
