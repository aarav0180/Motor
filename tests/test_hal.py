"""
tests/test_hal.py - Unit tests for HAL sensors.

Run with:  pytest tests/
"""

import numpy as np
import pytest

from hal.base import VibrationSensor
from hal.mock_sensor import MockSensor


SAMPLE_RATE = 12_000
WINDOW_SIZE = SAMPLE_RATE * 10


@pytest.fixture
def healthy_data():
    rng = np.random.default_rng(1)
    return rng.standard_normal(WINDOW_SIZE * 5).astype(np.float32)


class TestMockSensor:
    def test_get_window_shape(self, healthy_data):
        sensor = MockSensor(healthy_data, sample_rate=SAMPLE_RATE)
        window = sensor.get_10s_window()
        assert window.shape == (WINDOW_SIZE,)

    def test_get_window_dtype(self, healthy_data):
        sensor = MockSensor(healthy_data, sample_rate=SAMPLE_RATE)
        window = sensor.get_10s_window()
        assert window.dtype == np.float32

    def test_cursor_advances(self, healthy_data):
        sensor = MockSensor(healthy_data, sample_rate=SAMPLE_RATE)
        sensor.get_10s_window()
        assert sensor._cursor == WINDOW_SIZE

    def test_loops_at_end(self, healthy_data):
        # Provide exactly 1.5 windows of data so the second call loops
        short_data = healthy_data[: int(WINDOW_SIZE * 1.5)]
        sensor = MockSensor(short_data, sample_rate=SAMPLE_RATE)
        sensor.get_10s_window()   # first window
        sensor.get_10s_window()   # should loop back
        assert sensor._cursor == WINDOW_SIZE   # cursor reset + one window

    def test_reset(self, healthy_data):
        sensor = MockSensor(healthy_data, sample_rate=SAMPLE_RATE)
        sensor.get_10s_window()
        sensor.reset()
        assert sensor._cursor == 0
        assert sensor.windows_served == 0

    def test_windows_served_counter(self, healthy_data):
        sensor = MockSensor(healthy_data, sample_rate=SAMPLE_RATE)
        for _ in range(3):
            sensor.get_10s_window()
        assert sensor.windows_served == 3

    def test_rejects_2d_data(self):
        with pytest.raises(ValueError):
            MockSensor(np.ones((100, 10)))

    def test_no_simulate_delay_is_fast(self, healthy_data):
        import time
        sensor = MockSensor(healthy_data, sample_rate=SAMPLE_RATE, simulate_delay=False)
        t0 = time.perf_counter()
        sensor.get_10s_window()
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, "MockSensor without delay should return instantly"

    def test_validate_window_called(self, healthy_data):
        """validate_window should not raise for valid data."""
        sensor = MockSensor(healthy_data, sample_rate=SAMPLE_RATE)
        window = sensor.get_10s_window()
        sensor.validate_window(window)   # must not raise

    def test_from_npy(self, tmp_path, healthy_data):
        path = str(tmp_path / "data.npy")
        np.save(path, healthy_data)
        sensor = MockSensor.from_npy(path, sample_rate=SAMPLE_RATE)
        window = sensor.get_10s_window()
        assert len(window) == WINDOW_SIZE

    def test_dynamic_mock_sensor_alternates(self):
        from hal.mock_sensor import create_dynamic_mock_sensor
        sensor = create_dynamic_mock_sensor(cwru_dir="data/cwru", sample_rate=12_000, fault_every_n=3)
        assert sensor.dynamic_mode is True
        w1 = sensor.get_10s_window()
        assert w1.shape == (120_000,)
        w2 = sensor.get_10s_window()
        assert w2.shape == (120_000,)
        w3 = sensor.get_10s_window()
        assert w3.shape == (120_000,)
