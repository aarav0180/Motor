"""
tests/test_pipeline_e2e.py
--------------------------
End-to-end integration tests for data.pipeline (DataPipeline & DataMode).
Tests the single-switch pattern between MockSensor and PhysicalSensor,
verifying that no breaking changes disrupt the contract.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from data.pipeline import DataPipeline, DataMode
from data.loader import CWRU_SAMPLE_RATE_HZ, CWRU_RUN_FREQ_HZ
from hal import MockSensor, PhysicalSensor


class TestDataPipelineMockMode:
    """Test DataPipeline operating in DataMode.MOCK (laptop / CI simulation)."""

    def test_pipeline_setup_mock_mode(self):
        # Uses files in data/cwru/ created by generate_sample_cwru
        pipeline = DataPipeline(
            mode=DataMode.MOCK,
            cwru_dir="data/cwru",
            window_seconds=10,
            mock_fraction=0.20,
            train_fraction=0.80,
        )
        pipeline.setup()

        assert pipeline._is_ready is True
        sensor = pipeline.get_sensor()
        assert isinstance(sensor, MockSensor)
        assert sensor.sample_rate == CWRU_SAMPLE_RATE_HZ

        split = pipeline.get_split()
        assert len(split.train_healthy) > 0
        assert len(split.mock_sensor) > 0
        assert len(split.eval_faulty) > 0

    def test_mock_sensor_window_pull(self):
        pipeline = DataPipeline(
            mode=DataMode.MOCK,
            cwru_dir="data/cwru",
            window_seconds=10,
        ).setup()

        sensor = pipeline.get_sensor()
        window = sensor.get_10s_window()

        assert isinstance(window, np.ndarray)
        assert len(window) == CWRU_SAMPLE_RATE_HZ * 10
        assert not np.isnan(window).any()

        # Extract features from live window
        feat = pipeline.extract_live_features(window)
        assert "rms" in feat
        assert "crest_factor" in feat
        assert "kurtosis" in feat
        assert feat["rms"] > 0

    def test_train_and_eval_data_structures(self):
        pipeline = DataPipeline(
            mode=DataMode.MOCK,
            cwru_dir="data/cwru",
            window_seconds=10,
        ).setup()

        X_train, train_feats = pipeline.get_train_data()
        assert len(X_train) == len(train_feats)
        assert X_train.shape[1] == CWRU_SAMPLE_RATE_HZ * 10
        assert all("rms" in f for f in train_feats)

        X_eval, eval_feats, y_true = pipeline.get_eval_data()
        assert len(X_eval) == len(eval_feats) == len(y_true)
        assert set(y_true).issubset({0, 1})
        assert 0 in y_true  # at least some healthy eval windows
        assert 1 in y_true  # at least some faulty eval windows

    def test_not_setup_raises_runtime_error(self):
        pipeline = DataPipeline(mode=DataMode.MOCK)
        with pytest.raises(RuntimeError, match="not been set up"):
            pipeline.get_sensor()


class TestDataPipelineRealModeSwitch:
    """Test switching to DataMode.REAL (physical Pi sensor) via the single switch."""

    def test_switch_to_real_mode_instantiates_physical_sensor(self):
        # Mock smbus2 so this runs on any OS without hardware attached
        mock_bus_instance = MagicMock()
        mock_smbus2 = MagicMock()
        mock_smbus2.SMBus.return_value = mock_bus_instance

        with patch.dict("sys.modules", {"smbus2": mock_smbus2}):
            pipeline = DataPipeline(
                mode=DataMode.REAL,  # ★ THE SINGLE SWITCH ★
                cwru_dir="data/cwru",
                sample_rate=1000,
                pi_bus=1,
                pi_address=0x53,
                pi_sample_rate=1000,
            )
            pipeline.setup()

            sensor = pipeline.get_sensor()
            assert isinstance(sensor, PhysicalSensor)
            assert sensor.address == 0x53
            assert sensor.sample_rate == 1000

            # CWRU evaluation data is STILL available for cross-dataset checks!
            split = pipeline.get_split()
            assert len(split.eval_faulty) > 0
            X_eval, eval_feats, y_true = pipeline.get_eval_data()
            assert len(eval_feats) > 0

    def test_cleaner_preset_switches_with_mode(self):
        pipeline_mock = DataPipeline(mode=DataMode.MOCK)
        assert pipeline_mock.cleaner_config.bandpass_high_hz == 5_500.0  # CWRU Nyquist

        pipeline_real = DataPipeline(mode=DataMode.REAL)
        assert pipeline_real.cleaner_config.bandpass_high_hz == 450.0   # ADXL345 Nyquist
