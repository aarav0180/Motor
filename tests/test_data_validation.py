"""
tests/test_data_validation.py
-----------------------------
Unit tests for scripts/validate_data.py to verify data integrity validation
and failure detection.
"""

from pathlib import Path
import numpy as np
import pytest

from scripts.validate_data import (
    validate_signal_array,
    validate_cwru_dataset,
    validate_raw_dataset,
    DataValidationError,
)


class TestSignalArrayValidation:
    def test_valid_signal_passes(self):
        sig = np.random.randn(10_000).astype(np.float32)
        stats = validate_signal_array(sig, "test_channel", min_samples=1_000)
        assert stats["samples"] == 10_000
        assert stats["rms"] > 0
        assert stats["peak"] > 0
        assert stats["crest_factor"] > 1.0

    def test_too_few_samples_raises(self):
        sig = np.random.randn(500).astype(np.float32)
        with pytest.raises(DataValidationError, match="less than the minimum required"):
            validate_signal_array(sig, "short_channel", min_samples=1_000)

    def test_nan_values_raise(self):
        sig = np.random.randn(2_000).astype(np.float32)
        sig[100] = np.nan
        with pytest.raises(DataValidationError, match="contains 1 NaN values"):
            validate_signal_array(sig, "nan_channel", min_samples=1_000)

    def test_inf_values_raise(self):
        sig = np.random.randn(2_000).astype(np.float32)
        sig[250] = np.inf
        with pytest.raises(DataValidationError, match="contains 1 Inf values"):
            validate_signal_array(sig, "inf_channel", min_samples=1_000)

    def test_flatlined_signal_raises(self):
        sig = np.ones(2_000, dtype=np.float32) * 5.0
        with pytest.raises(DataValidationError, match="near-zero standard deviation"):
            validate_signal_array(sig, "dead_channel", min_samples=1_000)

    def test_non_1d_signal_raises(self):
        sig = np.random.randn(100, 10).astype(np.float32)
        with pytest.raises(DataValidationError, match="must be 1-D"):
            validate_signal_array(sig, "2d_channel", min_samples=500)


class TestDatasetValidators:
    def test_validate_cwru_dataset_passes_on_data_cwru(self):
        cwru_dir = Path("data/cwru")
        if cwru_dir.exists() and any(cwru_dir.glob("*.mat")):
            passed, issues = validate_cwru_dataset(cwru_dir, window_seconds=10)
            assert passed is True
            assert len(issues) == 0

    def test_validate_cwru_missing_dir(self, tmp_path):
        passed, issues = validate_cwru_dataset(tmp_path / "missing_dir")
        assert passed is False
        assert any("does not exist" in iss for iss in issues)

    def test_validate_cwru_empty_dir(self, tmp_path):
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        passed, issues = validate_cwru_dataset(empty_dir)
        assert passed is False
        assert any("No .mat files found" in iss for iss in issues)

    def test_validate_raw_dataset_with_files(self, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        # Save a valid 10-second window at 1000 Hz
        window = np.random.randn(10_000).astype(np.float32)
        np.save(str(raw_dir / "healthy_med_20261113.npy"), window)

        passed, issues = validate_raw_dataset(raw_dir, sample_rate=1000, window_seconds=10)
        assert passed is True
        assert len(issues) == 0
