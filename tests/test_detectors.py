"""
tests/test_detectors.py — Unit tests for all three detectors.

Run with:  pytest tests/
"""

import numpy as np
import pytest

from features.extractor import FeatureExtractor
from detectors import (
    RMSThresholdDetector,
    IsolationForestDetector,
    AutoencoderDetector,
)

SAMPLE_RATE = 1000
RUN_FREQ    = 40.0
WINDOW_SIZE = SAMPLE_RATE * 10
N_HEALTHY   = 30
N_FAULTY    = 10

rng = np.random.default_rng(42)


def _make_healthy():
    return (0.1 * rng.standard_normal(WINDOW_SIZE)).astype(np.float32)


def _make_faulty():
    w = (0.1 * rng.standard_normal(WINDOW_SIZE)).astype(np.float32)
    # Inject large impulses to simulate a fault
    for i in range(0, WINDOW_SIZE, 300):
        w[i] += rng.choice([-5.0, 5.0])
    return w


@pytest.fixture(scope="module")
def healthy_feats():
    fx = FeatureExtractor(sample_rate=SAMPLE_RATE, run_freq_hz=RUN_FREQ)
    return [fx.transform(_make_healthy()).to_dict() for _ in range(N_HEALTHY)]


@pytest.fixture(scope="module")
def faulty_feats():
    fx = FeatureExtractor(sample_rate=SAMPLE_RATE, run_freq_hz=RUN_FREQ)
    return [fx.transform(_make_faulty()).to_dict() for _ in range(N_FAULTY)]


# ---------------------------------------------------------------------------
# RMS Threshold
# ---------------------------------------------------------------------------

class TestRMSThreshold:
    def test_fit_and_predict_normal(self, healthy_feats):
        det = RMSThresholdDetector(sigma=3.0)
        det.fit(healthy_feats)
        score, label = det.predict(healthy_feats[0])
        assert isinstance(score, float)
        assert label in ("normal", "anomaly")

    def test_raises_before_fit(self, healthy_feats):
        det = RMSThresholdDetector()
        with pytest.raises(RuntimeError):
            det.predict(healthy_feats[0])

    def test_save_load(self, healthy_feats, tmp_path):
        det = RMSThresholdDetector()
        det.fit(healthy_feats)
        path = str(tmp_path / "rms.pkl")
        det.save(path)
        loaded = RMSThresholdDetector.load(path)
        assert abs(det.threshold - loaded.threshold) < 1e-9


# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------

class TestIsolationForest:
    def test_fit_and_score(self, healthy_feats, faulty_feats):
        det = IsolationForestDetector(contamination=0.05)
        det.fit(healthy_feats)
        h_score = det.score(healthy_feats[0])
        f_score = det.score(faulty_feats[0])
        assert isinstance(h_score, float)
        assert isinstance(f_score, float)

    def test_save_load(self, healthy_feats, tmp_path):
        det = IsolationForestDetector()
        det.fit(healthy_feats)
        path = str(tmp_path / "iso.pkl")
        det.save(path)
        loaded = IsolationForestDetector.load(path)
        assert abs(det.threshold - loaded.threshold) < 1e-9


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------

class TestAutoencoder:
    def test_compute_spectrum(self):
        window = _make_healthy()
        spec = AutoencoderDetector.compute_fft_spectrum(window, n_bins=512)
        assert spec.shape == (512,)
        assert spec.min() >= 0.0
        assert spec.max() <= 1.0 + 1e-6

    def test_fit_and_score(self):
        # Quick smoke-test with tiny epochs
        windows = [_make_healthy() for _ in range(20)]
        det = AutoencoderDetector(n_fft_bins=256, epochs=3, batch_size=8)
        det.fit(features=[], raw_windows=windows)
        assert det._is_fitted
        feat = det.prepare_feature(windows[0])
        score = det.score(feat)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_save_load(self, tmp_path):
        windows = [_make_healthy() for _ in range(10)]
        det = AutoencoderDetector(n_fft_bins=256, epochs=2, batch_size=4)
        det.fit(features=[], raw_windows=windows)
        path = str(tmp_path / "ae.pt")
        det.save(path)
        loaded = AutoencoderDetector.load(path)
        assert loaded._is_fitted
        feat  = det.prepare_feature(windows[0])
        s1    = det.score(feat)
        s2    = loaded.score(feat)
        assert abs(s1 - s2) < 1e-5
