"""
tests/test_features.py — Unit tests for the feature extractor.

Run with:  pytest tests/
"""

import numpy as np
import pytest

from features.extractor import (
    FeatureExtractor,
    FeatureVector,
    extract_features,
    _rms,
    _peak,
    _crest_factor,
    _band_energy,
    _spectral_entropy,
)

SAMPLE_RATE = 1000
RUN_FREQ    = 40.0
WINDOW_SIZE = SAMPLE_RATE * 10   # 10 s


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sine_window():
    """Pure 40 Hz sine — healthy motor at known running frequency."""
    t = np.linspace(0, 10, WINDOW_SIZE, endpoint=False)
    return np.sin(2 * np.pi * RUN_FREQ * t).astype(np.float32)


@pytest.fixture
def noise_window():
    rng = np.random.default_rng(0)
    return rng.standard_normal(WINDOW_SIZE).astype(np.float32)


@pytest.fixture
def impulsive_window():
    """Mostly quiet with periodic impulses — simulates bearing fault."""
    w = np.zeros(WINDOW_SIZE, dtype=np.float32)
    for i in range(0, WINDOW_SIZE, 200):
        w[i] = 10.0
    return w


# ---------------------------------------------------------------------------
# Time-domain helpers
# ---------------------------------------------------------------------------

def test_rms_sine():
    """RMS of a unit sine = 1/sqrt(2)."""
    t = np.linspace(0, 10, 10000, endpoint=False)
    x = np.sin(2 * np.pi * 50 * t)
    assert abs(_rms(x) - 1 / np.sqrt(2)) < 1e-3


def test_rms_dc():
    x = np.ones(1000)
    assert _rms(x) == pytest.approx(1.0, abs=1e-6)


def test_peak_positive():
    x = np.array([-3.0, 1.0, 2.0, -1.0])
    assert _peak(x) == pytest.approx(3.0)


def test_crest_factor_sine():
    t = np.linspace(0, 1, 10000, endpoint=False)
    x = np.sin(2 * np.pi * 10 * t)
    # Crest factor of a sine = sqrt(2) ≈ 1.414
    assert abs(_crest_factor(x) - np.sqrt(2)) < 0.01


def test_crest_factor_zero_rms():
    x = np.zeros(100)
    assert _crest_factor(x) == 0.0


# ---------------------------------------------------------------------------
# Frequency-domain helpers
# ---------------------------------------------------------------------------

def test_band_energy_nonzero(sine_window):
    import scipy.signal
    freqs, psd = scipy.signal.welch(sine_window, fs=SAMPLE_RATE)
    energy = _band_energy(psd, freqs, center_hz=40.0, bandwidth_hz=5.0)
    assert energy > 0


def test_spectral_entropy_bounded(noise_window):
    import scipy.signal
    freqs, psd = scipy.signal.welch(noise_window, fs=SAMPLE_RATE)
    h = _spectral_entropy(psd)
    assert h >= 0


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------

def test_extract_features_returns_feature_vector(sine_window):
    fv = extract_features(sine_window, sample_rate=SAMPLE_RATE, run_freq_hz=RUN_FREQ)
    assert isinstance(fv, FeatureVector)
    assert fv.rms > 0
    assert fv.peak > 0
    assert fv.crest_factor > 0
    assert len(fv.band_energies) == 3   # default n_harmonics=3


def test_feature_vector_to_dict(sine_window):
    fv = extract_features(sine_window, sample_rate=SAMPLE_RATE, run_freq_hz=RUN_FREQ)
    d = fv.to_dict()
    assert isinstance(d, dict)
    for key in ("rms", "peak", "crest_factor", "kurtosis", "band1_energy"):
        assert key in d, f"Missing key: {key}"


def test_feature_extractor_class(sine_window):
    fx = FeatureExtractor(sample_rate=SAMPLE_RATE, run_freq_hz=RUN_FREQ)
    fv = fx.transform(sine_window)
    assert isinstance(fv, FeatureVector)


def test_batch_transform(sine_window, noise_window):
    fx      = FeatureExtractor(sample_rate=SAMPLE_RATE, run_freq_hz=RUN_FREQ)
    results = fx.transform_batch([sine_window, noise_window])
    assert len(results) == 2
    assert all(isinstance(r, FeatureVector) for r in results)


def test_impulsive_has_high_kurtosis(impulsive_window):
    fv = extract_features(impulsive_window, sample_rate=SAMPLE_RATE, run_freq_hz=RUN_FREQ)
    # Impulsive signals have kurtosis >> 3 (Gaussian kurtosis = 0 in excess form)
    assert fv.kurtosis > 5.0


def test_raises_on_empty_window():
    with pytest.raises(ValueError):
        extract_features(np.array([]), sample_rate=SAMPLE_RATE, run_freq_hz=RUN_FREQ)


def test_raises_on_2d_window():
    with pytest.raises(ValueError):
        extract_features(np.ones((10, 10)), sample_rate=SAMPLE_RATE, run_freq_hz=RUN_FREQ)
