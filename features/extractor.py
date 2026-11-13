"""
features/extractor.py — Signal feature extraction from raw vibration windows.

All maths lives here.  Callers receive a plain dict of floats that can be
dropped straight into a pandas DataFrame row or passed to any detector.

Design decisions
----------------
- Pure functions + a thin class wrapper so both styles are usable.
- `extract_features()` is the single entry-point; it calls private helpers.
- FFT band energies are parameterised by running frequency so the same
  extractor works at every motor speed without code changes.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import scipy.signal
import scipy.stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data-class for clean, typed feature vectors
# ---------------------------------------------------------------------------

@dataclass
class FeatureVector:
    """
    Typed container for one window's worth of features.

    Convert to dict with `fv.to_dict()` or unpack directly into pandas.
    """
    # Time-domain statistics
    rms:          float = 0.0
    peak:         float = 0.0
    crest_factor: float = 0.0
    kurtosis:     float = 0.0
    skewness:     float = 0.0
    std:          float = 0.0

    # Frequency-domain
    band_energies: List[float] = field(default_factory=list)
    spectral_entropy: float = 0.0
    dominant_freq:    float = 0.0  # Hz of the highest PSD peak

    # Metadata (not used as ML features)
    sample_rate:  float = 0.0
    n_samples:    int   = 0
    run_freq_hz:  float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Flat dict suitable for a DataFrame row or JSON payload."""
        d: Dict[str, float] = {
            "rms":           self.rms,
            "peak":          self.peak,
            "crest_factor":  self.crest_factor,
            "kurtosis":      self.kurtosis,
            "skewness":      self.skewness,
            "std":           self.std,
            "spectral_entropy": self.spectral_entropy,
            "dominant_freq": self.dominant_freq,
        }
        for i, e in enumerate(self.band_energies, start=1):
            d[f"band{i}_energy"] = e
        return d


# ---------------------------------------------------------------------------
# Private signal-processing helpers
# ---------------------------------------------------------------------------

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def _peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)))


def _crest_factor(x: np.ndarray) -> float:
    r = _rms(x)
    return float(_peak(x) / r) if r > 0 else 0.0


def _band_energy(
    psd: np.ndarray,
    freqs: np.ndarray,
    center_hz: float,
    bandwidth_hz: float = 5.0,
) -> float:
    """
    Integrate PSD power within [center - bw/2, center + bw/2].

    Parameters
    ----------
    psd : np.ndarray
        Power spectral density values (from scipy.signal.welch).
    freqs : np.ndarray
        Corresponding frequency bins.
    center_hz : float
        Centre frequency of the band.
    bandwidth_hz : float
        Full width of the band in Hz.

    Returns
    -------
    float
        Sum of PSD values in the band (proxy for band energy).
    """
    half = bandwidth_hz / 2.0
    mask = (freqs >= center_hz - half) & (freqs <= center_hz + half)
    return float(np.sum(psd[mask]))


def _spectral_entropy(psd: np.ndarray) -> float:
    """
    Shannon entropy of the normalised power spectral density.

    Low entropy → energy concentrated in a few frequencies (machinery tone).
    High entropy → broadband / random noise.
    """
    psd_norm = psd / (psd.sum() + 1e-12)
    return float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))


def _dominant_freq(freqs: np.ndarray, psd: np.ndarray) -> float:
    """Return the frequency bin with the highest PSD value."""
    return float(freqs[np.argmax(psd)])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(
    window: np.ndarray,
    sample_rate: float,
    run_freq_hz: float,
    n_harmonics: int = 3,
    band_width_hz: float = 5.0,
    nperseg: Optional[int] = None,
) -> FeatureVector:
    """
    Extract time- and frequency-domain features from one vibration window.

    Parameters
    ----------
    window : np.ndarray
        1-D array of raw acceleration samples.
    sample_rate : float
        Sampling rate in Hz (actual achieved rate from the sensor).
    run_freq_hz : float
        Fundamental running frequency of the motor in Hz.
        Used to position FFT band-energy windows at 1×, 2×, … n_harmonics×.
    n_harmonics : int
        Number of harmonic bands to compute (default 3: 1×, 2×, 3× run freq).
    band_width_hz : float
        Full width of each harmonic band in Hz (default ±2.5 Hz).
    nperseg : int, optional
        Window length for Welch PSD estimate.  Defaults to min(256, len/4).

    Returns
    -------
    FeatureVector
    """
    if window.ndim != 1 or len(window) == 0:
        raise ValueError("window must be a non-empty 1-D array")

    # --- Time domain ---
    rms    = _rms(window)
    peak   = _peak(window)
    crest  = _crest_factor(window)
    kurt   = float(scipy.stats.kurtosis(window))
    skew   = float(scipy.stats.skew(window))
    std    = float(np.std(window))

    # --- Frequency domain ---
    _nperseg = nperseg or min(256, len(window) // 4)
    freqs, psd = scipy.signal.welch(window, fs=sample_rate, nperseg=_nperseg)

    band_energies = [
        _band_energy(psd, freqs, run_freq_hz * h, band_width_hz)
        for h in range(1, n_harmonics + 1)
    ]
    sp_entropy    = _spectral_entropy(psd)
    dom_freq      = _dominant_freq(freqs, psd)

    return FeatureVector(
        rms=rms,
        peak=peak,
        crest_factor=crest,
        kurtosis=kurt,
        skewness=skew,
        std=std,
        band_energies=band_energies,
        spectral_entropy=sp_entropy,
        dominant_freq=dom_freq,
        sample_rate=sample_rate,
        n_samples=len(window),
        run_freq_hz=run_freq_hz,
    )


class FeatureExtractor:
    """
    Stateful wrapper around `extract_features` that remembers common
    parameters so the pipeline only needs to call `extractor.transform(window)`.

    Example
    -------
    >>> fx = FeatureExtractor(sample_rate=1000, run_freq_hz=40.0)
    >>> fv = fx.transform(window)
    >>> row = fv.to_dict()
    """

    def __init__(
        self,
        sample_rate: float,
        run_freq_hz: float,
        n_harmonics: int = 3,
        band_width_hz: float = 5.0,
    ) -> None:
        self.sample_rate   = sample_rate
        self.run_freq_hz   = run_freq_hz
        self.n_harmonics   = n_harmonics
        self.band_width_hz = band_width_hz

    def transform(self, window: np.ndarray) -> FeatureVector:
        """Extract features from a single window using stored parameters."""
        return extract_features(
            window,
            sample_rate=self.sample_rate,
            run_freq_hz=self.run_freq_hz,
            n_harmonics=self.n_harmonics,
            band_width_hz=self.band_width_hz,
        )

    def transform_batch(self, windows: List[np.ndarray]) -> List[FeatureVector]:
        """Extract features from a list of windows."""
        return [self.transform(w) for w in windows]

    def __repr__(self) -> str:
        return (
            f"FeatureExtractor(sample_rate={self.sample_rate}, "
            f"run_freq_hz={self.run_freq_hz}, "
            f"n_harmonics={self.n_harmonics})"
        )
