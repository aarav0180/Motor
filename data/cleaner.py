"""
data/cleaner.py  —  Signal cleaning and pre-processing.

Applied to every raw vibration window (from either CWRU .mat files or the
live physical sensor) before feature extraction.

Cleaning steps (all optional, controlled by flags):
  1. DC removal       — subtract the signal mean (remove sensor offset)
  2. Detrending       — remove any linear drift (scipy.signal.detrend)
  3. Bandpass filter  — keep only vibration-relevant frequencies
  4. Outlier clipping — clip samples beyond k standard deviations
  5. Normalisation    — scale to [-1, 1] or unit-RMS

Each step is a pure function so it can be unit-tested and swapped out
independently.  `SignalCleaner` chains them in order using a config dict.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step functions  (pure, stateless)
# ---------------------------------------------------------------------------

def remove_dc(signal: np.ndarray) -> np.ndarray:
    """Subtract the signal mean to remove sensor DC offset."""
    return signal - np.mean(signal)


def detrend_signal(signal: np.ndarray) -> np.ndarray:
    """Remove linear trend (slow drift) using scipy.signal.detrend."""
    return scipy.signal.detrend(signal, type="linear").astype(np.float32)


def bandpass_filter(
    signal: np.ndarray,
    sample_rate: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter.

    Parameters
    ----------
    low_hz, high_hz : float
        Lower and upper cutoff frequencies in Hz.
        Typical range for vibration: 10 Hz – 500 Hz (at 1 kHz rate)
                                     10 Hz – 5 000 Hz (at 12 kHz CWRU rate)
    order : int
        Filter order (higher = steeper roll-off, more phase distortion).
    """
    nyq  = sample_rate / 2.0
    lo   = max(low_hz  / nyq, 1e-4)
    hi   = min(high_hz / nyq, 1.0 - 1e-4)
    if lo >= hi:
        logger.warning(
            "bandpass: low_hz (%.1f) >= high_hz (%.1f) — skipping filter",
            low_hz, high_hz,
        )
        return signal
    b, a = scipy.signal.butter(order, [lo, hi], btype="band")
    return scipy.signal.filtfilt(b, a, signal).astype(np.float32)


def clip_outliers(signal: np.ndarray, k: float = 5.0) -> np.ndarray:
    """
    Clip samples that exceed k standard deviations from the mean.

    Extreme spikes (electrical noise, sensor glitches) can dominate kurtosis
    and FFT; this removes them before feature extraction.
    """
    mu, std = np.mean(signal), np.std(signal)
    if std == 0:
        return signal
    return np.clip(signal, mu - k * std, mu + k * std).astype(np.float32)


def normalise_unit_rms(signal: np.ndarray) -> np.ndarray:
    """
    Scale signal to unit RMS.

    Useful when comparing signals from different sensors or gain settings.
    Does NOT change the shape of the signal, only its amplitude.
    """
    rms = np.sqrt(np.mean(signal ** 2))
    if rms < 1e-10:
        return signal
    return (signal / rms).astype(np.float32)


def normalise_minmax(signal: np.ndarray) -> np.ndarray:
    """Scale signal to [-1, 1]."""
    mn, mx = signal.min(), signal.max()
    span   = mx - mn
    if span < 1e-10:
        return np.zeros_like(signal)
    return (2.0 * (signal - mn) / span - 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class CleanerConfig:
    """
    Controls which cleaning steps are applied and with what parameters.

    Defaults are tuned for the CWRU DE channel at 12 kHz.
    For the ADXL345 at 1 kHz, adjust bandpass to (5.0, 450.0).
    """
    remove_dc:       bool  = True
    detrend:         bool  = True
    bandpass:        bool  = True
    bandpass_low_hz: float = 10.0
    bandpass_high_hz:float = 5_500.0  # well below Nyquist at 12 kHz
    bandpass_order:  int   = 4
    clip_outliers:   bool  = True
    clip_k:          float = 5.0
    normalise:       str   = "unit_rms"   # "unit_rms" | "minmax" | "none"

    @classmethod
    def for_adxl345(cls) -> "CleanerConfig":
        """Preset for the ADXL345 at ~1 kHz sample rate."""
        return cls(
            bandpass_low_hz=5.0,
            bandpass_high_hz=450.0,
        )

    @classmethod
    def for_cwru(cls) -> "CleanerConfig":
        """Preset for the CWRU DE channel at 12 kHz."""
        return cls(
            bandpass_low_hz=10.0,
            bandpass_high_hz=5_500.0,
        )


# ---------------------------------------------------------------------------
# Stateless convenience function
# ---------------------------------------------------------------------------

def clean_signal(
    signal: np.ndarray,
    sample_rate: float,
    cfg: Optional[CleanerConfig] = None,
) -> np.ndarray:
    """
    Apply the full cleaning pipeline to a 1-D signal array.

    Parameters
    ----------
    signal : np.ndarray  shape (N,)
    sample_rate : float  in Hz
    cfg : CleanerConfig  (uses defaults if None)

    Returns
    -------
    np.ndarray  cleaned signal, same length as input
    """
    if cfg is None:
        cfg = CleanerConfig()

    s = signal.astype(np.float32).copy()

    if cfg.remove_dc:
        s = remove_dc(s)

    if cfg.detrend:
        s = detrend_signal(s)

    if cfg.bandpass:
        s = bandpass_filter(
            s, sample_rate,
            low_hz=cfg.bandpass_low_hz,
            high_hz=cfg.bandpass_high_hz,
            order=cfg.bandpass_order,
        )

    if cfg.clip_outliers:
        s = clip_outliers(s, k=cfg.clip_k)

    if cfg.normalise == "unit_rms":
        s = normalise_unit_rms(s)
    elif cfg.normalise == "minmax":
        s = normalise_minmax(s)

    return s


# ---------------------------------------------------------------------------
# Stateful class (convenient when cleaning many files with the same config)
# ---------------------------------------------------------------------------

class SignalCleaner:
    """
    Applies the same CleanerConfig to every signal it receives.

    Tracks basic statistics (mean RMS before/after) for diagnostics.

    Example
    -------
    >>> cleaner = SignalCleaner(CleanerConfig.for_cwru(), sample_rate=12000)
    >>> clean   = cleaner.clean(raw_signal)
    >>> batch   = cleaner.clean_batch(list_of_windows)
    """

    def __init__(
        self,
        config: Optional[CleanerConfig] = None,
        sample_rate: float = 12_000,
    ) -> None:
        self.config      = config or CleanerConfig()
        self.sample_rate = sample_rate
        self._n_cleaned  = 0

    def clean(self, signal: np.ndarray) -> np.ndarray:
        """Clean a single 1-D signal and return the result."""
        result = clean_signal(signal, self.sample_rate, self.config)
        self._n_cleaned += 1
        return result

    def clean_batch(self, signals: list[np.ndarray]) -> list[np.ndarray]:
        """Clean a list of windows."""
        return [self.clean(s) for s in signals]

    def diagnose(self, raw: np.ndarray) -> dict:
        """
        Return a dict of before/after statistics for one signal.
        Useful for debugging cleaning parameters.
        """
        cleaned = self.clean(raw)
        def stats(s):
            return {
                "rms":  float(np.sqrt(np.mean(s**2))),
                "peak": float(np.max(np.abs(s))),
                "std":  float(np.std(s)),
                "min":  float(s.min()),
                "max":  float(s.max()),
            }
        return {"before": stats(raw), "after": stats(cleaned)}

    @property
    def signals_cleaned(self) -> int:
        return self._n_cleaned

    def __repr__(self) -> str:
        return (
            f"SignalCleaner(rate={self.sample_rate} Hz, "
            f"cleaned={self._n_cleaned})"
        )
