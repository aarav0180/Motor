"""
detectors/threshold.py — Simple RMS threshold detector (project baseline).

The simplest possible anomaly detector: flag a window if its RMS exceeds
a threshold learned from healthy training data.  Used as a baseline to
compare against more sophisticated methods (Isolation Forest, Autoencoder).
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .base import BaseDetector, LABEL_NORMAL, LABEL_ANOMALY

logger = logging.getLogger(__name__)


class RMSThresholdDetector(BaseDetector):
    """
    One-class anomaly detector based solely on the RMS feature.

    Training
    --------
    Computes mean and std of RMS across healthy windows.
    Threshold = mean + sigma * std  (default sigma = 3.0).

    Anomaly score
    -------------
    Normalised distance from the healthy RMS mean:
        score = (rms - mean_healthy) / std_healthy
    Score ≥ threshold → anomaly.

    Parameters
    ----------
    sigma : float
        Standard deviations above the healthy mean that define the threshold.
        Stored so it can be reapplied if training data changes.
    threshold : float, optional
        Override auto-computed threshold (useful for loading from disk).
    """

    def __init__(self, sigma: float = 3.0, threshold: Optional[float] = None) -> None:
        super().__init__(threshold=threshold)
        self.sigma = sigma
        self._mean_rms: float = 0.0
        self._std_rms:  float = 1.0

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def fit(self, features: List[Dict[str, float]]) -> "RMSThresholdDetector":
        """
        Learn healthy RMS statistics.

        Parameters
        ----------
        features : list of dict
            Each dict must contain the key 'rms'.
        """
        if not features:
            raise ValueError("features list must not be empty")

        rms_values = np.array([f["rms"] for f in features], dtype=np.float64)
        self._mean_rms = float(np.mean(rms_values))
        self._std_rms  = float(np.std(rms_values)) or 1e-6  # guard zero-std

        if self.threshold is None:
            self.threshold = self._mean_rms + self.sigma * self._std_rms

        self._is_fitted = True
        logger.info(
            "RMSThresholdDetector fitted — mean_rms=%.4f  std_rms=%.4f  "
            "threshold=%.4f (sigma=%.1f)",
            self._mean_rms, self._std_rms, self.threshold, self.sigma,
        )
        return self

    def score(self, feature: Dict[str, float]) -> float:
        """
        Return normalised z-score of the window's RMS.

        Score = (rms - mean_healthy) / std_healthy.
        Values > sigma suggest anomaly.
        """
        self._require_fitted()
        rms = feature["rms"]
        return (rms - self._mean_rms) / self._std_rms

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mean_rms":  self._mean_rms,
            "std_rms":   self._std_rms,
            "sigma":     self.sigma,
            "threshold": self.threshold,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        logger.info("RMSThresholdDetector saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "RMSThresholdDetector":
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        obj = cls(sigma=payload["sigma"], threshold=payload["threshold"])
        obj._mean_rms  = payload["mean_rms"]
        obj._std_rms   = payload["std_rms"]
        obj._is_fitted = True
        logger.info("RMSThresholdDetector loaded from %s", path)
        return obj
