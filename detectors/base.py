"""
detectors/base.py — Abstract base class for all anomaly detectors.

Each detector must implement:
  - fit(feature_dicts)   — learn from healthy training data
  - score(feature_dict)  — return a float anomaly score (higher = more anomalous)
  - predict(feature_dict)— return (score, label) where label ∈ {normal, anomaly}
  - save(path) / load(path) — serialisation

The pipeline only imports BaseDetector; concrete detector types are
swapped in via config or factory, keeping the service loop clean.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Canonical label strings
LABEL_NORMAL  = "normal"
LABEL_ANOMALY = "anomaly"


class BaseDetector(ABC):
    """
    Abstract anomaly detector.

    All detectors are trained on healthy-only data (one-class learning).
    After training, `predict()` returns a health score and a binary label.

    Parameters
    ----------
    threshold : float, optional
        Anomaly score above which a window is labelled 'anomaly'.
        If None, it must be set by `fit()` using validation data.
    """

    def __init__(self, threshold: Optional[float] = None) -> None:
        self.threshold = threshold
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, features: List[Dict[str, float]]) -> "BaseDetector":
        """
        Train the detector on healthy-only feature vectors.

        Parameters
        ----------
        features : list of dict
            Each dict is the output of FeatureVector.to_dict() for one window.

        Returns
        -------
        self  (for chaining)
        """

    @abstractmethod
    def score(self, feature: Dict[str, float]) -> float:
        """
        Compute an anomaly score for a single window.

        Higher values mean *more anomalous* (regardless of internal sign
        conventions — subclasses must normalise direction here).

        Parameters
        ----------
        feature : dict
            Output of FeatureVector.to_dict().

        Returns
        -------
        float
            Anomaly score in [0, ∞) or [0, 1] depending on subclass.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the trained detector to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseDetector":
        """Load a detector from disk and return an instance."""

    # ------------------------------------------------------------------
    # Shared concrete methods
    # ------------------------------------------------------------------

    def predict(self, feature: Dict[str, float]) -> Tuple[float, str]:
        """
        Return (anomaly_score, label).

        Uses `self.threshold`; raises if not set.
        """
        self._require_fitted()
        if self.threshold is None:
            raise RuntimeError(
                "threshold is not set.  Call fit() first or pass a threshold "
                "to the constructor."
            )
        s = self.score(feature)
        label = LABEL_ANOMALY if s >= self.threshold else LABEL_NORMAL
        return s, label

    def predict_batch(
        self, features: List[Dict[str, float]]
    ) -> List[Tuple[float, str]]:
        """Run predict() on each feature dict in the list."""
        return [self.predict(f) for f in features]

    def set_threshold_from_healthy(
        self,
        healthy_features: List[Dict[str, float]],
        sigma: float = 3.0,
    ) -> float:
        """
        Auto-set threshold as mean + sigma * std of scores on healthy data.

        Call this after fit() when you don't want to pick the threshold by hand.

        Parameters
        ----------
        healthy_features : list of dict
            Held-out healthy windows (should NOT overlap with training data).
        sigma : float
            Number of standard deviations above the mean (default 3).

        Returns
        -------
        float
            The computed threshold (also stored in self.threshold).
        """
        self._require_fitted()
        scores = [self.score(f) for f in healthy_features]
        mu, std = np.mean(scores), np.std(scores)
        self.threshold = float(mu + sigma * std)
        logger.info(
            "%s threshold set to %.4f (mu=%.4f, std=%.4f, sigma=%.1f)",
            self.__class__.__name__, self.threshold, mu, std, sigma,
        )
        return self.threshold

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted yet.  "
                "Call fit() before predict() or score()."
            )

    @staticmethod
    def _dicts_to_matrix(
        features: List[Dict[str, float]],
        keys: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Convert a list of feature dicts to a 2-D numpy matrix.

        Returns (matrix, ordered_key_list).  If `keys` is None, keys are
        inferred from the first dict (and must be consistent across all dicts).
        """
        if not features:
            raise ValueError("features list is empty")
        if keys is None:
            keys = [k for k in features[0] if isinstance(features[0][k], (int, float))]
        matrix = np.array([[f[k] for k in keys] for f in features], dtype=np.float32)
        return matrix, keys

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"{self.__class__.__name__}("
            f"threshold={self.threshold}, status={status})"
        )
