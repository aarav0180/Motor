"""
detectors/isolation_forest.py — Isolation Forest anomaly detector.

Trained on healthy feature vectors only (one-class learning).
Uses scikit-learn's IsolationForest under the hood, with a StandardScaler
preprocessing step baked in so the pipeline never needs to scale manually.

Anomaly score is converted to [0, 1] range (1 = most anomalous) using
the raw decision_function output, then thresholded.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .base import BaseDetector

logger = logging.getLogger(__name__)


class IsolationForestDetector(BaseDetector):
    """
    One-class anomaly detector using scikit-learn's Isolation Forest.

    Parameters
    ----------
    contamination : float
        Expected fraction of anomalies in the training set.
        Use a small value like 0.05 when training on healthy-only data.
    n_estimators : int
        Number of base estimators in the ensemble.
    random_state : int
        Reproducibility seed.
    threshold : float, optional
        Override the auto-computed anomaly threshold.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
        threshold: Optional[float] = None,
    ) -> None:
        super().__init__(threshold=threshold)
        self.contamination = contamination
        self.n_estimators  = n_estimators
        self.random_state  = random_state

        self._scaler = None   # fitted sklearn StandardScaler
        self._clf    = None   # fitted sklearn IsolationForest
        self._feature_keys: List[str] = []

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def fit(self, features: List[Dict[str, float]]) -> "IsolationForestDetector":
        """
        Fit scaler + Isolation Forest on healthy feature vectors.

        Parameters
        ----------
        features : list of dict
            Healthy training windows as feature dicts.
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        X, keys = self._dicts_to_matrix(features)
        self._feature_keys = keys

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._clf = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self._clf.fit(X_scaled)
        self._is_fitted = True

        # Auto-threshold from healthy validation scores if not provided
        if self.threshold is None:
            self.set_threshold_from_healthy(features, sigma=3.0)

        logger.info(
            "IsolationForestDetector fitted — %d samples, %d features, "
            "threshold=%.4f",
            len(features), len(keys), self.threshold,
        )
        return self

    def score(self, feature: Dict[str, float]) -> float:
        """
        Return anomaly score in [0, ∞).

        Internally, sklearn's decision_function returns negative values for
        anomalies.  We negate and shift so that 0 = perfectly normal and
        higher = more anomalous.
        """
        self._require_fitted()
        x = np.array([[feature[k] for k in self._feature_keys]], dtype=np.float32)
        x_scaled = self._scaler.transform(x)
        raw = float(self._clf.decision_function(x_scaled)[0])
        # decision_function: more negative → more anomalous
        # We flip sign so higher → more anomalous
        return -raw

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scaler":       self._scaler,
            "clf":          self._clf,
            "feature_keys": self._feature_keys,
            "threshold":    self.threshold,
            "contamination":self.contamination,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        logger.info("IsolationForestDetector saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "IsolationForestDetector":
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        obj = cls(
            contamination=payload["contamination"],
            n_estimators=payload["n_estimators"],
            random_state=payload["random_state"],
            threshold=payload["threshold"],
        )
        obj._scaler       = payload["scaler"]
        obj._clf          = payload["clf"]
        obj._feature_keys = payload["feature_keys"]
        obj._is_fitted    = True
        logger.info("IsolationForestDetector loaded from %s", path)
        return obj
