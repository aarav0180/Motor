"""
detectors/edge_models.py — Production edge inference wrappers for TorchScript models.

Wraps the pre-trained CWRU edge models:
  1. CWRUUnsupervisedDetector  : One-class 1D-CNN Autoencoder trained on healthy data.
                                 Computes waveform reconstruction error (MSE).
  2. CWRUAdvancedClassifierDetector: 4-class supervised 1D-CNN classifier diagnosing:
                                 Healthy, Ball Fault, Inner Race Fault, Outer Race Fault.
  3. CWRUHybridDetector        : Two-stage detector combining unsupervised anomaly detection
                                 with deep fault diagnosis when an anomaly occurs.

Both models execute in TorchScript on CPU in <1 ms latency (<5 ms on Raspberry Pi).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .base import BaseDetector, LABEL_NORMAL, LABEL_ANOMALY

logger = logging.getLogger(__name__)

# Canonical fault classes for CWRU Advanced 1D-CNN
CWRU_CLASSES: Dict[int, str] = {
    0: "healthy",
    1: "ball_fault",
    2: "inner_race_fault",
    3: "outer_race_fault",
}


def _normalize_signal(sig: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance signal normalization."""
    arr = np.asarray(sig, dtype=np.float32).flatten()
    std = float(np.std(arr))
    return (arr - float(np.mean(arr))) / (std + 1e-8)


# ---------------------------------------------------------------------------
# 1. Unsupervised Autoencoder (Trained strictly on healthy baseline data)
# ---------------------------------------------------------------------------

class CWRUUnsupervisedDetector(BaseDetector):
    """
    TorchScript 1D-CNN Autoencoder for vibration anomaly detection.
    Trained strictly on healthy operational data.
    Anomaly score is the mean squared reconstruction error (MSE).
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "models/cwru_unsupervised_xiong_edge.pt",
        threshold: float = 0.05,
    ) -> None:
        super().__init__(threshold=threshold)
        self.model_path = Path(model_path)
        self._model = None
        self._load_model()
        self._is_fitted = True

    def _load_model(self) -> None:
        import torch
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self._model = torch.jit.load(str(self.model_path), map_location="cpu")
        self._model.eval()
        logger.info("Loaded TorchScript Unsupervised Autoencoder from %s", self.model_path)

    def fit(self, features: List[Dict[str, float]]) -> "CWRUUnsupervisedDetector":
        """Pre-trained model; calibrate threshold if healthy features are provided."""
        scores = []
        for f in features:
            if "raw_signal" in f:
                scores.append(self.score(f))
        if scores:
            mu, std = float(np.mean(scores)), float(np.std(scores))
            self.threshold = float(mu + 3.5 * std)
            logger.info("Calibrated threshold from %d samples: %.6f", len(scores), self.threshold)
        self._is_fitted = True
        return self

    def prepare_feature(self, raw_window: np.ndarray) -> Dict[str, Any]:
        """Attach raw waveform to feature dict for deep learning inference."""
        return {"raw_signal": np.asarray(raw_window, dtype=np.float32)}

    def score(self, feature: Dict[str, Any]) -> float:
        """Calculate reconstruction MSE."""
        import torch
        if "raw_signal" not in feature:
            raise KeyError("Feature dict missing 'raw_signal'. Use prepare_feature(window).")

        raw = feature["raw_signal"]
        norm_sig = _normalize_signal(raw)

        # Truncate or use multiple of 1000 for perfect conv transpose alignment
        L = len(norm_sig)
        if L % 1000 != 0 and L > 1000:
            norm_sig = norm_sig[: (L // 1000) * 1000]

        x = torch.from_numpy(norm_sig).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            recon = self._model(x)
            mse = float(torch.mean((x - recon) ** 2).item())

        return float(round(mse, 6))

    def predict(self, feature: Dict[str, Any]) -> Tuple[float, str]:
        s = self.score(feature)
        label = LABEL_ANOMALY if s > self.threshold else LABEL_NORMAL
        return s, label

    def save(self, path: str) -> None:
        pass  # Model is managed as TorchScript .pt

    @classmethod
    def load(cls, path: str, threshold: float = 0.05) -> "CWRUUnsupervisedDetector":
        return cls(model_path=path, threshold=threshold)


# ---------------------------------------------------------------------------
# 2. Supervised 1D-CNN Fault Classifier (Diagnoses specific defect types)
# ---------------------------------------------------------------------------

class CWRUAdvancedClassifierDetector(BaseDetector):
    """
    TorchScript 1D-CNN Multi-Class Classifier.
    Diagnoses healthy vibration vs specific bearing defects:
      - 0: healthy
      - 1: ball_fault
      - 2: inner_race_fault
      - 3: outer_race_fault
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "models/cwru_advanced_edge.pt",
        threshold: float = 0.50,
    ) -> None:
        super().__init__(threshold=threshold)
        self.model_path = Path(model_path)
        self._model = None
        self._load_model()
        self._is_fitted = True

    def _load_model(self) -> None:
        import torch
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self._model = torch.jit.load(str(self.model_path), map_location="cpu")
        self._model.eval()
        logger.info("Loaded TorchScript Advanced Classifier from %s", self.model_path)

    def fit(self, features: List[Dict[str, float]]) -> "CWRUAdvancedClassifierDetector":
        self._is_fitted = True
        return self

    def prepare_feature(self, raw_window: np.ndarray) -> Dict[str, Any]:
        return {"raw_signal": np.asarray(raw_window, dtype=np.float32)}

    def diagnose(self, feature: Dict[str, Any]) -> Dict[str, Any]:
        """Return full diagnosis probabilities and predicted defect class."""
        import torch
        if "raw_signal" not in feature:
            raise KeyError("Feature dict missing 'raw_signal'. Use prepare_feature(window).")

        raw = feature["raw_signal"]
        norm_sig = _normalize_signal(raw)
        x = torch.from_numpy(norm_sig).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            logits = self._model(x)
            probs = torch.softmax(logits, dim=1).numpy().flatten()
            pred_class = int(np.argmax(probs))

        return {
            "class_id": pred_class,
            "label": CWRU_CLASSES.get(pred_class, "unknown"),
            "confidence": float(round(probs[pred_class], 4)),
            "anomaly_prob": float(round(1.0 - probs[0], 4)),
            "probabilities": {CWRU_CLASSES[i]: float(round(p, 4)) for i, p in enumerate(probs)},
        }

    def score(self, feature: Dict[str, Any]) -> float:
        """Anomaly probability (1.0 - P(healthy))."""
        res = self.diagnose(feature)
        return res["anomaly_prob"]

    def predict(self, feature: Dict[str, Any]) -> Tuple[float, str]:
        diag = self.diagnose(feature)
        score = diag["anomaly_prob"]
        label = LABEL_NORMAL if diag["class_id"] == 0 else LABEL_ANOMALY
        return score, label

    def save(self, path: str) -> None:
        pass

    @classmethod
    def load(cls, path: str, threshold: float = 0.50) -> "CWRUAdvancedClassifierDetector":
        return cls(model_path=path, threshold=threshold)


# ---------------------------------------------------------------------------
# 3. Hybrid Two-Stage Edge Detector (Unsupervised Gate + Supervised Diagnosis)
# ---------------------------------------------------------------------------

class CWRUHybridDetector(BaseDetector):
    """
    Two-stage production edge detector:
      - Stage 1: Unsupervised Autoencoder checks for anomalies. If healthy, exit immediately.
      - Stage 2: If anomaly flagged, run Advanced Classifier to pinpoint the exact defect.
    """

    def __init__(
        self,
        unsupervised_path: Union[str, Path] = "models/cwru_unsupervised_xiong_edge.pt",
        classifier_path: Union[str, Path] = "models/cwru_advanced_edge.pt",
        threshold: float = 0.05,
    ) -> None:
        super().__init__(threshold=threshold)
        self.unsupervised = CWRUUnsupervisedDetector(model_path=unsupervised_path, threshold=threshold)
        self.classifier = CWRUAdvancedClassifierDetector(model_path=classifier_path)
        self._is_fitted = True

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, val: float) -> None:
        self._threshold = val
        if hasattr(self, "unsupervised"):
            self.unsupervised.threshold = val

    def fit(self, features: List[Dict[str, float]]) -> "CWRUHybridDetector":
        self.unsupervised.fit(features)
        self._is_fitted = True
        return self

    def prepare_feature(self, raw_window: np.ndarray) -> Dict[str, Any]:
        return {"raw_signal": np.asarray(raw_window, dtype=np.float32)}

    def score(self, feature: Dict[str, Any]) -> float:
        return self.unsupervised.score(feature)

    def predict(self, feature: Dict[str, Any]) -> Tuple[float, str]:
        mse, label = self.unsupervised.predict(feature)
        if label == LABEL_ANOMALY:
            diag = self.classifier.diagnose(feature)
            # Annotate feature with deep diagnosis for MQTT alerts
            feature["diagnosis"] = diag["label"]
            feature["diagnosis_confidence"] = diag["confidence"]
        else:
            feature["diagnosis"] = "healthy"
            feature["diagnosis_confidence"] = 1.0 - mse
        return mse, label

    def save(self, path: str) -> None:
        pass

    @classmethod
    def load(
        cls,
        unsupervised_path: str = "models/cwru_unsupervised_xiong_edge.pt",
        classifier_path: str = "models/cwru_advanced_edge.pt",
        threshold: float = 0.05,
    ) -> "CWRUHybridDetector":
        return cls(
            unsupervised_path=unsupervised_path,
            classifier_path=classifier_path,
            threshold=threshold,
        )
