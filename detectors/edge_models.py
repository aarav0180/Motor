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
import scipy.stats

from .base import BaseDetector, LABEL_NORMAL, LABEL_ANOMALY

logger = logging.getLogger(__name__)

# Canonical fault classes for legacy CWRU Advanced 1D-CNN
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


def extract_statistical_features(raw_window: np.ndarray) -> np.ndarray:
    """
    Extract the 9 statistical features expected by best_anomaly_model & best_multiclass_model:
    ['max', 'min', 'mean', 'sd', 'rms', 'skewness', 'kurtosis', 'crest', 'form']
    """
    sig = np.asarray(raw_window, dtype=np.float64).flatten()
    if len(sig) == 0:
        return np.zeros((1, 9), dtype=np.float64)
    mx = float(np.max(sig))
    mn = float(np.min(sig))
    mean = float(np.mean(sig))
    sd = float(np.std(sig))
    rms = float(np.sqrt(np.mean(sig ** 2)))
    skew = float(scipy.stats.skew(sig))
    kurt = float(scipy.stats.kurtosis(sig))
    crest = float(np.max(np.abs(sig)) / (rms + 1e-8))
    form = float(rms / (np.mean(np.abs(sig)) + 1e-8))
    return np.array([mx, mn, mean, sd, rms, skew, kurt, crest, form], dtype=np.float64).reshape(1, -1)


# ---------------------------------------------------------------------------
# 1. Anomaly Detector (Tells strictly if motor is healthy or if it's not)
# ---------------------------------------------------------------------------

class CWRUUnsupervisedDetector(BaseDetector):
    """
    Vibration Anomaly Detector.
    Supports:
      1. Best Anomaly Model (Isolation Forest with StandardScaler checkpoint)
      2. TorchScript 1D-CNN Autoencoder (MSE reconstruction)

    Only tells whether the motor is healthy or anomalous (no defect breakdown).
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "models/best_anomaly_model.pt",
        threshold: float = 0.08,
    ) -> None:
        super().__init__(threshold=threshold)
        self.model_path = Path(model_path)
        self._is_dict_model = False
        self._clf = None
        self._scaler = None
        self._model = None
        self._load_model()
        self._is_fitted = True

    def _load_model(self) -> None:
        import torch
        import warnings
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # 1. Try loading as dictionary checkpoint (best_anomaly_model.pt)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                checkpoint = torch.load(str(self.model_path), map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "model" in checkpoint and "scaler" in checkpoint:
                self._is_dict_model = True
                self._clf = checkpoint["model"]
                self._scaler = checkpoint["scaler"]
                self._feature_names = checkpoint.get("feature_names", [])
                logger.info("Loaded Isolation Forest Anomaly model from %s", self.model_path)
                return
        except Exception as exc:
            logger.debug("Checkpoint dict load skipped (%s), trying TorchScript...", exc)

        # 2. Fallback to TorchScript Autoencoder
        self._is_dict_model = False
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
        """Attach raw waveform to feature dict for inference."""
        return {"raw_signal": np.asarray(raw_window, dtype=np.float32)}

    def score(self, feature: Dict[str, Any]) -> float:
        """Calculate anomaly score."""
        if "raw_signal" not in feature:
            raise KeyError("Feature dict missing 'raw_signal'. Use prepare_feature(window).")

        raw = feature["raw_signal"]

        # Isolation Forest scoring
        if self._is_dict_model:
            import warnings
            feats = extract_statistical_features(raw)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                scaled = self._scaler.transform(feats)
                dec = float(self._clf.decision_function(scaled)[0])
            # In IsolationForest, negative decision indicates anomaly
            score = float(round(max(0.0, -dec), 4))
            return score

        # TorchScript Autoencoder MSE
        import torch
        norm_sig = _normalize_signal(raw)
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
        # Anomaly model only tells if motor is healthy or if it's not
        feature["diagnosis"] = "healthy" if label == LABEL_NORMAL else "anomaly"
        feature["diagnosis_confidence"] = float(round(min(1.0, s / max(self.threshold, 1e-4)), 4)) if label == LABEL_ANOMALY else 0.95
        return s, label

    def save(self, path: str) -> None:
        pass

    @classmethod
    def load(cls, path: str, threshold: float = 0.08) -> "CWRUUnsupervisedDetector":
        return cls(model_path=path, threshold=threshold)


# ---------------------------------------------------------------------------
# 2. Multiclass Fault Classifier (Identifies fault AND exactly what the fault is)
# ---------------------------------------------------------------------------

class CWRUAdvancedClassifierDetector(BaseDetector):
    """
    Multiclass Fault Classifier.
    Diagnoses healthy operation vs specific bearing defects:
      - Healthy (Normal_1)
      - Ball Faults (Ball_007_1, Ball_014_1, Ball_021_1)
      - Inner Race Faults (IR_007_1, IR_014_1, IR_021_1)
      - Outer Race Faults (OR_007_6_1, OR_014_6_1, OR_021_6_1)
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "models/best_multiclass_model.pt",
        threshold: float = 0.50,
    ) -> None:
        super().__init__(threshold=threshold)
        self.model_path = Path(model_path)
        self._is_dict_model = False
        self._clf = None
        self._scaler = None
        self._label_encoder = None
        self._power_transformer = None
        self._use_power = False
        self._model = None
        self._load_model()
        self._is_fitted = True

    def _load_model(self) -> None:
        import torch
        import warnings
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # 1. Try loading as dictionary checkpoint (best_multiclass_model.pt)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                checkpoint = torch.load(str(self.model_path), map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "model" in checkpoint and "scaler" in checkpoint:
                self._is_dict_model = True
                self._clf = checkpoint["model"]
                self._scaler = checkpoint["scaler"]
                self._label_encoder = checkpoint.get("label_encoder")
                self._power_transformer = checkpoint.get("power_transformer")
                self._use_power = checkpoint.get("use_power_transformer", False)
                logger.info("Loaded Bagging Multiclass Classifier from %s", self.model_path)
                return
        except Exception as exc:
            logger.debug("Checkpoint dict load skipped (%s), trying TorchScript...", exc)

        # 2. Fallback to TorchScript Classifier
        self._is_dict_model = False
        self._model = torch.jit.load(str(self.model_path), map_location="cpu")
        self._model.eval()
        logger.info("Loaded TorchScript Advanced Classifier from %s", self.model_path)

    def fit(self, features: List[Dict[str, float]]) -> "CWRUAdvancedClassifierDetector":
        self._is_fitted = True
        return self

    def prepare_feature(self, raw_window: np.ndarray) -> Dict[str, Any]:
        return {"raw_signal": np.asarray(raw_window, dtype=np.float32)}

    def diagnose(self, feature: Dict[str, Any]) -> Dict[str, Any]:
        """Return full diagnosis probabilities and pinpoint exact defect type."""
        if "raw_signal" not in feature:
            raise KeyError("Feature dict missing 'raw_signal'. Use prepare_feature(window).")

        raw = feature["raw_signal"]

        # Bagging Multiclass Model diagnosis
        if self._is_dict_model:
            import warnings
            feats = extract_statistical_features(raw)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                scaled = self._scaler.transform(feats)
                if self._use_power and self._power_transformer is not None:
                    scaled = self._power_transformer.transform(scaled)

                probs = self._clf.predict_proba(scaled)[0]
            pred_idx = int(np.argmax(probs))
            raw_cls = str(self._label_encoder.inverse_transform([pred_idx])[0])
            confidence = float(round(probs[pred_idx], 4))

            classes = list(self._label_encoder.classes_)
            if "Normal_1" in classes:
                norm_idx = classes.index("Normal_1")
                anomaly_prob = float(round(1.0 - probs[norm_idx], 4))
            else:
                anomaly_prob = 0.0 if raw_cls.lower() == "normal" else 1.0

            # Pinpoint exact defect name
            if raw_cls == "Normal_1":
                diag_label = "healthy"
                is_fault = False
            else:
                is_fault = True
                if raw_cls.startswith("Ball_"):
                    depth = raw_cls.split("_")[1]
                    diag_label = f"ball_fault ({depth} mil)"
                elif raw_cls.startswith("IR_"):
                    depth = raw_cls.split("_")[1]
                    diag_label = f"inner_race_fault ({depth} mil)"
                elif raw_cls.startswith("OR_"):
                    parts = raw_cls.split("_")
                    diag_label = f"outer_race_fault ({parts[1]} mil)"
                else:
                    diag_label = raw_cls.lower()

            prob_dict = {
                str(c): float(round(p, 4))
                for c, p in zip(classes, probs)
            }

            return {
                "class_id": pred_idx,
                "label": diag_label,
                "raw_class": raw_cls,
                "is_fault": is_fault,
                "confidence": confidence,
                "anomaly_prob": anomaly_prob,
                "probabilities": prob_dict,
            }

        # TorchScript fallback
        import torch
        norm_sig = _normalize_signal(raw)
        x = torch.from_numpy(norm_sig).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            logits = self._model(x)
            probs = torch.softmax(logits, dim=1).numpy().flatten()
            pred_class = int(np.argmax(probs))

        return {
            "class_id": pred_class,
            "label": CWRU_CLASSES.get(pred_class, "unknown"),
            "raw_class": CWRU_CLASSES.get(pred_class, "unknown"),
            "is_fault": (pred_class != 0),
            "confidence": float(round(probs[pred_class], 4)),
            "anomaly_prob": float(round(1.0 - probs[0], 4)) if len(probs) > 0 else 0.0,
            "probabilities": {CWRU_CLASSES.get(i, f"class_{i}"): float(round(p, 4)) for i, p in enumerate(probs)},
        }

    def score(self, feature: Dict[str, Any]) -> float:
        """Anomaly probability (1.0 - P(healthy))."""
        res = self.diagnose(feature)
        return res["anomaly_prob"]

    def predict(self, feature: Dict[str, Any]) -> Tuple[float, str]:
        diag = self.diagnose(feature)
        feature["diagnosis"] = diag["label"]
        feature["diagnosis_confidence"] = diag["confidence"]
        feature["raw_class"] = diag.get("raw_class", diag["label"])
        score = diag["anomaly_prob"]
        is_fault = diag.get("is_fault", diag["class_id"] != 0)
        label = LABEL_ANOMALY if is_fault else LABEL_NORMAL
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
      - Stage 1: Anomaly Detector checks for anomalies. If healthy, exit immediately.
      - Stage 2: If anomaly flagged, run Multiclass Classifier to pinpoint the exact defect.
    """

    def __init__(
        self,
        unsupervised_path: Union[str, Path] = "models/best_anomaly_model.pt",
        classifier_path: Union[str, Path] = "models/best_multiclass_model.pt",
        threshold: float = 0.08,
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
        score, label = self.unsupervised.predict(feature)
        if label == LABEL_ANOMALY:
            diag = self.classifier.diagnose(feature)
            # Annotate feature with deep diagnosis for MQTT alerts
            feature["diagnosis"] = diag["label"]
            feature["diagnosis_confidence"] = diag["confidence"]
        else:
            feature["diagnosis"] = "healthy"
            feature["diagnosis_confidence"] = float(round(1.0 - min(1.0, score), 4))
        return score, label

    def save(self, path: str) -> None:
        pass

    @classmethod
    def load(
        cls,
        unsupervised_path: str = "models/best_anomaly_model.pt",
        classifier_path: str = "models/best_multiclass_model.pt",
        threshold: float = 0.08,
    ) -> "CWRUHybridDetector":
        return cls(
            unsupervised_path=unsupervised_path,
            classifier_path=classifier_path,
            threshold=threshold,
        )
