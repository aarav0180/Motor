"""
tests/test_edge_models.py
-------------------------
Unit and integration tests for the TorchScript deep learning edge models:
  - CWRUUnsupervisedDetector (models/cwru_unsupervised_xiong_edge.pt)
  - CWRUAdvancedClassifierDetector (models/cwru_advanced_edge.pt)
  - CWRUHybridDetector (two-stage inference)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import json
import numpy as np
import pytest
import scipy.io

from detectors import (
    CWRUUnsupervisedDetector,
    CWRUAdvancedClassifierDetector,
    CWRUHybridDetector,
)
from service.monitor import build_detector, MonitorService


MODELS_DIR = Path("models")
UNSUP_PATH = MODELS_DIR / "cwru_unsupervised_xiong_edge.pt"
ADV_PATH = MODELS_DIR / "cwru_advanced_edge.pt"
BEST_ANOM_PATH = MODELS_DIR / "best_anomaly_model.pt"
BEST_MULTI_PATH = MODELS_DIR / "best_multiclass_model.pt"


def _load_cwru_sample(mat_name: str) -> np.ndarray:
    """Load a real CWRU signal from data/cwru/."""
    p = Path("data/cwru") / mat_name
    mat = scipy.io.loadmat(str(p))
    key = [k for k in mat.keys() if "DE_time" in k][0]
    return mat[key].flatten().astype(np.float32)


class TestCWRUUnsupervisedDetector:
    def test_init_and_load(self):
        det = CWRUUnsupervisedDetector(model_path=UNSUP_PATH, threshold=0.05)
        assert det._is_fitted is True
        assert det.threshold == 0.05

    def test_missing_model_raises(self):
        with pytest.raises(FileNotFoundError):
            CWRUUnsupervisedDetector(model_path="nonexistent.pt")

    def test_predict_real_cwru_healthy_vs_faulty(self):
        det = CWRUUnsupervisedDetector(model_path=UNSUP_PATH, threshold=0.05)

        # 1. Healthy: 97.mat
        sig_healthy = _load_cwru_sample("97.mat")[:12000]
        feat_h = det.prepare_feature(sig_healthy)
        score_h, label_h = det.predict(feat_h)

        assert score_h < 0.01
        assert label_h == "normal"

        # 2. Faulty: 105.mat (inner race)
        sig_fault = _load_cwru_sample("105.mat")[:12000]
        feat_f = det.prepare_feature(sig_fault)
        score_f, label_f = det.predict(feat_f)

        assert score_f > 0.30
        assert label_f == "anomaly"

    def test_dynamic_threshold_update(self):
        det = CWRUUnsupervisedDetector(model_path=UNSUP_PATH, threshold=0.05)
        sig_fault = _load_cwru_sample("105.mat")[:12000]
        feat = det.prepare_feature(sig_fault)

        # Initially flagged as anomaly (score ~0.61 > 0.05)
        score, label = det.predict(feat)
        assert label == "anomaly"

        # Dynamically raise threshold above score (simulating technician false-alarm suppression)
        det.threshold = score * 1.05
        new_score, new_label = det.predict(feat)
        assert new_label == "normal"

    def test_best_anomaly_model_healthy_vs_anomaly(self):
        if not BEST_ANOM_PATH.exists():
            pytest.skip("best_anomaly_model.pt not present")
        det = CWRUUnsupervisedDetector(model_path=BEST_ANOM_PATH, threshold=0.08)
        assert det._is_dict_model is True

        # Healthy: 97.mat -> score < 0.08, label normal, only tells healthy
        sig_h = _load_cwru_sample("97.mat")[:1000]
        feat_h = det.prepare_feature(sig_h)
        score_h, label_h = det.predict(feat_h)
        assert score_h < 0.08
        assert label_h == "normal"
        assert feat_h["diagnosis"] == "healthy"

        # Anomaly: 105.mat -> score > 0.08, label anomaly, only tells anomaly (no defect breakdown)
        sig_f = _load_cwru_sample("105.mat")[:1000]
        feat_f = det.prepare_feature(sig_f)
        score_f, label_f = det.predict(feat_f)
        assert score_f > 0.08
        assert label_f == "anomaly"
        assert feat_f["diagnosis"] == "anomaly"


class TestCWRUAdvancedClassifierDetector:
    def test_init_and_load(self):
        det = CWRUAdvancedClassifierDetector(model_path=ADV_PATH)
        assert det._is_fitted is True

    def test_diagnose_healthy_and_fault_classes(self):
        det = CWRUAdvancedClassifierDetector(model_path=ADV_PATH)

        # Healthy: 97.mat -> class 0 (healthy)
        sig_healthy = _load_cwru_sample("97.mat")[:12000]
        feat_h = det.prepare_feature(sig_healthy)
        diag_h = det.diagnose(feat_h)
        assert diag_h["class_id"] == 0
        assert diag_h["label"] == "healthy"
        assert diag_h["confidence"] > 0.95

        # Inner race fault: 105.mat -> class 2 (inner_race_fault)
        sig_ir = _load_cwru_sample("105.mat")[:12000]
        feat_ir = det.prepare_feature(sig_ir)
        diag_ir = det.diagnose(feat_ir)
        assert diag_ir["class_id"] == 2
        assert diag_ir["label"] == "inner_race_fault"
        assert diag_ir["confidence"] > 0.95

        # Outer race fault: 130.mat -> class 3 (outer_race_fault)
        sig_or = _load_cwru_sample("130.mat")[:12000]
        feat_or = det.prepare_feature(sig_or)
        diag_or = det.diagnose(feat_or)
        assert diag_or["class_id"] == 3
        assert diag_or["label"] == "outer_race_fault"
        assert diag_or["confidence"] > 0.95

    def test_predict_returns_binary_label(self):
        det = CWRUAdvancedClassifierDetector(model_path=ADV_PATH)
        sig_healthy = _load_cwru_sample("97.mat")[:12000]
        score, label = det.predict(det.prepare_feature(sig_healthy))
        assert label == "normal"

        sig_fault = _load_cwru_sample("105.mat")[:12000]
        score, label = det.predict(det.prepare_feature(sig_fault))
        assert label == "anomaly"

    def test_best_multiclass_model_identifies_fault_and_defect_type(self):
        if not BEST_MULTI_PATH.exists():
            pytest.skip("best_multiclass_model.pt not present")
        det = CWRUAdvancedClassifierDetector(model_path=BEST_MULTI_PATH)
        assert det._is_dict_model is True

        # Healthy: 97.mat
        sig_h = _load_cwru_sample("97.mat")[:1000]
        feat_h = det.prepare_feature(sig_h)
        score_h, label_h = det.predict(feat_h)
        assert label_h == "normal"
        assert feat_h["diagnosis"] == "healthy"

        # Inner Race Fault: 105.mat (gives fault AND what the fault is)
        sig_ir = _load_cwru_sample("105.mat")[:1000]
        feat_ir = det.prepare_feature(sig_ir)
        score_ir, label_ir = det.predict(feat_ir)
        assert label_ir == "anomaly"
        assert "inner_race_fault" in feat_ir["diagnosis"]

        # Ball Fault: 118.mat (gives fault AND what the fault is)
        sig_ball = _load_cwru_sample("118.mat")[:1000]
        feat_ball = det.prepare_feature(sig_ball)
        score_ball, label_ball = det.predict(feat_ball)
        assert label_ball == "anomaly"
        assert "ball_fault" in feat_ball["diagnosis"]


class TestCWRUHybridDetector:
    def test_hybrid_two_stage_inference(self):
        hybrid = CWRUHybridDetector(
            unsupervised_path=UNSUP_PATH,
            classifier_path=ADV_PATH,
            threshold=0.05,
        )

        # 1. Normal window
        sig_h = _load_cwru_sample("97.mat")[:12000]
        feat_h = hybrid.prepare_feature(sig_h)
        score_h, label_h = hybrid.predict(feat_h)
        assert label_h == "normal"
        assert feat_h["diagnosis"] == "healthy"

        # 2. Fault window
        sig_f = _load_cwru_sample("105.mat")[:12000]
        feat_f = hybrid.prepare_feature(sig_f)
        score_f, label_f = hybrid.predict(feat_f)
        assert label_f == "anomaly"
        assert feat_f["diagnosis"] == "inner_race_fault"
        assert feat_f["diagnosis_confidence"] > 0.95


class TestServiceIntegrationWithEdgeModels:
    def test_build_detector_factories(self):
        d_unsup = build_detector({"active": "unsupervised_autoencoder", "alert_threshold": 0.04})
        assert isinstance(d_unsup, CWRUUnsupervisedDetector)
        assert d_unsup.threshold == 0.04

        d_adv = build_detector({"active": "advanced_classifier"})
        assert isinstance(d_adv, CWRUAdvancedClassifierDetector)

        d_hyb = build_detector({"active": "hybrid_edge", "alert_threshold": 0.06})
        assert isinstance(d_hyb, CWRUHybridDetector)
        assert d_hyb.threshold == 0.06

    def test_monitor_service_with_hybrid_detector(self, tmp_path):
        sig_raw = _load_cwru_sample("105.mat")[:10000]
        raw_file = tmp_path / "sample.npy"
        np.save(str(raw_file), sig_raw)

        cfg_dict = {
            "sensor": {"type": "mock", "mock_data_path": str(raw_file), "sample_rate": 1000},
            "motor": {"run_frequencies": {"medium": 40.0}},
            "detector": {
                "active": "hybrid_edge",
                "alert_threshold": 0.05,
                "unsupervised_path": str(UNSUP_PATH),
                "classifier_path": str(ADV_PATH),
            },
            "mqtt": {
                "broker": "localhost",
                "port": 1883,
                "topic_health": "test/health",
                "topic_alert": "test/alert",
                "topic_config": "test/config",
            },
        }
        import yaml
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(cfg_dict, f)

        with patch("paho.mqtt.client.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            service = MonitorService(config_path=str(config_path), force_mock=True)
            service.publisher._connected = True
            service._process_one_window()

            # Verify MQTT received health update and diagnosis
            assert mock_client.publish.called
            calls = [c[0] for c in mock_client.publish.call_args_list]
            health_calls = [c for c in calls if c[0] == "test/health"]
            assert len(health_calls) == 1
            payload = json.loads(health_calls[0][1])
            assert payload["label"] == "anomaly"
            assert payload["diagnosis"] == "inner_race_fault"
