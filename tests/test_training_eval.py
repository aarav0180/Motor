"""
tests/test_training_eval.py
---------------------------
Integration tests for training scripts and evaluation pipelines:
  - training/train_rms_threshold.py
  - training/train_isolation_forest.py
  - training/train_autoencoder.py
  - evaluation/metrics.py
  - evaluation/evaluate_rig.py
  - evaluation/evaluate_cwru.py
"""

from argparse import Namespace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from training.train_rms_threshold import main as train_rms_main
from training.train_isolation_forest import main as train_iso_main
from training.train_autoencoder import main as train_ae_main
from evaluation.metrics import compute_metrics, print_metrics_table, threshold_sweep
from evaluation.evaluate_rig import run_evaluation as run_eval_rig
from evaluation.evaluate_cwru import run_evaluation as run_eval_cwru
from detectors import RMSThresholdDetector, IsolationForestDetector, AutoencoderDetector


@pytest.fixture(scope="module")
def synthetic_training_env(tmp_path_factory):
    """Create a temporary environment with raw windows and output model directory."""
    base_dir = tmp_path_factory.mktemp("train_eval_env")
    data_dir = base_dir / "raw"
    data_dir.mkdir()
    models_dir = base_dir / "models"
    models_dir.mkdir()

    # Generate 15 healthy windows
    fs = 1000
    win_len = fs * 10
    rng = np.random.default_rng(123)

    for i in range(15):
        healthy_win = (rng.standard_normal(win_len) * 0.2).astype(np.float32)
        np.save(str(data_dir / f"healthy_medium_sample_{i:02d}.npy"), healthy_win)

    return {"data_dir": str(data_dir), "models_dir": str(models_dir), "base_dir": base_dir}


class TestTrainingPipelines:
    def test_train_rms_threshold_script(self, synthetic_training_env):
        args = Namespace(
            data_dir=synthetic_training_env["data_dir"],
            output_dir=synthetic_training_env["models_dir"],
            speed="medium",
            sample_rate=1000,
            run_freq=40.0,
            sigma=3.0,
        )
        train_rms_main(args)

        model_path = Path(synthetic_training_env["models_dir"]) / "rms_threshold.pkl"
        assert model_path.exists()
        loaded = RMSThresholdDetector.load(str(model_path))
        assert loaded._is_fitted is True

    def test_train_isolation_forest_script(self, synthetic_training_env):
        args = Namespace(
            data_dir=synthetic_training_env["data_dir"],
            output_dir=synthetic_training_env["models_dir"],
            speed="medium",
            sample_rate=1000,
            run_freq=40.0,
        )
        train_iso_main(args)

        model_path = Path(synthetic_training_env["models_dir"]) / "iso_forest.pkl"
        assert model_path.exists()
        loaded = IsolationForestDetector.load(str(model_path))
        assert loaded._is_fitted is True

    def test_train_autoencoder_script(self, synthetic_training_env):
        args = Namespace(
            data_dir=synthetic_training_env["data_dir"],
            output_dir=synthetic_training_env["models_dir"],
            speed="medium",
            n_fft_bins=128,  # small for fast test
            epochs=2,
            lr=1e-3,
            batch_size=4,
        )
        train_ae_main(args)

        model_path = Path(synthetic_training_env["models_dir"]) / "autoencoder.pt"
        assert model_path.exists()
        loaded = AutoencoderDetector.load(str(model_path))
        assert loaded._is_fitted is True


class TestMetricsAndEvaluation:
    def test_compute_metrics_calculation(self):
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        y_pred = [0, 0, 0, 1, 1, 1, 1, 0]  # 1 FP, 1 FN, 3 TP, 3 TN

        m = compute_metrics(y_true, y_pred)
        assert m["tp"] == 3
        assert m["fp"] == 1
        assert m["fn"] == 1
        assert m["tn"] == 3
        assert m["precision"] == pytest.approx(3 / 4)
        assert m["recall"] == pytest.approx(3 / 4)
        assert m["f1"] == pytest.approx(0.75)
        assert m["accuracy"] == pytest.approx(6 / 8)
        assert m["false_alarm_rate"] == pytest.approx(1 / 4)

    def test_threshold_sweep(self):
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        labels = np.array([0, 0, 0, 1, 1, 1])

        best_t, best_m = threshold_sweep(scores, labels, n_steps=20)
        assert 0.3 <= best_t <= 0.7
        assert best_m["f1"] == 1.0

    def test_print_metrics_table_formatting(self, capsys):
        res = {
            "Model A": {"precision": 0.95, "recall": 0.90, "f1": 0.925, "false_alarm_rate": 0.05, "accuracy": 0.94}
        }
        print_metrics_table(res, title="Unit Test Table")
        out = capsys.readouterr().out
        assert "Unit Test Table" in out
        assert "Model A" in out
        assert "0.925" in out

    def test_evaluate_rig_script(self, synthetic_training_env, tmp_path):
        # Create a test rig features CSV
        csv_path = tmp_path / "rig_features.csv"
        df = pd.DataFrame([
            {"label": "healthy", "rms": 0.20, "peak": 0.6, "crest_factor": 3.0, "kurtosis": 0.1, "spectral_entropy": 5.0, "dominant_freq": 40.0, "band1_energy": 0.1, "band2_energy": 0.05, "band3_energy": 0.01, "skewness": 0.0, "std": 0.2},
            {"label": "imbalance_5g", "rms": 0.85, "peak": 2.5, "crest_factor": 2.9, "kurtosis": 1.5, "spectral_entropy": 4.5, "dominant_freq": 40.0, "band1_energy": 0.8, "band2_energy": 0.4, "band3_energy": 0.2, "skewness": 0.1, "std": 0.85},
        ])
        df.to_csv(str(csv_path), index=False)

        # Run evaluate_rig without throwing exceptions
        run_eval_rig(str(csv_path), models_dir=synthetic_training_env["models_dir"])

    def test_evaluate_cwru_script(self, synthetic_training_env):
        # Run evaluate_cwru using data/cwru/ directory
        cwru_dir = Path("data/cwru")
        if cwru_dir.exists() and any(cwru_dir.glob("*.mat")):
            run_eval_cwru(str(cwru_dir), models_dir=synthetic_training_env["models_dir"])
