"""
service/monitor.py — Main live monitoring service loop.

Entry point for the deployed Pi service.  Wires together:
  HAL (sensor) → Feature Extraction → Detector → MQTT Publisher

Usage
-----
    python -m service.monitor                     # uses config.yaml
    python -m service.monitor --config my.yaml    # custom config path
    python -m service.monitor --mock              # force MockSensor

The service runs until interrupted (Ctrl-C / SIGTERM).
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import yaml
import numpy as np

from hal  import PhysicalSensor, MockSensor, VibrationSensor
from features import FeatureExtractor
from detectors import (
    BaseDetector,
    RMSThresholdDetector,
    IsolationForestDetector,
    AutoencoderDetector,
    CWRUUnsupervisedDetector,
    CWRUAdvancedClassifierDetector,
    CWRUHybridDetector,
)
from service.mqtt_publisher import MQTTPublisher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("monitor")


# ---------------------------------------------------------------------------
# Factories — build objects from config dicts
# ---------------------------------------------------------------------------

def build_sensor(cfg: dict, force_mock: bool = False) -> VibrationSensor:
    """Instantiate the sensor from the [sensor] config section."""
    if force_mock or cfg.get("type") == "mock":
        data_path = Path(cfg.get("mock_data_path", "data/raw/healthy_sample.npy"))
        if data_path.exists():
            data = np.load(str(data_path))
            return MockSensor(
                data=data,
                sample_rate=cfg.get("sample_rate", 12_000),
                simulate_delay=cfg.get("simulate_delay", False),
                label="mock",
            )
        from hal.mock_sensor import create_dynamic_mock_sensor
        return create_dynamic_mock_sensor(
            cwru_dir=cfg.get("cwru_dir", "data/cwru"),
            sample_rate=cfg.get("sample_rate", 12_000),
            simulate_delay=cfg.get("simulate_delay", False),
            fault_every_n=cfg.get("fault_every_n", 4),
        )
    return PhysicalSensor(
        bus_num=cfg.get("bus", 1),
        address=int(cfg.get("address", "0x53"), 16)
                if isinstance(cfg.get("address"), str)
                else cfg.get("address", 0x53),
        sample_rate=cfg.get("sample_rate", 1000),
    )


def build_detector(cfg: dict) -> BaseDetector:
    """Load a pre-trained detector from the [detector] config section."""
    name       = cfg.get("active", "rms_threshold")
    models_dir = Path(cfg.get("models_dir", "detectors/saved"))
    threshold  = cfg.get("alert_threshold", None)

    if name == "rms_threshold":
        path = models_dir / "rms_threshold.pkl"
        if path.exists():
            return RMSThresholdDetector.load(str(path))
        raise FileNotFoundError(f"Model not found: {path}.  Run training first.")

    if name == "isolation_forest":
        path = models_dir / "iso_forest.pkl"
        if path.exists():
            return IsolationForestDetector.load(str(path))
        raise FileNotFoundError(f"Model not found: {path}.  Run training first.")

    if name == "autoencoder":
        path = models_dir / "autoencoder.pt"
        if path.exists():
            return AutoencoderDetector.load(str(path))
        raise FileNotFoundError(f"Model not found: {path}.  Run training first.")

    if name in ("unsupervised_autoencoder", "unsupervised_xiong"):
        model_path = cfg.get("model_path", "models/cwru_unsupervised_xiong_edge.pt")
        thresh = float(threshold) if threshold is not None else 0.05
        return CWRUUnsupervisedDetector(model_path=model_path, threshold=thresh)

    if name in ("advanced_classifier", "classifier"):
        model_path = cfg.get("model_path", "models/cwru_advanced_edge.pt")
        thresh = float(threshold) if threshold is not None else 0.50
        return CWRUAdvancedClassifierDetector(model_path=model_path, threshold=thresh)

    if name in ("hybrid", "hybrid_edge"):
        unsup_path = cfg.get("unsupervised_path", "models/cwru_unsupervised_xiong_edge.pt")
        cls_path = cfg.get("classifier_path", "models/cwru_advanced_edge.pt")
        thresh = float(threshold) if threshold is not None else 0.05
        return CWRUHybridDetector(
            unsupervised_path=unsup_path,
            classifier_path=cls_path,
            threshold=thresh,
        )

    raise ValueError(f"Unknown detector type: '{name}'")


# ---------------------------------------------------------------------------
# Main service loop
# ---------------------------------------------------------------------------

class MonitorService:
    """
    Orchestrates the sensor → feature extraction → detection → MQTT pipeline.

    Parameters
    ----------
    config_path : str
        Path to config.yaml.
    force_mock : bool
        Override sensor type to MockSensor regardless of config.
    """

    def __init__(self, config_path: str = "config.yaml", force_mock: bool = False) -> None:
        with open(config_path) as fh:
            self.cfg = yaml.safe_load(fh)

        self.sensor    = build_sensor(self.cfg["sensor"], force_mock=force_mock)
        self.detector  = build_detector(self.cfg["detector"])
        self.extractor = FeatureExtractor(
            sample_rate=self.cfg["sensor"]["sample_rate"],
            run_freq_hz=self.cfg["motor"]["run_frequencies"].get("medium", 40.0),
        )
        topic_config = self.cfg.get("mqtt", {}).get("topic_config", "motor/config/threshold")
        topic_sensor_config = self.cfg.get("mqtt", {}).get("topic_sensor_config", "motor/config/sensor")
        topic_model_config = self.cfg.get("mqtt", {}).get("topic_model_config", "motor/config/model")
        self.publisher = MQTTPublisher(
            broker=self.cfg["mqtt"]["broker"],
            port=self.cfg["mqtt"].get("port", 1883),
            topic_health=self.cfg["mqtt"]["topic_health"],
            topic_alert=self.cfg["mqtt"]["topic_alert"],
            topic_config=topic_config,
            topic_sensor_config=topic_sensor_config,
            topic_model_config=topic_model_config,
            username=self.cfg.get("mqtt", {}).get("username"),
            password=self.cfg.get("mqtt", {}).get("password"),
            use_tls=bool(self.cfg.get("mqtt", {}).get("use_tls", False)),
            on_config_message=self._handle_remote_config,
            on_sensor_config_message=self._handle_sensor_config,
            on_model_config_message=self._handle_model_config,
        )
        self._running = False

    def _handle_remote_config(self, payload: dict) -> None:
        """
        Handle remote calibration / threshold updates from Server/App (Human-in-the-Loop).
        Dynamically updates detector threshold on the edge without restarting the service.
        """
        if not isinstance(payload, dict):
            return
        new_threshold = payload.get("threshold") or payload.get("alert_threshold")
        if new_threshold is not None:
            try:
                new_threshold = float(new_threshold)
                if hasattr(self.detector, "threshold"):
                    self.detector.threshold = new_threshold
                self.cfg.setdefault("detector", {})["alert_threshold"] = new_threshold
                logger.info(
                    "Remote config applied from Server: updated alert_threshold = %.4f",
                    new_threshold,
                )
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Failed to parse remote threshold value '%s': %s",
                    new_threshold, exc,
                )

    def _handle_sensor_config(self, payload: dict) -> None:
        """Handle dynamic sensor mode switch ('mock' vs 'physical') from Server/Web UI."""
        if not isinstance(payload, dict):
            return
        mode = payload.get("sensor_mode") or payload.get("mode")
        if mode:
            self.set_sensor_mode(str(mode))

    def set_sensor_mode(self, mode: str) -> bool:
        """
        Hot-swap the active sensor between MockSensor and PhysicalSensor live.
        Catches hardware / I2C missing errors safely when running on laptop.
        """
        mode = mode.lower().strip()
        logger.info("Switching sensor mode to '%s'...", mode)
        try:
            if mode == "physical":
                from hal.physical_sensor import PhysicalSensor
                new_sensor = PhysicalSensor(
                    bus_num=self.cfg["sensor"].get("bus", 1),
                    address=int(self.cfg["sensor"].get("address", "0x53"), 16)
                            if isinstance(self.cfg["sensor"].get("address"), str)
                            else self.cfg["sensor"].get("address", 0x53),
                    sample_rate=self.cfg["sensor"].get("sample_rate", 1000),
                )
            else:
                new_sensor = build_sensor(self.cfg["sensor"], force_mock=True)

            self.sensor = new_sensor
            self.cfg["sensor"]["type"] = mode
            logger.info("Successfully hot-swapped sensor to '%s'.", mode)
            return True
        except Exception as exc:
            logger.error(
                "Failed to switch sensor mode to '%s' (%s). Retaining current sensor.",
                mode, exc,
            )
            return False

    def _handle_model_config(self, payload: dict) -> None:
        """Handle remote model selection switch ('unsupervised' vs 'supervised') from Server/Web UI."""
        if not isinstance(payload, dict):
            return
        model = payload.get("model") or payload.get("active_model")
        if model:
            self.set_model_mode(str(model))

    def set_model_mode(self, model: str) -> bool:
        """
        Hot-swap active detector between Unsupervised Autoencoder and Supervised Classifier.
        """
        model = model.lower().strip()
        logger.info("Switching detector model to '%s'...", model)
        try:
            if model in ("supervised", "classifier", "advanced_classifier"):
                target_detector = "advanced_classifier"
            else:
                target_detector = "unsupervised_autoencoder"

            self.cfg.setdefault("detector", {})["active"] = target_detector
            self.detector = build_detector(self.cfg["detector"])
            logger.info("Successfully hot-swapped detector to '%s'.", target_detector)
            return True
        except Exception as exc:
            logger.error("Failed to switch detector model to '%s' (%s). Retaining current detector.", model, exc)
            return False

    def run(self) -> None:
        """Start the monitoring loop.  Blocks until stopped."""
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT,  self._handle_signal)

        self.publisher.connect()
        self._running = True
        logger.info("Monitor service started.  Press Ctrl-C to stop.")

        try:
            while self._running:
                self._process_one_window()
        finally:
            self.publisher.disconnect()
            logger.info("Monitor service stopped.")

    def _process_one_window(self) -> None:
        """Acquire one window, extract features, score, publish."""
        t0 = time.perf_counter()
        window = self.sensor.get_10s_window()
        fv     = self.extractor.transform(window)
        feat   = fv.to_dict()

        # Attach raw window or spectrum if detector requires it
        detector_name = self.cfg["detector"]["active"]
        if hasattr(self.detector, "prepare_feature"):
            feat.update(self.detector.prepare_feature(window))

        score, label = self.detector.predict(feat)
        elapsed = time.perf_counter() - t0

        logger.info(
            "Window processed in %.2f s — score=%.4f  label=%s",
            elapsed, score, label,
        )

        extra_health = {
            "detector": detector_name,
            "proc_time_s": round(elapsed, 3),
        }
        if "diagnosis" in feat:
            extra_health["diagnosis"] = feat["diagnosis"]
            extra_health["confidence"] = feat.get("diagnosis_confidence", 1.0)

        self.publisher.publish_health(
            score=score, label=label,
            extra=extra_health,
        )

        if label == "anomaly":
            alert_msg = f"FAULT DETECTED — score={score:.4f}"
            if "diagnosis" in feat and feat["diagnosis"] != "healthy":
                alert_msg = f"FAULT DETECTED [{feat['diagnosis'].upper()}] — score={score:.4f}"
            self.publisher.publish_alert(
                message=alert_msg,
                score=score,
            )

    def _handle_signal(self, signum, frame) -> None:
        logger.info("Signal %d received — shutting down.", signum)
        self._running = False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Motor Fault Detection Service")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--mock", action="store_true", help="Use MockSensor (offline mode)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    service = MonitorService(config_path=args.config, force_mock=args.mock)
    service.run()
