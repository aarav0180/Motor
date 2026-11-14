"""
server/mock_streamer.py — Background Edge Mock Streamer for Tier 2 Cloud.

Simulates an edge device (Tier 1) over MQTT when sensor mode is set to 'mock'.
Allows the cloud web dashboard and alert engine to function live with continuous
CWRU bearing data replay and fault injection even when no physical Raspberry Pi is online.

Strictly follows the 3-Tier decoupled architecture:
Communicates exclusively via the MQTT broker on standard topics:
  - Publishes: `motor/health`, `motor/alert`
  - Subscribes: `motor/config/threshold`, `motor/config/sensor`, `motor/config/model`
"""

import json
import logging
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from .config import settings
from .database import get_config

logger = logging.getLogger("server.mock_streamer")


class MockEdgeStreamer:
    """
    Simulated Edge Device daemon publishing continuous CWRU telemetry to MQTT.
    """

    def __init__(
        self,
        broker: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: Optional[bool] = None,
        stream_interval: float = 2.5,
    ) -> None:
        self.broker = broker or settings.mqtt_broker
        self.port = port or settings.mqtt_port
        self.username = username or settings.mqtt_username
        self.password = password or settings.mqtt_password
        self.use_tls = use_tls if use_tls is not None else settings.mqtt_use_tls
        self.stream_interval = stream_interval

        self.running = False
        self.active = True  # True when sensor_mode == 'mock'
        self.active_model = "unsupervised"  # 'unsupervised' or 'supervised'
        self.active_threshold = float(settings.default_threshold)

        self._thread: Optional[threading.Thread] = None
        self._client = None
        self._connected = False
        self._sensor = None
        self._extractor = None
        self._unsup_detector = None
        self._super_detector = None

    def _init_components(self) -> None:
        """Lazy-initialize sensor, feature extractor, and PyTorch detectors."""
        try:
            from hal.mock_sensor import create_dynamic_mock_sensor
            from features.extractor import FeatureExtractor
            from detectors.edge_models import (
                CWRUUnsupervisedDetector,
                CWRUAdvancedClassifierDetector,
            )

            self._sensor = create_dynamic_mock_sensor(
                cwru_dir="data/cwru",
                sample_rate=12_000,
                simulate_delay=False,
                fault_every_n=4,
            )
            self._extractor = FeatureExtractor(sample_rate=12_000, run_freq_hz=40.0)

            try:
                self._unsup_detector = CWRUUnsupervisedDetector(
                    model_path="models/cwru_unsupervised_xiong_edge.pt",
                    threshold=self.active_threshold,
                )
            except Exception as exc:
                logger.warning("Could not load unsupervised model (%s); using fallback scoring.", exc)
                self._unsup_detector = None

            try:
                self._super_detector = CWRUAdvancedClassifierDetector(
                    model_path="models/cwru_advanced_edge.pt",
                    threshold=0.50,
                )
            except Exception as exc:
                logger.warning("Could not load supervised model (%s); using fallback scoring.", exc)
                self._super_detector = None

            logger.info("MockEdgeStreamer initialized models & dynamic CWRU sensor.")
        except Exception as exc:
            logger.error("Failed to initialize MockEdgeStreamer components: %s", exc)

    def start(self) -> None:
        """Launch background worker thread."""
        if self.running:
            return

        # Check initial sensor mode and model from DB
        current_sensor = str(get_config("sensor_mode", settings.default_sensor_mode))
        self.active = (current_sensor.lower() == "mock")
        self.active_model = str(get_config("active_model", settings.default_model))
        self.active_threshold = float(get_config("alert_threshold", settings.default_threshold))

        self.running = True
        self._thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="MockEdgeStreamer-Worker",
        )
        self._thread.start()
        logger.info("MockEdgeStreamer worker started (active=%s, model=%s).", self.active, self.active_model)

    def stop(self) -> None:
        """Stop worker cleanly."""
        self.running = False
        if self._client is not None:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
            self._connected = False
        logger.info("MockEdgeStreamer stopped.")

    def _setup_mqtt(self) -> bool:
        """Connect dedicated MQTT client for edge mock simulator."""
        try:
            import paho.mqtt.client as mqtt

            client_id = f"motor-edge-mock-streamer-{int(time.time())}"
            self._client = mqtt.Client(client_id=client_id)

            if self.username:
                self._client.username_pw_set(self.username, self.password)
            if self.use_tls:
                import ssl
                self._client.tls_set(cert_reqs=ssl.CERT_REQUIRED)

            def on_connect(client, userdata, flags, rc):
                if rc == 0:
                    self._connected = True
                    logger.info("MockEdgeStreamer connected to MQTT broker %s:%d", self.broker, self.port)
                    client.subscribe([
                        (settings.topic_config, 1),
                        (settings.topic_sensor_config, 1),
                        (settings.topic_model_config, 1),
                    ])
                else:
                    self._connected = False
                    logger.warning("MockEdgeStreamer MQTT connection failed (code %d)", rc)

            def on_disconnect(client, userdata, rc):
                self._connected = False

            def on_message(client, userdata, msg):
                self._handle_mqtt_config(msg.topic, msg.payload)

            self._client.on_connect = on_connect
            self._client.on_disconnect = on_disconnect
            self._client.on_message = on_message

            self._client.connect(self.broker, self.port, keepalive=60)
            self._client.loop_start()
            return True
        except Exception as exc:
            logger.warning("MockEdgeStreamer could not connect to MQTT (%s); will retry.", exc)
            self._connected = False
            return False

    def _handle_mqtt_config(self, topic: str, raw_bytes: bytes) -> None:
        """Process remote config updates broadcast over MQTT."""
        try:
            payload = json.loads(raw_bytes.decode())
        except Exception:
            return

        if topic == settings.topic_sensor_config:
            mode = payload.get("sensor_mode") or payload.get("mode")
            if mode:
                self.active = (str(mode).lower() == "mock")
                logger.info("MockEdgeStreamer updated active state = %s (sensor_mode=%s)", self.active, mode)

        elif topic == settings.topic_model_config:
            model = payload.get("model") or payload.get("active_model")
            if model:
                self.active_model = str(model).lower()
                logger.info("MockEdgeStreamer switched active model = %s", self.active_model)

        elif topic == settings.topic_config:
            new_thresh = payload.get("threshold") or payload.get("alert_threshold")
            if new_thresh is not None:
                self.active_threshold = float(new_thresh)
                if self._unsup_detector and hasattr(self._unsup_detector, "threshold"):
                    self._unsup_detector.threshold = self.active_threshold
                logger.info("MockEdgeStreamer updated threshold = %.4f", self.active_threshold)

    def _worker_loop(self) -> None:
        """Main loop: acquire window, score, publish to MQTT every stream_interval."""
        self._init_components()

        # Connect to MQTT
        connected_ok = self._setup_mqtt()
        if not connected_ok:
            # Brief backoff before retrying
            time.sleep(2.0)

        while self.running:
            # If paused (e.g. running physical Pi), wait
            if not self.active:
                time.sleep(1.0)
                continue

            # Ensure MQTT connection
            if not self._connected:
                time.sleep(2.0)
                continue

            if self._sensor is None or self._extractor is None:
                time.sleep(1.0)
                continue

            try:
                # 1. Acquire dynamic window
                window = self._sensor.get_10s_window()

                # 2. Extract features
                fv = self._extractor.transform(window)
                feat = fv.to_dict()

                # 3. Score with active detector
                if self.active_model == "supervised" and self._super_detector is not None:
                    feat.update(self._super_detector.prepare_feature(window))
                    score, label = self._super_detector.predict(feat)
                    detector_name = "advanced_classifier"
                    diagnosis = feat.get("diagnosis", "healthy" if label == "normal" else "inner_race_fault")
                    confidence = float(feat.get("diagnosis_confidence", 0.98 if label == "anomaly" else 0.95))
                elif self._unsup_detector is not None:
                    feat.update(self._unsup_detector.prepare_feature(window))
                    score, label = self._unsup_detector.predict(feat)
                    detector_name = "unsupervised_autoencoder"
                    diagnosis = self._sensor.current_label if label == "anomaly" else "healthy"
                    confidence = float(min(1.0, score / max(self.active_threshold, 1e-4))) if label == "anomaly" else 0.95
                else:
                    # Synthetic fallback scoring based on RMS & Kurtosis
                    rms = float(feat.get("rms", 0.1))
                    kurt = float(feat.get("kurtosis", 3.0))
                    score = min(1.0, (rms * 2.0) + (max(0.0, kurt - 3.0) * 0.1))
                    label = "anomaly" if score > self.active_threshold else "normal"
                    detector_name = "fallback_heuristic"
                    diagnosis = "healthy" if label == "normal" else "bearing_fault"
                    confidence = 0.92

                # 4. Formulate Telemetry Payload
                now = time.time()
                telemetry_payload = {
                    "motor_id": "motor_1",
                    "rms": round(float(feat.get("rms", 0.0)), 5),
                    "kurtosis": round(float(feat.get("kurtosis", 3.0)), 3),
                    "crest_factor": round(float(feat.get("crest_factor", 2.5)), 3),
                    "peak": round(float(feat.get("peak", 0.0)), 5),
                    "variance": round(float(feat.get("std", 0.0) ** 2), 5),
                    "score": round(float(score), 5),
                    "label": label,
                    "detector": detector_name,
                    "diagnosis": diagnosis,
                    "confidence": round(confidence, 4),
                    "sensor_mode": "mock",
                    "ts": now,
                }

                # 5. Publish to motor/health
                self._client.publish(settings.topic_health, json.dumps(telemetry_payload), qos=1)

                # 6. If anomaly, publish to motor/alert
                if label == "anomaly":
                    alert_payload = {
                        "motor_id": "motor_1",
                        "score": round(float(score), 5),
                        "message": f"FAULT DETECTED [{diagnosis.upper()}] — score={score:.4f}",
                        "diagnosis": diagnosis,
                        "ts": now,
                    }
                    self._client.publish(settings.topic_alert, json.dumps(alert_payload), qos=1)
                    logger.info("MockEdgeStreamer published ALERT [%s] score=%.4f", diagnosis, score)

            except Exception as exc:
                logger.error("Error in MockEdgeStreamer cycle: %s", exc)

            time.sleep(self.stream_interval)


# Global singleton instance
mock_streamer = MockEdgeStreamer()
