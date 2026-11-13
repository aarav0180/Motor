"""
tests/test_e2e_edge_server.py
-----------------------------
End-to-end integration test demonstrating the full 3-Tier loop:
  Tier 1 Edge Daemon (Inference) ──► MQTT ──► Tier 2 Server (Ingestion & DB)
  Tier 3 Client App Feedback    ──► Server API ──► MQTT ──► Tier 1 Edge (Threshold Adaptation)
"""

from unittest.mock import MagicMock, patch
import json
from pathlib import Path
import numpy as np
import pytest
import scipy.io
from fastapi.testclient import TestClient

from server.config import settings
from server.database import init_db, get_latest_telemetry
from server.main import app
from server.mqtt_bridge import mqtt_bridge
from service.monitor import MonitorService


@pytest.fixture
def e2e_env(tmp_path):
    """Sets up unified edge-to-server testing sandbox."""
    db_file = str(tmp_path / "e2e_server.db")
    settings.database_path = db_file
    init_db(db_file)

    # Load 105.mat inner race fault sample
    mat_p = Path("data/cwru/105.mat")
    mat = scipy.io.loadmat(str(mat_p))
    sig = mat["X105_DE_time"].flatten()[:10000].astype(np.float32)
    mock_sensor_file = tmp_path / "fault_sensor.npy"
    np.save(str(mock_sensor_file), sig)

    # Build config for MonitorService
    cfg_file = tmp_path / "e2e_config.yaml"
    cfg_data = {
        "sensor": {
            "type": "mock",
            "mock_data_path": str(mock_sensor_file),
            "sample_rate": 1000,
            "simulate_delay": False,
        },
        "motor": {"run_frequencies": {"medium": 40.0}},
        "detector": {
            "active": "hybrid_edge",
            "alert_threshold": 0.05,
            "unsupervised_path": "models/cwru_unsupervised_xiong_edge.pt",
            "classifier_path": "models/cwru_advanced_edge.pt",
        },
        "mqtt": {
            "broker": "localhost",
            "port": 1883,
            "topic_health": "motor/health",
            "topic_alert": "motor/alert",
            "topic_config": "motor/config/threshold",
        },
    }
    import yaml
    with open(cfg_file, "w") as f:
        yaml.dump(cfg_data, f)

    return {"cfg_file": str(cfg_file), "db_file": db_file}


class TestEndToEndEdgeServerCycle:
    def test_full_edge_to_server_and_feedback_adaptation(self, e2e_env):
        client = TestClient(app)

        # Shared simulated MQTT transport broker
        published_messages = []

        def mock_publish(topic, payload_str, qos=1):
            published_messages.append((topic, payload_str))
            info = MagicMock()
            info.wait_for_publish.return_value = True
            return info

        with patch("paho.mqtt.client.Client") as mock_mqtt_cls:
            mock_client = MagicMock()
            mock_mqtt_cls.return_value = mock_client
            mock_client.publish.side_effect = mock_publish

            # -------------------------------------------------------------
            # 1. Tier 1: Initialize Edge Daemon & Process 1 Window
            # -------------------------------------------------------------
            edge_service = MonitorService(config_path=e2e_env["cfg_file"], force_mock=True)
            edge_service.publisher._connected = True

            # Process window through deep learning edge detector
            edge_service._process_one_window()

            # Verify edge published health update & alert
            assert len(published_messages) >= 2
            health_msgs = [m for m in published_messages if m[0] == "motor/health"]
            assert len(health_msgs) == 1
            health_payload = json.loads(health_msgs[0][1])

            assert health_payload["label"] == "anomaly"
            assert health_payload["diagnosis"] == "inner_race_fault"
            assert health_payload["score"] > 0.50

            # -------------------------------------------------------------
            # 2. Tier 2: Server Ingests Edge Telemetry
            # -------------------------------------------------------------
            mqtt_bridge._client = mock_client
            mqtt_bridge._connected = True

            mock_paho_msg = MagicMock()
            mock_paho_msg.topic = "motor/health"
            mock_paho_msg.payload = health_msgs[0][1].encode()

            # Deliver to server message handler
            mqtt_bridge._on_message(None, None, mock_paho_msg)

            # Query server REST API to verify ingested data
            resp = client.get("/api/v1/telemetry/latest")
            assert resp.status_code == 200
            latest_telemetry = resp.json()
            assert latest_telemetry["diagnosis"] == "inner_race_fault"
            assert latest_telemetry["score"] == health_payload["score"]

            # -------------------------------------------------------------
            # 3. Tier 3: Human Technician Submits False Alarm Feedback
            # -------------------------------------------------------------
            # Technician reviews and decides to raise threshold with 5% margin
            current_score = latest_telemetry["score"]

            feedback_payload = {
                "event": "FALSE_ALARM",
                "score": current_score,
                "notes": "Transient line vibration — recalibrating edge threshold.",
            }

            resp_fb = client.post("/api/v1/feedback", json=feedback_payload)
            assert resp_fb.status_code == 200
            fb_data = resp_fb.json()
            expected_new_thresh = round(current_score * 1.05, 4)
            assert fb_data["adapted_threshold"] == expected_new_thresh

            # -------------------------------------------------------------
            # 4. Loop Closure: Edge Receives Adapted Threshold via MQTT
            # -------------------------------------------------------------
            # The server published threshold update down to MQTT
            config_msgs = [m for m in published_messages if m[0] == "motor/config/threshold"]
            assert len(config_msgs) >= 1
            last_config_payload = json.loads(config_msgs[-1][1])
            assert last_config_payload["threshold"] == expected_new_thresh

            # Deliver update to Edge Daemon remote config handler
            edge_service._handle_remote_config(last_config_payload)

            # Verify edge detector dynamically adapted its threshold without restarting!
            assert edge_service.detector.threshold == expected_new_thresh
            assert edge_service.cfg["detector"]["alert_threshold"] == expected_new_thresh
