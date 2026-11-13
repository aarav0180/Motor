"""
tests/test_app_service.py
-------------------------
Unit and integration tests for the live monitoring app & MQTT service
(service.monitor and service.mqtt_publisher).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from hal import MockSensor, PhysicalSensor
from detectors import RMSThresholdDetector
from service.monitor import build_sensor, build_detector, MonitorService
from service.mqtt_publisher import MQTTPublisher


class TestBuildSensor:
    def test_build_mock_sensor_from_file(self, tmp_path):
        data_file = tmp_path / "healthy.npy"
        np.save(str(data_file), np.random.randn(12000).astype(np.float32))

        cfg = {
            "type": "mock",
            "mock_data_path": str(data_file),
            "sample_rate": 12000,
            "simulate_delay": False,
        }
        sensor = build_sensor(cfg, force_mock=True)
        assert isinstance(sensor, MockSensor)
        assert sensor.sample_rate == 12000

    def test_build_mock_sensor_fallback_synthetic(self, tmp_path):
        cfg = {
            "type": "mock",
            "mock_data_path": str(tmp_path / "nonexistent.npy"),
            "sample_rate": 1000,
            "simulate_delay": False,
        }
        # Fallback will kick in cleanly
        sensor = build_sensor(cfg, force_mock=True)
        assert isinstance(sensor, MockSensor)

    def test_build_physical_sensor(self):
        mock_smbus2 = MagicMock()
        mock_bus = MagicMock()
        mock_smbus2.SMBus.return_value = mock_bus

        with patch.dict("sys.modules", {"smbus2": mock_smbus2}):
            cfg = {
                "type": "physical",
                "bus": 1,
                "address": "0x53",
                "sample_rate": 1000,
            }
            sensor = build_sensor(cfg, force_mock=False)
            assert isinstance(sensor, PhysicalSensor)
            assert sensor.address == 0x53


class TestBuildDetector:
    def test_build_rms_detector_success(self, tmp_path):
        det = RMSThresholdDetector(sigma=3.0)
        det.fit([{"rms": 0.5}, {"rms": 0.52}, {"rms": 0.48}])
        det_path = tmp_path / "rms_threshold.pkl"
        det.save(str(det_path))

        cfg = {
            "active": "rms_threshold",
            "models_dir": str(tmp_path),
        }
        loaded = build_detector(cfg)
        assert isinstance(loaded, RMSThresholdDetector)
        assert loaded._is_fitted is True

    def test_build_detector_missing_model_raises(self, tmp_path):
        cfg = {
            "active": "rms_threshold",
            "models_dir": str(tmp_path),
        }
        with pytest.raises(FileNotFoundError, match="Model not found"):
            build_detector(cfg)

    def test_unknown_detector_raises(self, tmp_path):
        cfg = {"active": "quantum_detector", "models_dir": str(tmp_path)}
        with pytest.raises(ValueError, match="Unknown detector type"):
            build_detector(cfg)


class TestMQTTPublisher:
    @patch("paho.mqtt.client.Client")
    def test_publish_health_and_alert(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        pub = MQTTPublisher(broker="localhost", port=1883)
        pub._connected = True

        # Publish health
        pub.publish_health(score=0.123456, label="normal", extra={"detector": "rms"})
        assert mock_client.publish.called
        topic, payload_str = mock_client.publish.call_args[0]
        assert topic == "motor/health"
        payload = json.loads(payload_str)
        assert payload["score"] == 0.123456
        assert payload["label"] == "normal"
        assert payload["detector"] == "rms"

        # Publish alert
        pub.publish_alert(message="FAULT DETECTED", score=1.85)
        topic, payload_str = mock_client.publish.call_args[0]
        assert topic == "motor/alert"
        payload = json.loads(payload_str)
        assert payload["message"] == "FAULT DETECTED"
        assert payload["score"] == 1.85

    @patch("paho.mqtt.client.Client")
    def test_subscribe_config_and_receive_callback(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        received = []
        pub = MQTTPublisher(
            broker="localhost",
            topic_config="motor/config/threshold",
            on_config_message=lambda msg: received.append(msg),
        )

        # Simulate on_connect callback
        pub._on_connect(mock_client, None, None, 0)
        assert mock_client.subscribe.called
        subscribed_topics = [c[0][0] for c in mock_client.subscribe.call_args_list]
        assert "motor/config/threshold" in subscribed_topics

        # Simulate incoming config message
        mock_msg = MagicMock()
        mock_msg.topic = "motor/config/threshold"
        mock_msg.payload = b'{"threshold": 0.425, "reason": "false_alarm"}'
        pub._on_message(mock_client, None, mock_msg)

        assert len(received) == 1
        assert received[0]["threshold"] == 0.425


class TestMonitorServiceIntegration:
    """Test full MonitorService pipeline processing one window."""

    def test_monitor_service_process_one_window(self, tmp_path):
        # 1. Setup mock model
        det = RMSThresholdDetector(sigma=3.0)
        det.fit([{"rms": 0.25}, {"rms": 0.26}, {"rms": 0.24}])
        det_dir = tmp_path / "detectors"
        det_dir.mkdir()
        det.save(str(det_dir / "rms_threshold.pkl"))

        # 2. Setup mock data file
        raw_file = tmp_path / "healthy.npy"
        np.save(str(raw_file), np.random.randn(10_000).astype(np.float32) * 0.25)

        # 3. Setup test config.yaml
        config_path = tmp_path / "test_config.yaml"
        cfg_dict = {
            "sensor": {
                "type": "mock",
                "mock_data_path": str(raw_file),
                "sample_rate": 1000,
                "simulate_delay": False,
            },
            "motor": {
                "run_frequencies": {"medium": 40.0},
            },
            "detector": {
                "active": "rms_threshold",
                "models_dir": str(det_dir),
                "alert_threshold": 3.0,
            },
            "mqtt": {
                "broker": "localhost",
                "port": 1883,
                "topic_health": "test/health",
                "topic_alert": "test/alert",
            },
        }
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(cfg_dict, f)

        # 4. Instantiate MonitorService with mocked MQTT client
        with patch("paho.mqtt.client.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            service = MonitorService(config_path=str(config_path), force_mock=True)
            service.publisher._connected = True

            # Process a single window through the real pipeline
            service._process_one_window()

            # Verify MQTT received the health update
            assert mock_client.publish.called
            topic, payload_str = mock_client.publish.call_args[0]
            assert topic in ("test/health", "test/alert")
            payload = json.loads(payload_str)
            assert "score" in payload

    def test_monitor_service_remote_threshold_update(self, tmp_path):
        det = RMSThresholdDetector(sigma=3.0)
        det.fit([{"rms": 0.25}, {"rms": 0.26}, {"rms": 0.24}])
        det_dir = tmp_path / "detectors"
        det_dir.mkdir()
        det.save(str(det_dir / "rms_threshold.pkl"))

        raw_file = tmp_path / "healthy.npy"
        np.save(str(raw_file), np.random.randn(1000).astype(np.float32))

        config_path = tmp_path / "test_config.yaml"
        cfg_dict = {
            "sensor": {"type": "mock", "mock_data_path": str(raw_file), "sample_rate": 1000},
            "motor": {"run_frequencies": {"medium": 40.0}},
            "detector": {"active": "rms_threshold", "models_dir": str(det_dir), "alert_threshold": 1.0},
            "mqtt": {"broker": "localhost", "topic_health": "h", "topic_alert": "a", "topic_config": "c"},
        }
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(cfg_dict, f)

        with patch("paho.mqtt.client.Client"):
            service = MonitorService(config_path=str(config_path), force_mock=True)
            assert service.detector.threshold != 0.777

            # Simulate remote config payload delivery
            service._handle_remote_config({"threshold": 0.777, "reason": "human_feedback"})
            assert service.detector.threshold == 0.777
            assert service.cfg["detector"]["alert_threshold"] == 0.777

    def test_monitor_service_remote_sensor_mode_switch(self, tmp_path):
        det = RMSThresholdDetector(sigma=3.0)
        det.fit([{"rms": 0.25}, {"rms": 0.26}])
        det_dir = tmp_path / "detectors"
        det_dir.mkdir()
        det.save(str(det_dir / "rms_threshold.pkl"))

        raw_file = tmp_path / "healthy.npy"
        np.save(str(raw_file), np.random.randn(1000).astype(np.float32))

        config_path = tmp_path / "test_config.yaml"
        cfg_dict = {
            "sensor": {"type": "mock", "mock_data_path": str(raw_file), "sample_rate": 1000},
            "motor": {"run_frequencies": {"medium": 40.0}},
            "detector": {"active": "rms_threshold", "models_dir": str(det_dir), "alert_threshold": 1.0},
            "mqtt": {"broker": "localhost", "topic_health": "h", "topic_alert": "a", "topic_config": "c", "topic_sensor_config": "s"},
        }
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(cfg_dict, f)

        with patch("paho.mqtt.client.Client"):
            service = MonitorService(config_path=str(config_path), force_mock=True)

            # 1. Switch mode to mock explicitly
            success = service.set_sensor_mode("mock")
            assert success is True
            assert service.cfg["sensor"]["type"] == "mock"

            # 2. Simulate incoming MQTT sensor mode message
            service._handle_sensor_config({"sensor_mode": "mock", "reason": "web_ui_test"})
            assert service.cfg["sensor"]["type"] == "mock"
