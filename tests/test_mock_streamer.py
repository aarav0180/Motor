"""
tests/test_mock_streamer.py — Tests for Cloud Edge Mock Streamer.
"""

from unittest.mock import MagicMock, patch
import json
import pytest

from server.config import settings
from server.mock_streamer import MockEdgeStreamer


class TestMockEdgeStreamer:
    """Verifies edge mock simulator configuration, MQTT handling, and state transitions."""

    def test_initial_state(self):
        streamer = MockEdgeStreamer(broker="test.broker", port=1883)
        assert streamer.running is False
        assert streamer.active is True
        assert streamer.active_model in ("unsupervised", "supervised")

    def test_handle_sensor_mode_config(self):
        streamer = MockEdgeStreamer()
        streamer.active = True

        # Switch to physical mode -> streamer should pause
        payload = json.dumps({"sensor_mode": "physical", "reason": "test"}).encode()
        streamer._handle_mqtt_config(settings.topic_sensor_config, payload)
        assert streamer.active is False

        # Switch back to mock mode -> streamer should resume
        payload = json.dumps({"sensor_mode": "mock", "reason": "test"}).encode()
        streamer._handle_mqtt_config(settings.topic_sensor_config, payload)
        assert streamer.active is True

    def test_handle_model_mode_config(self):
        streamer = MockEdgeStreamer()
        streamer.active_model = "unsupervised"

        # Switch to supervised
        payload = json.dumps({"model": "supervised", "reason": "test"}).encode()
        streamer._handle_mqtt_config(settings.topic_model_config, payload)
        assert streamer.active_model == "supervised"

        # Switch to unsupervised
        payload = json.dumps({"model": "unsupervised", "reason": "test"}).encode()
        streamer._handle_mqtt_config(settings.topic_model_config, payload)
        assert streamer.active_model == "unsupervised"

    def test_handle_threshold_config(self):
        streamer = MockEdgeStreamer()
        streamer.active_threshold = 0.05

        payload = json.dumps({"threshold": 0.12345, "reason": "false_alarm"}).encode()
        streamer._handle_mqtt_config(settings.topic_config, payload)
        assert streamer.active_threshold == 0.12345

    def test_init_components(self):
        streamer = MockEdgeStreamer()
        streamer._init_components()
        assert streamer._sensor is not None
        assert streamer._extractor is not None
