"""
tests/test_server_mqtt_bridge.py
--------------------------------
Unit tests for the Tier 2 MQTT Bridge (ingestion and remote calibration dispatch).
"""

from unittest.mock import MagicMock, patch
import json
import pytest

from server.config import settings
from server.database import init_db, get_latest_telemetry, get_alerts
from server.mqtt_bridge import MQTTBridge


@pytest.fixture(autouse=True)
def setup_db(tmp_path):
    db_file = str(tmp_path / "mqtt_test.db")
    settings.database_path = db_file
    init_db(db_file)
    yield


class TestMQTTBridgeIngestion:
    @patch("paho.mqtt.client.Client")
    def test_bridge_connect_and_subscriptions(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        bridge = MQTTBridge(broker="test.broker", port=1883)
        bridge._on_connect(mock_client, None, None, 0)

        assert bridge.is_connected is True
        assert mock_client.subscribe.called
        subscribed_topics = mock_client.subscribe.call_args[0][0]
        assert (bridge.topic_health, 1) in subscribed_topics
        assert (bridge.topic_alert, 1) in subscribed_topics

    @patch("paho.mqtt.client.Client")
    def test_handle_health_message_persists_to_database(self, mock_client_cls):
        bridge = MQTTBridge()

        # Simulate incoming telemetry payload from Edge
        mock_msg = MagicMock()
        mock_msg.topic = bridge.topic_health
        payload = {
            "score": 0.6125,
            "label": "anomaly",
            "detector": "hybrid_edge",
            "diagnosis": "inner_race_fault",
            "confidence": 0.999,
            "ts": 1731550000.0,
        }
        mock_msg.payload = json.dumps(payload).encode()

        with patch("server.ws_manager.ws_manager.broadcast_sync") as mock_ws:
            bridge._on_message(None, None, mock_msg)

            # 1. Verify persisted in DB
            rec = get_latest_telemetry()
            assert rec is not None
            assert rec["score"] == 0.6125
            assert rec["diagnosis"] == "inner_race_fault"

            # 2. Verify pushed to WebSocket manager
            assert mock_ws.called
            ws_call_data = mock_ws.call_args[0][0]
            assert ws_call_data["type"] == "TELEMETRY_UPDATE"
            assert ws_call_data["data"]["score"] == 0.6125

    @patch("paho.mqtt.client.Client")
    def test_handle_alert_message_persists_to_database(self, mock_client_cls):
        bridge = MQTTBridge()

        mock_msg = MagicMock()
        mock_msg.topic = bridge.topic_alert
        payload = {
            "message": "FAULT DETECTED [BALL_FAULT]",
            "score": 0.804,
            "diagnosis": "ball_fault",
            "ts": 1731550010.0,
        }
        mock_msg.payload = json.dumps(payload).encode()

        with patch("server.ws_manager.ws_manager.broadcast_sync") as mock_ws, \
             patch("server.push_notifier.push_notifier.broadcast_alert") as mock_push:
            bridge._on_message(None, None, mock_msg)

            alerts = get_alerts(unacknowledged_only=True)
            assert len(alerts) == 1
            assert alerts[0]["score"] == 0.804
            assert alerts[0]["message"] == "FAULT DETECTED [BALL_FAULT]"

            assert mock_ws.called
            ws_call_data = mock_ws.call_args[0][0]
            assert ws_call_data["type"] == "ALERT"

            # 3. Verify Web Push notification broadcast triggered
            assert mock_push.called
            assert mock_push.call_args[0][0]["score"] == 0.804

    @patch("paho.mqtt.client.Client")
    def test_publish_threshold_update_sends_to_edge(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        bridge = MQTTBridge()
        bridge._connected = True

        success = bridge.publish_threshold_update(new_threshold=0.065, reason="human_test")
        assert success is True

        assert mock_client.publish.called
        topic, payload_str = mock_client.publish.call_args[0]
        assert topic == bridge.topic_config
        data = json.loads(payload_str)
        assert data["threshold"] == 0.065
        assert data["reason"] == "human_test"

    @patch("paho.mqtt.client.Client")
    def test_publish_sensor_mode_update_sends_to_edge(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        bridge = MQTTBridge()
        bridge._connected = True

        with patch("server.ws_manager.ws_manager.broadcast_sync") as mock_ws:
            success = bridge.publish_sensor_mode_update(mode="physical", reason="web_ui_test")
            assert success is True

            # Verify MQTT publish
            assert mock_client.publish.called
            topic, payload_str = mock_client.publish.call_args[0]
            assert topic == bridge.topic_sensor_config
            data = json.loads(payload_str)
            assert data["sensor_mode"] == "physical"
            assert data["reason"] == "web_ui_test"

            # Verify WebSocket broadcast to dashboards
            assert mock_ws.called
            ws_data = mock_ws.call_args[0][0]
            assert ws_data["type"] == "CONFIG_UPDATE"
            assert ws_data["data"]["sensor_mode"] == "physical"
