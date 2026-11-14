"""
tests/test_server_api.py
------------------------
FastAPI integration tests for all Tier 2 REST endpoints.
"""

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from server.config import settings
from server.database import init_db, insert_telemetry, insert_alert
from server.main import app


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path):
    """Sets up an isolated test database for each API test."""
    db_file = str(tmp_path / "api_test.db")
    settings.database_path = db_file
    init_db(db_file)
    with patch("server.main.mqtt_bridge.start"), patch("server.main.mqtt_bridge.stop"), \
         patch("server.mqtt_bridge.mqtt_bridge.start"), patch("server.mqtt_bridge.mqtt_bridge.stop"), \
         patch("server.main.mock_streamer.start"), patch("server.main.mock_streamer.stop"):
        yield


@pytest.fixture
def client():
    """Provides a FastAPI test client."""
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_check_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "degraded")
        assert "uptime_seconds" in data
        assert "mqtt_connected" in data


class TestTelemetryEndpoints:
    def test_latest_telemetry_empty_returns_404(self, client):
        resp = client.get("/api/v1/telemetry/latest")
        assert resp.status_code == 404

    def test_latest_and_history_telemetry(self, client):
        insert_telemetry({
            "timestamp": 1000.0,
            "score": 0.003,
            "label": "normal",
            "detector": "hybrid_edge",
            "diagnosis": "healthy",
            "confidence": 0.99,
        })
        insert_telemetry({
            "timestamp": 1010.0,
            "score": 0.612,
            "label": "anomaly",
            "detector": "hybrid_edge",
            "diagnosis": "inner_race_fault",
            "confidence": 0.998,
        })

        # Test GET /latest
        resp = client.get("/api/v1/telemetry/latest")
        assert resp.status_code == 200
        latest = resp.json()
        assert latest["score"] == 0.612
        assert latest["diagnosis"] == "inner_race_fault"

        # Test GET /history
        resp_hist = client.get("/api/v1/telemetry/history?limit=10")
        assert resp_hist.status_code == 200
        hist = resp_hist.json()
        assert len(hist) == 2
        assert hist[0]["score"] == 0.003
        assert hist[1]["score"] == 0.612


class TestAlertsEndpoints:
    def test_alerts_listing_and_acknowledge(self, client):
        a_id = insert_alert({
            "message": "BEARING VIBRATION EXCEEDED",
            "score": 0.75,
            "diagnosis": "ball_fault",
        })

        # Test GET /alerts
        resp = client.get("/api/v1/alerts")
        assert resp.status_code == 200
        alerts = resp.json()
        assert len(alerts) == 1
        assert alerts[0]["acknowledged"] is False

        # Test POST /acknowledge
        resp_ack = client.post(f"/api/v1/alerts/{a_id}/acknowledge")
        assert resp_ack.status_code == 200

        # Verify unacknowledged filter
        resp_unacked = client.get("/api/v1/alerts?unacknowledged_only=true")
        assert len(resp_unacked.json()) == 0


class TestFeedbackEndpoints:
    def test_false_alarm_adapts_threshold_and_broadcasts(self, client):
        with patch("server.mqtt_bridge.mqtt_bridge.publish_threshold_update") as mock_pub:
            mock_pub.return_value = True

            payload = {
                "event": "FALSE_ALARM",
                "score": 0.048,
                "notes": "Testing threshold self-tuning",
            }
            resp = client.post("/api/v1/feedback", json=payload)
            assert resp.status_code == 200
            data = resp.json()

            assert data["status"] == "success"
            assert data["event"] == "FALSE_ALARM"
            expected_adapted = round(0.048 * 1.05, 4)
            assert data["adapted_threshold"] == expected_adapted
            assert data["mqtt_broadcast"] is True

            # Verify MQTT publish was called with the adapted threshold
            assert mock_pub.called
            args, kwargs = mock_pub.call_args
            assert kwargs["new_threshold"] == expected_adapted

    def test_confirmed_fault_logs_without_threshold_change(self, client):
        with patch("server.mqtt_bridge.mqtt_bridge.publish_threshold_update") as mock_pub:
            payload = {
                "event": "CONFIRMED_FAULT",
                "score": 0.825,
                "notes": "Bearing ball surface chipped",
            }
            resp = client.post("/api/v1/feedback", json=payload)
            assert resp.status_code == 200
            data = resp.json()

            assert data["event"] == "CONFIRMED_FAULT"
            assert data["adapted_threshold"] is None
            assert mock_pub.called is False


class TestConfigEndpoints:
    def test_get_and_update_config(self, client):
        with patch("server.mqtt_bridge.mqtt_bridge.publish_threshold_update") as mock_pub:
            mock_pub.return_value = True

            # Test GET /config
            resp = client.get("/api/v1/config")
            assert resp.status_code == 200
            cfg = resp.json()
            assert "alert_threshold" in cfg

            # Test POST /config/threshold
            update_payload = {"threshold": 0.088, "reason": "operator_tuning"}
            resp_post = client.post("/api/v1/config/threshold", json=update_payload)
            assert resp_post.status_code == 200
            assert resp_post.json()["new_threshold"] == 0.088

            # Verify GET reflects update
            resp_after = client.get("/api/v1/config")
            assert resp_after.json()["alert_threshold"] == 0.088

    def test_switch_sensor_mode_endpoint(self, client):
        with patch("server.mqtt_bridge.mqtt_bridge.publish_sensor_mode_update") as mock_pub:
            mock_pub.return_value = True

            # 1. Switch to physical mode
            resp = client.post("/api/v1/config/sensor-mode", json={"mode": "physical"})
            assert resp.status_code == 200
            assert resp.json()["sensor_mode"] == "physical"
            assert mock_pub.called
            assert mock_pub.call_args[1]["mode"] == "physical"

            # 2. Verify GET /config reflects new sensor mode
            cfg_resp = client.get("/api/v1/config")
            assert cfg_resp.status_code == 200
            assert cfg_resp.json()["sensor_mode"] == "physical"

            # 3. Invalid mode rejected by Pydantic validation
            bad_resp = client.post("/api/v1/config/sensor-mode", json={"mode": "invalid_mode"})
            assert bad_resp.status_code == 422

    def test_switch_model_mode_endpoint(self, client):
        with patch("server.mqtt_bridge.mqtt_bridge.publish_model_mode_update") as mock_pub:
            mock_pub.return_value = True

            # 1. Switch to supervised classifier
            resp = client.post("/api/v1/config/model", json={"model": "supervised"})
            assert resp.status_code == 200
            assert resp.json()["active_model"] == "supervised"
            assert mock_pub.called
            assert mock_pub.call_args[1]["model"] == "supervised"

            # 2. Verify GET /config reflects new model mode
            cfg_resp = client.get("/api/v1/config")
            assert cfg_resp.status_code == 200
            assert cfg_resp.json()["active_model"] == "supervised"

            # 3. Invalid model mode rejected
            bad_resp = client.post("/api/v1/config/model", json={"model": "unsupported_detector"})
            assert bad_resp.status_code == 422
