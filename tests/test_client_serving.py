"""
tests/test_client_serving.py — Tests for Tier 3 Client web asset delivery & static mounting.
"""

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from server.config import settings
from server.database import init_db
from server.main import app


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path):
    """Provides an isolated database for client serving tests."""
    db_file = str(tmp_path / "client_test.db")
    settings.database_path = db_file
    init_db(db_file)
    with patch("server.main.mqtt_bridge.start"), patch("server.main.mqtt_bridge.stop"), \
         patch("server.mqtt_bridge.mqtt_bridge.start"), patch("server.mqtt_bridge.mqtt_bridge.stop"), \
         patch("server.main.mock_streamer.start"), patch("server.main.mock_streamer.stop"):
        yield db_file


@pytest.fixture
def client():
    return TestClient(app)


class TestClientServing:
    """Verifies that FastAPI properly serves all Tier 3 Client files."""

    def test_root_index_html_serving(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        assert "VIBRA-GUARD" in res.text
        assert "telemetryChart" in res.text
        assert "hitlModal" in res.text

    def test_pwa_manifest_serving(self, client):
        res = client.get("/manifest.json")
        assert res.status_code == 200
        data = res.json()
        assert data["short_name"] == "VibraGuard"
        assert data["display"] == "standalone"

    def test_service_worker_serving(self, client):
        res = client.get("/sw.js")
        assert res.status_code == 200
        assert "notificationclick" in res.text
        assert "registration.showNotification" in res.text

    def test_css_styles_serving(self, client):
        res = client.get("/css/styles.css")
        assert res.status_code == 200
        assert "pulse-critical" in res.text

    def test_all_js_modules_served(self, client):
        modules = [
            "config.js",
            "api.js",
            "ws.js",
            "audio.js",
            "charts.js",
            "push.js",
            "app.js",
        ]
        for mod in modules:
            res = client.get(f"/js/{mod}")
            assert res.status_code == 200, f"Module {mod} failed with {res.status_code}"
            assert len(res.text) > 50

    def test_api_routes_retain_priority_over_static(self, client):
        # /health and /api/v1 routes must not be shadowed by static files
        res_health = client.get("/health")
        assert res_health.status_code == 200
        assert res_health.json()["status"] in ("healthy", "degraded")

        res_config = client.get("/api/v1/config")
        assert res_config.status_code == 200
        assert "alert_threshold" in res_config.json()
