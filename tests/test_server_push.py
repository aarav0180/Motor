"""
tests/test_server_push.py — Test Web Push subscriptions and notification dispatch.
"""

import json
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from server.config import settings
from server.database import (
    add_push_subscription,
    count_push_subscriptions,
    get_all_push_subscriptions,
    init_db,
    remove_push_subscription,
)
from server.main import app
from server.push_notifier import PushNotifier, push_notifier


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    """Provides an isolated test database for each push test."""
    db_file = str(tmp_path / "test_push.db")
    settings.database_path = db_file
    init_db(db_file)
    with patch("server.main.mqtt_bridge.start"), patch("server.main.mqtt_bridge.stop"), \
         patch("server.mqtt_bridge.mqtt_bridge.start"), patch("server.mqtt_bridge.mqtt_bridge.stop"), \
         patch("server.main.mock_streamer.start"), patch("server.main.mock_streamer.stop"):
        yield db_file


@pytest.fixture
def test_client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


class TestPushDatabase:
    """Test push subscription CRUD operations in database."""

    def test_add_and_get_push_subscription(self, temp_db):
        endpoint = "https://fcm.googleapis.com/fcm/send/fake_token_123"
        p256dh = "BCX1234567890abcdef"
        auth = "secretAuth123"

        sub_id = add_push_subscription(
            endpoint=endpoint,
            p256dh=p256dh,
            auth=auth,
            client_id="technician_phone_1",
            db_path=temp_db,
        )
        assert sub_id >= 0
        assert count_push_subscriptions(db_path=temp_db) == 1

        subs = get_all_push_subscriptions(db_path=temp_db)
        assert len(subs) == 1
        assert subs[0]["endpoint"] == endpoint
        assert subs[0]["p256dh"] == p256dh
        assert subs[0]["auth"] == auth
        assert subs[0]["client_id"] == "technician_phone_1"

    def test_duplicate_endpoint_updates_existing(self, temp_db):
        endpoint = "https://fcm.googleapis.com/fcm/send/fake_token_123"
        add_push_subscription(endpoint, "key1", "auth1", "device_a", db_path=temp_db)
        add_push_subscription(endpoint, "key2", "auth2", "device_b", db_path=temp_db)

        assert count_push_subscriptions(db_path=temp_db) == 1
        subs = get_all_push_subscriptions(db_path=temp_db)
        assert subs[0]["p256dh"] == "key2"
        assert subs[0]["client_id"] == "device_b"

    def test_remove_push_subscription(self, temp_db):
        endpoint = "https://fcm.googleapis.com/fcm/send/token_to_remove"
        add_push_subscription(endpoint, "k", "a", db_path=temp_db)
        assert count_push_subscriptions(db_path=temp_db) == 1

        removed = remove_push_subscription(endpoint, db_path=temp_db)
        assert removed is True
        assert count_push_subscriptions(db_path=temp_db) == 0

        # Removing non-existent returns False
        assert remove_push_subscription("non_existent", db_path=temp_db) is False


class TestPushNotifierService:
    """Test PushNotifier VAPID generation, encoding, and dispatching."""

    def test_vapid_key_generation_and_public_key_format(self):
        notifier = PushNotifier(subject="mailto:test@motor.local")
        pub_key = notifier.get_public_key()
        assert isinstance(pub_key, str)
        # Uncompressed P-256 public key is 65 bytes -> base64url is 87 chars without padding
        assert len(pub_key) == 87
        assert "=" not in pub_key

    @patch("pywebpush.webpush")
    def test_send_to_subscription_success(self, mock_webpush):
        mock_webpush.return_value = MagicMock(status_code=201)
        notifier = PushNotifier()

        sub = {
            "endpoint": "https://updates.push.services.mozilla.com/wpush/v2/test_sub",
            "p256dh": "dummy_p256dh_key",
            "auth": "dummy_auth_key",
        }
        payload = {"title": "Test Alert", "body": "Bearing defect"}

        success = notifier.send_to_subscription(sub, payload)
        assert success is True
        mock_webpush.assert_called_once()

    @patch("pywebpush.webpush")
    def test_send_to_expired_subscription_removes_from_db(self, mock_webpush, temp_db, monkeypatch):
        import pywebpush

        # Simulate 410 Gone response
        mock_resp = MagicMock(status_code=410)
        mock_webpush.side_effect = pywebpush.WebPushException("Subscription expired", response=mock_resp)

        endpoint = "https://fcm.googleapis.com/fcm/send/expired_token"
        add_push_subscription(endpoint, "k", "a", db_path=temp_db)
        assert count_push_subscriptions(db_path=temp_db) == 1

        notifier = PushNotifier()
        sub = {"endpoint": endpoint, "p256dh": "k", "auth": "a"}

        # Patch remove_push_subscription to use temp_db
        with patch("server.push_notifier.remove_push_subscription") as mock_remove:
            success = notifier.send_to_subscription(sub, {"title": "test"})
            assert success is False
            mock_remove.assert_called_once_with(endpoint)

    def test_broadcast_alert_queues_all_subscribers(self, temp_db):
        notifier = PushNotifier()
        add_push_subscription("https://ep1", "k1", "a1", db_path=temp_db)
        add_push_subscription("https://ep2", "k2", "a2", db_path=temp_db)

        with patch.object(notifier, "send_to_subscription") as mock_send:
            queued = notifier.broadcast_alert(
                {"score": 0.892, "message": "High Vibration", "diagnosis": "ball_fault"},
                db_path=temp_db,
            )
            assert queued == 2


class TestPushApiEndpoints:
    """Test REST API routes for push notifications."""

    def test_get_vapid_public_key(self, test_client):
        res = test_client.get("/api/v1/push/key")
        assert res.status_code == 200
        data = res.json()
        assert "public_key" in data
        assert len(data["public_key"]) == 87
        assert "subject" in data

    def test_subscribe_and_unsubscribe_flow(self, test_client):
        endpoint = "https://fcm.googleapis.com/fcm/send/test_client_token_456"
        sub_payload = {
            "endpoint": endpoint,
            "keys": {
                "p256dh": "test_p256dh_value",
                "auth": "test_auth_value",
            },
            "client_id": "test_technician_browser",
        }

        # 1. Subscribe
        sub_res = test_client.post("/api/v1/push/subscribe", json=sub_payload)
        assert sub_res.status_code == 201
        assert sub_res.json()["status"] == "success"

        # 2. List subscriptions
        list_res = test_client.get("/api/v1/push/subscriptions")
        assert list_res.status_code == 200
        items = [s for s in list_res.json() if s["endpoint"] == endpoint]
        assert len(items) == 1
        assert items[0]["client_id"] == "test_technician_browser"

        # 3. Unsubscribe
        unsub_res = test_client.post("/api/v1/push/unsubscribe", json={"endpoint": endpoint})
        assert unsub_res.status_code == 200
        assert unsub_res.json()["status"] == "success"

        # 4. Verify gone
        list_res2 = test_client.get("/api/v1/push/subscriptions")
        assert not any(s["endpoint"] == endpoint for s in list_res2.json())

    def test_unsubscribe_nonexistent_returns_404(self, test_client):
        res = test_client.post("/api/v1/push/unsubscribe", json={"endpoint": "https://unknown"})
        assert res.status_code == 404

    def test_send_test_push_endpoint(self, test_client):
        res = test_client.post("/api/v1/push/test")
        assert res.status_code == 200
        data = res.json()
        assert "status" in data
