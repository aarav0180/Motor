"""
server/push_notifier.py — Web Push Notification Service (RFC 8030 / VAPID).

Dispatches push notifications to dormant, backgrounded, or closed web/mobile apps
when a bearing fault or vibration threshold breach is detected on the edge.
"""

import base64
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from py_vapid import Vapid
import pywebpush

from .database import (
    get_all_push_subscriptions,
    get_config,
    remove_push_subscription,
    set_config,
)

logger = logging.getLogger("server.push_notifier")


class PushNotifier:
    """
    Manages VAPID key pairs and asynchronous push notifications for client devices.
    """

    def __init__(self, subject: str = "mailto:admin@motor-monitor.local") -> None:
        self.subject = subject
        self._executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="push_worker")
        self._vapid: Optional[Vapid] = None
        self._public_key_b64: Optional[str] = None
        self._lock = threading.Lock()

    def _ensure_vapid(self) -> Vapid:
        """Lazily initialize or retrieve persistent VAPID key pair."""
        if self._vapid is not None:
            return self._vapid

        with self._lock:
            if self._vapid is not None:
                return self._vapid

            # 1. Check if private key is stored in system_config
            saved_pem = get_config("vapid_private_pem")
            if saved_pem:
                try:
                    self._vapid = Vapid.from_pem(saved_pem.encode())
                    logger.info("Loaded persistent VAPID key pair from system_config.")
                except Exception as exc:
                    logger.warning("Failed to parse saved VAPID PEM (%s); regenerating...", exc)
                    self._vapid = None

            # 2. If not found or failed, generate new key pair and persist
            if self._vapid is None:
                self._vapid = Vapid()
                self._vapid.generate_keys()
                pem_str = self._vapid.private_pem().decode()
                set_config("vapid_private_pem", pem_str)
                logger.info("Generated and persisted new VAPID key pair.")

            # Compute uncompressed 65-byte base64url public key for browser pushManager
            raw_bytes = self._vapid.public_key.public_bytes(
                Encoding.X962, PublicFormat.UncompressedPoint
            )
            self._public_key_b64 = base64.urlsafe_b64encode(raw_bytes).decode().rstrip("=")
            return self._vapid

    def get_public_key(self) -> str:
        """Return base64url P-256 public key for browser pushManager.subscribe()."""
        self._ensure_vapid()
        assert self._public_key_b64 is not None
        return self._public_key_b64

    def get_private_pem(self) -> str:
        """Return the private key PEM string."""
        vapid = self._ensure_vapid()
        return vapid.private_pem().decode()

    def send_to_subscription(
        self,
        subscription: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> bool:
        """
        Send a web push notification to a single registered subscription.
        Automatically cleans up expired or unsubscribed endpoints (HTTP 404/410).
        """
        endpoint = subscription.get("endpoint", "")
        p256dh = subscription.get("p256dh", "")
        auth = subscription.get("auth", "")

        if not endpoint or not p256dh or not auth:
            logger.warning("Invalid subscription record: missing required keys.")
            return False

        subscription_info = {
            "endpoint": endpoint,
            "keys": {
                "p256dh": p256dh,
                "auth": auth,
            },
        }

        vapid_private_pem = self.get_private_pem()
        data_str = json.dumps(payload)

        try:
            pywebpush.webpush(
                subscription_info=subscription_info,
                data=data_str,
                vapid_private_key=vapid_private_pem,
                vapid_claims={"sub": self.subject},
                timeout=5,
            )
            logger.debug("Web Push sent successfully to %s", endpoint[:30])
            return True
        except pywebpush.WebPushException as exc:
            # Check for HTTP 404 Not Found or 410 Gone (device unsubscribed / expired)
            status_code = getattr(exc.response, "status_code", None)
            if status_code in (404, 410):
                logger.info("Device unsubscribed or expired (HTTP %d). Removing %s", status_code, endpoint[:30])
                remove_push_subscription(endpoint)
            else:
                logger.warning("Web Push delivery failed to %s: %s", endpoint[:30], exc)
            return False
        except Exception as exc:
            logger.error("Unexpected error delivering Web Push to %s: %s", endpoint[:30], exc)
            return False

    def broadcast_alert(
        self,
        alert_payload: Dict[str, Any],
        db_path: Optional[str] = None,
    ) -> int:
        """
        Asynchronously broadcast a fault alert to all registered client devices.
        Returns the number of candidate subscriptions queued.
        """
        subscriptions = get_all_push_subscriptions(db_path=db_path)
        if not subscriptions:
            logger.debug("No push subscribers registered.")
            return 0

        score = float(alert_payload.get("score", 0.0))
        message = str(alert_payload.get("message", "Vibration anomaly detected"))
        diagnosis = str(alert_payload.get("diagnosis", "anomaly"))
        timestamp = float(alert_payload.get("ts") or alert_payload.get("timestamp") or 0.0)

        push_data = {
            "title": "🚨 MOTOR FAULT DETECTED",
            "body": f"Score: {score:.4f} — {message}",
            "icon": "/static/favicon.ico",
            "tag": "motor-fault-alert",
            "data": {
                "score": score,
                "diagnosis": diagnosis,
                "timestamp": timestamp,
                "url": "/alerts",
            },
        }

        # Queue deliveries in thread pool
        for sub in subscriptions:
            self._executor.submit(self.send_to_subscription, sub, push_data)

        logger.info("Queued push notifications for %d subscribed devices.", len(subscriptions))
        return len(subscriptions)


# Singleton push notifier instance
push_notifier = PushNotifier()
