"""
server/mqtt_bridge.py — MQTT Bridge worker between Edge (Pi) and Server.

Subscribes to:
  - `motor/health` : per-window telemetry from Edge
  - `motor/alert`  : threshold breach alerts from Edge

Publishes to:
  - `motor/config/threshold` : remote dynamic threshold adaptations (Human-in-the-Loop)
"""

import json
import logging
import threading
import time
from typing import Any, Dict, Optional

from .config import settings
from .database import insert_telemetry, insert_alert
from .push_notifier import push_notifier
from .ws_manager import ws_manager

logger = logging.getLogger("server.mqtt")


class MQTTBridge:
    """Threaded MQTT client connecting Tier 2 Server to Tier 1 Edge."""

    def __init__(
        self,
        broker: Optional[str] = None,
        port: Optional[int] = None,
        topic_health: Optional[str] = None,
        topic_alert: Optional[str] = None,
        topic_config: Optional[str] = None,
        topic_sensor_config: Optional[str] = None,
        client_id: str = "motor-cloud-server",
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: Optional[bool] = None,
    ) -> None:
        self.broker = broker or settings.mqtt_broker
        self.port = port or settings.mqtt_port
        self.topic_health = topic_health or settings.topic_health
        self.topic_alert = topic_alert or settings.topic_alert
        self.topic_config = topic_config or settings.topic_config
        self.topic_sensor_config = topic_sensor_config or settings.topic_sensor_config
        self.client_id = client_id
        self.username = username or settings.mqtt_username
        self.password = password or settings.mqtt_password
        self.use_tls = use_tls if use_tls is not None else settings.mqtt_use_tls

        self._connected = False
        self._last_heartbeat: Optional[float] = None
        self._thread: Optional[threading.Thread] = None

        try:
            import paho.mqtt.client as mqtt
            self._client = mqtt.Client(client_id=self.client_id)
            if self.username:
                self._client.username_pw_set(self.username, self.password)
            if self.use_tls:
                import ssl
                self._client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.on_message = self._on_message
        except ImportError:
            logger.warning("paho-mqtt not available; MQTTBridge running in mock mode.")
            self._client = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def last_heartbeat(self) -> Optional[float]:
        return self._last_heartbeat

    def start(self) -> None:
        """Start background MQTT subscriber in a daemon thread (non-blocking)."""
        if self._client is None:
            return

        def _worker():
            try:
                logger.info("Connecting MQTT Bridge to %s:%d ...", self.broker, self.port)
                self._client.connect(self.broker, self.port, keepalive=60)
                self._client.loop_start()
            except Exception as exc:
                logger.warning("MQTT Bridge could not connect to %s:%d (%s). Running in offline mode.", self.broker, self.port, exc)

        self._thread = threading.Thread(target=_worker, daemon=True, name="MQTTBridge-Worker")
        self._thread.start()

    def stop(self) -> None:
        """Cleanly stop the background MQTT subscriber."""
        if self._client is not None:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
            self._connected = False
            logger.info("MQTT Bridge stopped.")

    def publish_threshold_update(
        self,
        new_threshold: float,
        reason: str = "human_feedback",
    ) -> bool:
        """
        Broadcast an adapted threshold down to Tier 1 Edge daemon over MQTT.
        This closes the Human-in-the-Loop active feedback cycle.
        """
        payload = {
            "threshold": round(new_threshold, 6),
            "reason": reason,
            "ts": time.time(),
        }
        if self._client is None or not self._connected:
            logger.warning("MQTT not connected; could not publish threshold to %s", self.topic_config)
            return False

        try:
            info = self._client.publish(self.topic_config, json.dumps(payload), qos=1)
            info.wait_for_publish(timeout=2.0)
            logger.info(
                "Broadcasted threshold update down to Edge on %s: %.6f (reason=%s)",
                self.topic_config, new_threshold, reason,
            )
            # Also notify any connected WebSocket clients
            ws_manager.broadcast_sync({
                "type": "CONFIG_UPDATE",
                "data": {"alert_threshold": new_threshold, "reason": reason},
            })
            return True
        except Exception as exc:
            logger.error("Failed to publish threshold update: %s", exc)
            return False

    def publish_sensor_mode_update(
        self,
        mode: str,
        reason: str = "web_ui_toggle",
    ) -> bool:
        """
        Broadcast a sensor mode switch ('mock' vs 'physical') to Tier 1 Edge daemon over MQTT.
        Allows switching between dataset replay and live I2C accelerometer from the Web UI.
        """
        payload = {
            "sensor_mode": mode,
            "reason": reason,
            "ts": time.time(),
        }
        # Also notify any connected WebSocket clients immediately
        ws_manager.broadcast_sync({
            "type": "CONFIG_UPDATE",
            "data": {"sensor_mode": mode, "reason": reason},
        })

        if self._client is None or not self._connected:
            logger.warning("MQTT not connected; stored mode %s locally, but could not push to %s", mode, self.topic_sensor_config)
            return False

        try:
            info = self._client.publish(self.topic_sensor_config, json.dumps(payload), qos=1, retain=True)
            info.wait_for_publish(timeout=2.0)
            logger.info("Broadcasted sensor mode down to Edge on %s: %s", self.topic_sensor_config, mode)
            return True
        except Exception as exc:
            logger.error("Failed to publish sensor mode update: %s", exc)
            return False

    # ------------------------------------------------------------------
    # MQTT Callbacks
    # ------------------------------------------------------------------

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            self._connected = True
            logger.info("MQTT Bridge connected to %s:%d", self.broker, self.port)
            client.subscribe([(self.topic_health, 1), (self.topic_alert, 1)])
            logger.info("Subscribed to Edge topics: %s, %s", self.topic_health, self.topic_alert)
        else:
            self._connected = False
            logger.error("MQTT Bridge connection failed with code %d", rc)

    def _on_disconnect(self, client, userdata, rc) -> None:
        self._connected = False
        if rc != 0:
            logger.warning("MQTT Bridge disconnected unexpectedly (code %d). Retrying...", rc)

    def _on_message(self, client, userdata, msg) -> None:
        """Process incoming messages from the Edge."""
        try:
            raw_data = msg.payload.decode()
            payload = json.loads(raw_data)
        except Exception as exc:
            logger.warning("Invalid JSON payload from %s: %s", msg.topic, exc)
            return

        self._last_heartbeat = time.time()

        if msg.topic == self.topic_health:
            self._handle_health_message(payload)
        elif msg.topic == self.topic_alert:
            self._handle_alert_message(payload)

    def _handle_health_message(self, payload: Dict[str, Any]) -> None:
        """Ingest telemetry into DB, push to WebSockets, and update retained MQTT status."""
        try:
            insert_telemetry(payload)
            ws_manager.broadcast_sync({
                "type": "TELEMETRY_UPDATE",
                "data": payload,
            })
            # Publish retained status for newly connecting/reconnecting MQTT clients
            if self._client is not None and self._connected:
                try:
                    self._client.publish(
                        "motor/status",
                        json.dumps(payload),
                        qos=1,
                        retain=True,
                    )
                except Exception:
                    pass
        except Exception as exc:
            logger.error("Error storing telemetry: %s", exc)

    def _handle_alert_message(self, payload: Dict[str, Any]) -> None:
        """Ingest alert into DB, broadcast to WebSockets, and dispatch Web Push to sleeping apps."""
        try:
            insert_alert(payload)
            ws_manager.broadcast_sync({
                "type": "ALERT",
                "data": payload,
            })
            # Dispatch background push notifications to dormant / closed apps
            push_notifier.broadcast_alert(payload)
            logger.warning("EDGE ALERT RECEIVED & DISPATCHED: %s", payload.get("message"))
        except Exception as exc:
            logger.error("Error storing alert: %s", exc)


mqtt_bridge = MQTTBridge()
