"""
service/mqtt_publisher.py — Thin MQTT client wrapper.

Wraps paho-mqtt so the rest of the service never touches paho directly.
Responsible for:
  - Connecting to the broker (with retry)
  - Publishing JSON health payloads to `motor/health`
  - Publishing text alerts to `motor/alert`
  - Graceful disconnect on shutdown
"""

import json
import logging
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class MQTTPublisher:
    """
    Publishes motor health scores and fault alerts over MQTT,
    and optionally listens for remote configuration updates (Tier 1 <-> Tier 2).

    Parameters
    ----------
    broker : str
        Hostname or IP of the Mosquitto broker.
    port : int
        MQTT broker port (default 1883; use 8883 for TLS).
    topic_health : str
        Topic for per-window health payloads.
    topic_alert : str
        Topic for fault alert messages.
    topic_config : str, optional
        Topic for receiving remote configuration/threshold updates from Server.
    client_id : str
        MQTT client identifier (must be unique per broker connection).
    keepalive : int
        MQTT keepalive interval in seconds.
    on_config_message : callable, optional
        Callback invoked when a message arrives on topic_config: callback(dict_payload).
    """

    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        topic_health: str = "motor/health",
        topic_alert: str = "motor/alert",
        topic_config: Optional[str] = "motor/config/threshold",
        topic_sensor_config: Optional[str] = "motor/config/sensor",
        client_id: str = "motor-fault-detector",
        keepalive: int = 60,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = False,
        on_config_message: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_sensor_config_message: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.broker                   = broker
        self.port                     = port
        self.topic_health             = topic_health
        self.topic_alert              = topic_alert
        self.topic_config             = topic_config
        self.topic_sensor_config      = topic_sensor_config
        self.keepalive                = keepalive
        self.username                 = username
        self.password                 = password
        self.use_tls                  = use_tls
        self.on_config_message        = on_config_message
        self.on_sensor_config_message = on_sensor_config_message

        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise ImportError(
                "paho-mqtt is required.  Install with: pip install paho-mqtt"
            ) from exc

        self._client = mqtt.Client(client_id=client_id)
        if self.username:
            self._client.username_pw_set(self.username, self.password)
        if self.use_tls:
            import ssl
            self._client.tls_set(cert_reqs=ssl.CERT_REQUIRED)

        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message    = self._on_message
        self._connected = False

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self, retries: int = 5, retry_delay: float = 2.0) -> None:
        """Connect to the broker, retrying on failure."""
        for attempt in range(1, retries + 1):
            try:
                self._client.connect(self.broker, self.port, self.keepalive)
                self._client.loop_start()
                time.sleep(0.5)   # Allow on_connect callback to fire
                if self._connected:
                    return
            except Exception as exc:
                logger.warning(
                    "MQTT connect attempt %d/%d failed: %s", attempt, retries, exc
                )
                time.sleep(retry_delay)
        raise ConnectionError(
            f"Could not connect to MQTT broker at {self.broker}:{self.port} "
            f"after {retries} attempts."
        )

    def disconnect(self) -> None:
        """Cleanly disconnect from the broker."""
        self._client.loop_stop()
        self._client.disconnect()
        logger.info("MQTT disconnected from %s:%d", self.broker, self.port)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish_health(
        self,
        score: float,
        label: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Publish a health update payload.

        Parameters
        ----------
        score : float
            Anomaly score from the active detector.
        label : str
            'normal' or 'anomaly'.
        extra : dict, optional
            Additional fields merged into the payload (e.g. detector name).
        """
        payload: Dict[str, Any] = {
            "score": round(score, 6),
            "label": label,
            "ts":    time.time(),
        }
        if extra:
            payload.update(extra)
        self._publish(self.topic_health, payload)

    def publish_alert(self, message: str, score: Optional[float] = None) -> None:
        """
        Publish a fault alert.

        Parameters
        ----------
        message : str
            Human-readable alert description.
        score : float, optional
            The anomaly score that triggered the alert.
        """
        payload: Dict[str, Any] = {"message": message, "ts": time.time()}
        if score is not None:
            payload["score"] = round(score, 6)
        self._publish(self.topic_alert, payload)
        logger.warning("ALERT published: %s", message)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _publish(self, topic: str, payload: Dict[str, Any]) -> None:
        if not self._connected:
            logger.error("Cannot publish — not connected to broker.")
            return
        self._client.publish(topic, json.dumps(payload), qos=1)
        logger.debug("Published to %s: %s", topic, payload)

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            self._connected = True
            logger.info("MQTT connected to %s:%d", self.broker, self.port)
            if self.topic_config:
                self._client.subscribe(self.topic_config, qos=1)
                logger.info("Subscribed to remote config topic: %s", self.topic_config)
            if self.topic_sensor_config:
                self._client.subscribe(self.topic_sensor_config, qos=1)
                logger.info("Subscribed to sensor config topic: %s", self.topic_sensor_config)
        else:
            logger.error("MQTT connection failed with code %d", rc)

    def _on_message(self, client, userdata, msg) -> None:
        if self.topic_config and msg.topic == self.topic_config:
            try:
                raw_payload = msg.payload.decode()
                try:
                    payload = json.loads(raw_payload)
                except json.JSONDecodeError:
                    payload = {"threshold": float(raw_payload.strip())}
                logger.info("Received remote config on %s: %s", msg.topic, payload)
                if self.on_config_message:
                    self.on_config_message(payload)
            except Exception as exc:
                logger.error("Error processing config message from %s: %s", msg.topic, exc)

        elif self.topic_sensor_config and msg.topic == self.topic_sensor_config:
            try:
                raw_payload = msg.payload.decode()
                try:
                    payload = json.loads(raw_payload)
                except json.JSONDecodeError:
                    payload = {"sensor_mode": raw_payload.strip()}
                logger.info("Received remote sensor mode config on %s: %s", msg.topic, payload)
                if self.on_sensor_config_message:
                    self.on_sensor_config_message(payload)
            except Exception as exc:
                logger.error("Error processing sensor mode message from %s: %s", msg.topic, exc)

    def _on_disconnect(self, client, userdata, rc) -> None:
        self._connected = False
        if rc != 0:
            logger.warning("Unexpected MQTT disconnect (rc=%d)", rc)

    # Context manager support
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return f"MQTTPublisher({self.broker}:{self.port}, status={status})"
