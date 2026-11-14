"""
server/config.py — Central configuration loader for Tier 2 Server.

Reads settings from project root `config.yaml` with environment variable overrides
for cloud deployments (e.g. Docker, Railway, Render).
"""

import os
from pathlib import Path
from typing import Any, Dict
import yaml

# Path discovery
SERVER_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVER_DIR.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


class ServerSettings:
    """Server runtime configuration."""

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        raw_cfg: Dict[str, Any] = {}
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    raw_cfg = yaml.safe_load(f) or {}
            except Exception:
                raw_cfg = {}

        mqtt_cfg = raw_cfg.get("mqtt", {})
        det_cfg = raw_cfg.get("detector", {})

        # MQTT Broker settings
        self.mqtt_broker: str = os.getenv("MQTT_BROKER", mqtt_cfg.get("broker", "localhost"))
        self.mqtt_port: int = int(os.getenv("MQTT_PORT", str(mqtt_cfg.get("port", 1883))))
        self.mqtt_username: Optional[str] = os.getenv("MQTT_USERNAME", mqtt_cfg.get("username", None))
        self.mqtt_password: Optional[str] = os.getenv("MQTT_PASSWORD", mqtt_cfg.get("password", None))
        self.mqtt_use_tls: bool = os.getenv("MQTT_USE_TLS", str(mqtt_cfg.get("use_tls", "false"))).lower() in ("true", "1", "yes")
        self.topic_health: str = os.getenv("MQTT_TOPIC_HEALTH", mqtt_cfg.get("topic_health", "motor/health"))
        self.topic_alert: str = os.getenv("MQTT_TOPIC_ALERT", mqtt_cfg.get("topic_alert", "motor/alert"))
        self.topic_config: str = os.getenv("MQTT_TOPIC_CONFIG", mqtt_cfg.get("topic_config", "motor/config/threshold"))
        self.topic_sensor_config: str = os.getenv("MQTT_TOPIC_SENSOR_CONFIG", mqtt_cfg.get("topic_sensor_config", "motor/config/sensor"))
        self.topic_model_config: str = os.getenv("MQTT_TOPIC_MODEL_CONFIG", mqtt_cfg.get("topic_model_config", "motor/config/model"))

        # Model settings
        self.default_model: str = os.getenv("ACTIVE_MODEL", "unsupervised")

        # Sensor settings
        sensor_cfg = raw_cfg.get("sensor", {})
        self.default_sensor_mode: str = os.getenv("SENSOR_MODE", sensor_cfg.get("type", "mock"))

        # Server HTTP / WS settings
        self.server_host: str = os.getenv("SERVER_HOST", "0.0.0.0")
        self.server_port: int = int(os.getenv("SERVER_PORT", "8000"))
        self.cors_origins: list = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",")]

        # Persistence
        default_db = str(REPO_ROOT / "data" / "server_telemetry.db")
        self.database_path: str = os.getenv("DATABASE_PATH", default_db)

        # Baseline threshold
        self.default_threshold: float = float(det_cfg.get("alert_threshold") or 0.05)


settings = ServerSettings()
