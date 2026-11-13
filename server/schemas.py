"""
server/schemas.py — Pydantic V2 data validation contracts for Tier 2 API.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class TelemetryRecord(BaseModel):
    """Telemetry data point emitted by Edge Daemon and stored in DB."""
    id: Optional[int] = None
    timestamp: float = Field(..., description="UNIX epoch timestamp in seconds")
    score: float = Field(..., description="Anomaly reconstruction error or probability")
    label: str = Field(..., description="'normal' or 'anomaly'")
    detector: str = Field(default="autoencoder", description="Active detector name")
    diagnosis: str = Field(default="healthy", description="Mechanical defect or 'healthy'")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Diagnostic confidence")
    proc_time_s: float = Field(default=0.0, description="Processing latency on edge CPU")


class AlertRecord(BaseModel):
    """Fault alert emitted on threshold breach."""
    id: Optional[int] = None
    timestamp: float
    message: str
    score: float
    diagnosis: str = "anomaly"
    acknowledged: bool = False
    ack_timestamp: Optional[float] = None


class FeedbackRequest(BaseModel):
    """Human-in-the-Loop technician feedback submission."""
    event: Literal["FALSE_ALARM", "CONFIRMED_FAULT"] = Field(
        ..., description="'FALSE_ALARM' triggers self-tuning; 'CONFIRMED_FAULT' logs fault for retraining"
    )
    score: float = Field(..., description="Anomaly score that triggered the alert")
    notes: Optional[str] = Field(default=None, description="Optional technician notes or repair details")


class FeedbackResponse(BaseModel):
    """Result of processing human feedback."""
    status: str = "success"
    event: str
    score: float
    adapted_threshold: Optional[float] = None
    mqtt_broadcast: bool = False
    message: str


class ThresholdUpdateRequest(BaseModel):
    """Direct manual threshold adjustment request."""
    threshold: float = Field(..., gt=0.0, description="New alert threshold")
    reason: Optional[str] = Field(default="manual_calibration", description="Reason for calibration")


class SensorModeRequest(BaseModel):
    """Request to switch active sensor between dataset mock and physical I2C."""
    mode: Literal["mock", "physical"] = Field(
        ..., description="'mock' for dataset replay, 'physical' for live ADXL345/MPU6050 I2C sensor"
    )
    reason: Optional[str] = Field(default="web_ui_toggle", description="Reason for mode switch")


class ConfigResponse(BaseModel):
    """Current system parameters."""
    alert_threshold: float
    sensor_mode: str = "mock"
    mqtt_broker: str
    mqtt_port: int
    topic_health: str
    topic_alert: str
    topic_config: str
    topic_sensor_config: str = "motor/config/sensor"


class ServerHealthResponse(BaseModel):
    """Server heartbeat and status."""
    status: str
    mqtt_connected: bool
    db_connected: bool
    edge_online: bool
    last_edge_heartbeat_s: Optional[float] = None
    uptime_seconds: float


class WebSocketMessage(BaseModel):
    """Payload pushed over WebSocket stream."""
    type: Literal["TELEMETRY_UPDATE", "ALERT", "CONFIG_UPDATE"]
    data: Dict[str, Any]


class PushSubscriptionKeys(BaseModel):
    """Standard W3C Push API client key pair."""
    p256dh: str = Field(..., description="Client P-256 public encryption key in base64")
    auth: str = Field(..., description="Client authentication secret in base64")


class PushSubscriptionPayload(BaseModel):
    """Payload sent by web browser / PWA to subscribe to push alerts."""
    endpoint: str = Field(..., description="Browser push service endpoint URL")
    keys: PushSubscriptionKeys
    client_id: Optional[str] = Field(default=None, description="Optional technician/device identifier")


class PushSubscriptionRecord(BaseModel):
    """Registered push subscriber representation."""
    id: int
    endpoint: str
    client_id: Optional[str] = None
    created_at: float


class VapidPublicKeyResponse(BaseModel):
    """VAPID public key needed by browser pushManager.subscribe()."""
    public_key: str
    subject: str
