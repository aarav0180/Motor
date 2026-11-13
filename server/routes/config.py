"""
server/routes/config.py — System settings and threshold management.
"""

from fastapi import APIRouter
from ..config import settings
from ..database import get_config, set_config
from ..mqtt_bridge import mqtt_bridge
from ..schemas import ConfigResponse, ThresholdUpdateRequest, SensorModeRequest

router = APIRouter(prefix="/api/v1/config", tags=["Configuration"])


@router.get("", response_model=ConfigResponse)
def get_system_config():
    """Retrieve active system parameters and detection thresholds."""
    current_thresh = float(get_config("alert_threshold", settings.default_threshold))
    current_sensor_mode = str(get_config("sensor_mode", settings.default_sensor_mode))
    return ConfigResponse(
        alert_threshold=current_thresh,
        sensor_mode=current_sensor_mode,
        mqtt_broker=settings.mqtt_broker,
        mqtt_port=settings.mqtt_port,
        topic_health=settings.topic_health,
        topic_alert=settings.topic_alert,
        topic_config=settings.topic_config,
        topic_sensor_config=settings.topic_sensor_config,
    )


@router.post("/threshold")
def update_threshold(req: ThresholdUpdateRequest):
    """Manually update the detection threshold and broadcast update to Edge daemon."""
    set_config("alert_threshold", req.threshold)
    broadcast_ok = mqtt_bridge.publish_threshold_update(
        new_threshold=req.threshold,
        reason=req.reason or "manual_calibration",
    )
    return {
        "status": "success",
        "new_threshold": req.threshold,
        "mqtt_broadcast": broadcast_ok,
    }


@router.post("/sensor-mode")
def set_sensor_mode(req: SensorModeRequest):
    """Switch active sensor mode between dataset replay and live physical I2C sensor."""
    mode = req.mode
    set_config("sensor_mode", mode)
    broadcast_ok = mqtt_bridge.publish_sensor_mode_update(
        mode=mode,
        reason=req.reason or "web_ui_toggle",
    )
    return {
        "status": "success",
        "sensor_mode": mode,
        "mqtt_broadcast": broadcast_ok,
        "message": f"Sensor mode updated to {mode}.",
    }
