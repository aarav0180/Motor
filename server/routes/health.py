"""
server/routes/health.py — Server status and edge connectivity heartbeat.
"""

import time
from fastapi import APIRouter
from ..config import settings
from ..database import get_db
from ..mqtt_bridge import mqtt_bridge
from ..schemas import ServerHealthResponse

router = APIRouter(tags=["Health & Status"])

SERVER_START_TIME = time.time()


@router.get("/health", response_model=ServerHealthResponse)
def get_server_health():
    """Returns runtime health, MQTT status, and edge heartbeat."""
    db_ok = True
    try:
        with get_db() as conn:
            conn.execute("SELECT 1;").fetchone()
    except Exception:
        db_ok = False

    last_hb = mqtt_bridge.last_heartbeat
    now = time.time()
    hb_delta = round(now - last_hb, 1) if last_hb else None
    edge_online = (hb_delta is not None and hb_delta < 30.0)

    return ServerHealthResponse(
        status="healthy" if db_ok else "degraded",
        mqtt_connected=mqtt_bridge.is_connected,
        db_connected=db_ok,
        edge_online=edge_online,
        last_edge_heartbeat_s=hb_delta,
        uptime_seconds=round(now - SERVER_START_TIME, 1),
    )
