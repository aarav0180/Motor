"""
server/routes/alerts.py — Fault alerts log and technician acknowledgment.
"""

from typing import List
from fastapi import APIRouter, HTTPException, Query
from ..database import get_alerts, acknowledge_alert
from ..schemas import AlertRecord

router = APIRouter(prefix="/api/v1/alerts", tags=["Alerts"])


@router.get("", response_model=List[AlertRecord])
def list_alerts(
    unacknowledged_only: bool = Query(default=False, description="Filter only active unreviewed alerts"),
    limit: int = Query(default=20, ge=1, le=200, description="Max alerts to return"),
):
    """Retrieve fault alert history."""
    alerts = get_alerts(unacknowledged_only=unacknowledged_only, limit=limit)
    return [AlertRecord(**a) for a in alerts]


@router.post("/{alert_id}/acknowledge")
def ack_alert(alert_id: int):
    """Mark an alert as reviewed by the technician."""
    success = acknowledge_alert(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Alert with ID {alert_id} not found.")
    return {"status": "success", "message": f"Alert {alert_id} marked as acknowledged."}
