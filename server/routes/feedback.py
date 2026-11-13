"""
server/routes/feedback.py — Human-in-the-Loop feedback & dynamic threshold calibration.
"""

from typing import Any, Dict, List
from fastapi import APIRouter
from ..database import insert_feedback, get_feedback_history, set_config
from ..mqtt_bridge import mqtt_bridge
from ..schemas import FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/api/v1/feedback", tags=["Human-in-the-Loop Feedback"])


@router.post("", response_model=FeedbackResponse)
def submit_feedback(req: FeedbackRequest):
    """
    Submit technician feedback on an anomaly alert.

    - FALSE_ALARM: Computes adapted threshold = score * 1.05, saves it,
      and immediately broadcasts the new threshold down to Tier 1 Edge via MQTT.
    - CONFIRMED_FAULT: Records genuine defect in DB for maintenance logs and retraining.
    """
    new_thresh = None
    broadcast_ok = False

    if req.event == "FALSE_ALARM":
        new_thresh = round(req.score * 1.05, 4)
        # Update server config state
        set_config("alert_threshold", new_thresh)
        # Broadcast remote config down to the edge daemon via MQTT
        broadcast_ok = mqtt_bridge.publish_threshold_update(
            new_threshold=new_thresh,
            reason="human_false_alarm",
        )
        msg = f"False alarm recorded. Threshold self-tuned to {new_thresh:.4f} and pushed to Edge."
    else:
        msg = "Confirmed fault recorded in maintenance log for future model retraining."

    # Record in feedback audit log
    insert_feedback({
        "event": req.event,
        "score": req.score,
        "adapted_threshold": new_thresh,
        "notes": req.notes,
    })

    return FeedbackResponse(
        status="success",
        event=req.event,
        score=req.score,
        adapted_threshold=new_thresh,
        mqtt_broadcast=broadcast_ok,
        message=msg,
    )


@router.get("", response_model=List[Dict[str, Any]])
def list_feedback_history():
    """Retrieve technician feedback audit log."""
    return get_feedback_history(limit=50)
