"""
server/routes/telemetry.py — Endpoints for querying motor vibration telemetry.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from ..database import get_latest_telemetry, get_telemetry_history
from ..schemas import TelemetryRecord

router = APIRouter(prefix="/api/v1/telemetry", tags=["Telemetry"])


@router.get("/latest", response_model=TelemetryRecord)
def get_latest():
    """Retrieve the most recent vibration telemetry window from the edge."""
    record = get_latest_telemetry()
    if not record:
        raise HTTPException(status_code=404, detail="No telemetry recorded yet from Edge device.")
    return TelemetryRecord(**record)


@router.get("/history", response_model=List[TelemetryRecord])
def get_history(
    limit: int = Query(default=60, ge=1, le=1000, description="Max number of points to retrieve"),
    start: Optional[float] = Query(default=None, description="Start timestamp (UNIX epoch)"),
    end: Optional[float] = Query(default=None, description="End timestamp (UNIX epoch)"),
):
    """Retrieve historical telemetry records for rolling graphs and trend analysis."""
    records = get_telemetry_history(limit=limit, start=start, end=end)
    return [TelemetryRecord(**r) for r in records]
