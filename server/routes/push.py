"""
server/routes/push.py — Web Push notification subscription and test endpoints.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..database import (
    add_push_subscription,
    count_push_subscriptions,
    get_all_push_subscriptions,
    remove_push_subscription,
)
from ..push_notifier import push_notifier
from ..schemas import (
    PushSubscriptionPayload,
    PushSubscriptionRecord,
    VapidPublicKeyResponse,
)

router = APIRouter(prefix="/api/v1/push", tags=["Push Notifications"])


class UnsubscribePayload(BaseModel):
    endpoint: str


@router.get("/key", response_model=VapidPublicKeyResponse)
def get_vapid_public_key():
    """
    Return the server's VAPID public key.
    The web client/PWA converts this base64url string to Uint8Array and passes it to:
    navigator.serviceWorker.ready.then(reg => reg.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: ...
    }))
    """
    public_key = push_notifier.get_public_key()
    return VapidPublicKeyResponse(public_key=public_key, subject=push_notifier.subject)


@router.post("/subscribe", status_code=201)
def subscribe_device(payload: PushSubscriptionPayload):
    """
    Register a browser/mobile device token for background push alerts.
    """
    sub_id = add_push_subscription(
        endpoint=payload.endpoint,
        p256dh=payload.keys.p256dh,
        auth=payload.keys.auth,
        client_id=payload.client_id,
    )
    return {
        "status": "success",
        "subscription_id": sub_id,
        "message": "Device successfully subscribed to real-time vibration alerts.",
    }


@router.post("/unsubscribe")
def unsubscribe_device(payload: UnsubscribePayload):
    """
    Deregister an endpoint when user turns off notifications or unregisters.
    """
    removed = remove_push_subscription(payload.endpoint)
    if not removed:
        raise HTTPException(status_code=404, detail="Subscription endpoint not found.")
    return {"status": "success", "message": "Device unsubscribed."}


@router.get("/subscriptions", response_model=List[PushSubscriptionRecord])
def list_subscriptions():
    """List registered devices/endpoints (omitting private encryption keys)."""
    subs = get_all_push_subscriptions()
    return [
        PushSubscriptionRecord(
            id=s["id"],
            endpoint=s["endpoint"],
            client_id=s.get("client_id"),
            created_at=s["created_at"],
        )
        for s in subs
    ]


@router.post("/test")
def send_test_push():
    """
    Trigger a simulated alert push notification to all subscribed devices.
    Used by technicians to test device sound, vibration, and banner display.
    """
    count = count_push_subscriptions()
    if count == 0:
        return {
            "status": "warning",
            "message": "No devices are currently subscribed to push notifications.",
            "subscribers_queued": 0,
        }

    simulated_alert = {
        "score": 0.885,
        "message": "TEST: Simulated bearing fault alert",
        "diagnosis": "inner_race_fault",
        "ts": 0.0,
    }
    queued = push_notifier.broadcast_alert(simulated_alert)
    return {
        "status": "success",
        "message": f"Test push alert dispatched to {queued} device(s).",
        "subscribers_queued": queued,
    }
