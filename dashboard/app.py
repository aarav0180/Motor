"""
dashboard/app.py — Streamlit live monitoring dashboard (Local Dev Mock).

ARCHITECTURE NOTE:
------------------
In the decoupled 3-tier architecture:
  - Tier 1: Headless edge daemon on Raspberry Pi (`service.monitor`).
  - Tier 2: Server API (FastAPI / Mosquitto MQTT broker).
  - Tier 3: Client App (Web / Mobile frontend, deferred to next stage).

This Streamlit app currently serves as a local developer mock/test harness to
simulate Tier 3 and verify the Human-in-the-Loop feedback loop during development,
prior to building the full standalone web/mobile client and server API.

Subscribes to MQTT topics and shows:
  - Rolling anomaly score plot (last 60 windows)
  - Current label (NORMAL / ANOMALY) with colour indicator
  - Fault alert log
  - Human-in-the-Loop feedback buttons (Confirm Fault / False Alarm)
    which broadcast adapted thresholds back to the edge via MQTT.

Usage
-----
    streamlit run dashboard/app.py

Requires: streamlit, paho-mqtt, pandas, plotly
"""

import json
import queue
import threading
import time
from collections import deque
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import paho.mqtt.client as mqtt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BROKER         = "localhost"
PORT           = 1883
TOPIC_HEALTH   = "motor/health"
TOPIC_ALERT    = "motor/alert"
TOPIC_FEEDBACK = "motor/feedback"
TOPIC_CONFIG   = "motor/config/threshold"
FEEDBACK_FILE  = Path("data/feedback.json")
MAX_POINTS     = 60   # rolling window length in the plot


def _record_feedback(event: str, score: float) -> float:
    \"\"\"Log human feedback and broadcast adapted threshold to Edge via MQTT.\"\"\"
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if FEEDBACK_FILE.exists():
        try:
            with open(FEEDBACK_FILE, "r") as f:
                history = json.load(f)
        except Exception:
            history = []

    # If false alarm, adapt threshold with 5% safety margin
    new_thresh = round(score * 1.05, 4) if event == "FALSE_ALARM" else None

    entry = {
        "timestamp": time.time(),
        "time_str": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event": event,
        "score": round(score, 6),
        "adapted_threshold": new_thresh,
    }
    history.append(entry)
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(history, f, indent=2)

    # Broadcast remote config update to Tier 1 Edge daemon if threshold was adapted
    if new_thresh is not None:
        try:
            pub_client = mqtt.Client(client_id="dashboard-feedback-pub")
            pub_client.connect(BROKER, PORT, keepalive=10)
            payload = {
                "threshold": new_thresh,
                "reason": "human_false_alarm",
                "ts": time.time(),
            }
            pub_client.publish(TOPIC_CONFIG, json.dumps(payload), qos=1)
            pub_client.disconnect()
        except Exception:
            pass

    return new_thresh or score

# ---------------------------------------------------------------------------
# Shared state between MQTT thread and Streamlit
# ---------------------------------------------------------------------------
_health_queue: "queue.Queue[dict]" = queue.Queue()
_alert_queue:  "queue.Queue[str]"  = queue.Queue()


# ---------------------------------------------------------------------------
# MQTT subscriber (runs in a background thread)
# ---------------------------------------------------------------------------

def _on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
    except json.JSONDecodeError:
        return
    if msg.topic == TOPIC_HEALTH:
        _health_queue.put(payload)
    elif msg.topic == TOPIC_ALERT:
        _alert_queue.put(payload.get("message", str(payload)))


def _start_mqtt_thread():
    """Launch MQTT subscriber in a daemon thread (started once per session)."""
    client = mqtt.Client(client_id="dashboard-subscriber")
    client.on_message = _on_message
    client.connect(BROKER, PORT, keepalive=60)
    client.subscribe([(TOPIC_HEALTH, 1), (TOPIC_ALERT, 1)])
    client.loop_start()


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Motor Fault Monitor",
        page_icon="⚙️",
        layout="wide",
    )
    st.title("⚙️ Motor Fault Detection — Live Dashboard")

    # Start MQTT thread once per Streamlit session
    if "mqtt_started" not in st.session_state:
        _start_mqtt_thread()
        st.session_state.mqtt_started = True
        st.session_state.scores  = deque(maxlen=MAX_POINTS)
        st.session_state.times   = deque(maxlen=MAX_POINTS)
        st.session_state.labels  = deque(maxlen=MAX_POINTS)
        st.session_state.alerts  = []

    # Drain incoming messages
    while not _health_queue.empty():
        h = _health_queue.get()
        st.session_state.scores.append(h.get("score", 0.0))
        st.session_state.times.append(
            pd.to_datetime(h.get("ts", time.time()), unit="s")
        )
        st.session_state.labels.append(h.get("label", "?"))

    while not _alert_queue.empty():
        st.session_state.alerts.append(
            f"{pd.Timestamp.now().strftime('%H:%M:%S')} — {_alert_queue.get()}"
        )

    # --- Layout ---
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Anomaly Score (last 60 windows)")
        if st.session_state.scores:
            fig = go.Figure()
            colours = [
                "red" if l == "anomaly" else "steelblue"
                for l in st.session_state.labels
            ]
            fig.add_trace(go.Scatter(
                x=list(st.session_state.times),
                y=list(st.session_state.scores),
                mode="lines+markers",
                marker=dict(color=colours, size=8),
                line=dict(color="steelblue"),
                name="Score",
            ))
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Anomaly Score",
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Waiting for data from the monitor service …")

    with col2:
        st.subheader("Current Status")
        if st.session_state.labels:
            last_label = st.session_state.labels[-1]
            last_score = st.session_state.scores[-1]
            if last_label == "anomaly":
                st.error(f"🔴 ANOMALY\nScore: {last_score:.4f}")
            else:
                st.success(f"🟢 NORMAL\nScore: {last_score:.4f}")
        else:
            st.info("—")

        st.subheader("Alert Log")
        alerts = st.session_state.alerts[-10:]   # last 10
        if alerts:
            for a in reversed(alerts):
                st.warning(a)
        else:
            st.caption("No alerts.")

        # --- Human-in-the-Loop Active Feedback ---
        st.markdown("---")
        st.subheader("👨‍🔧 Human-in-the-Loop Feedback")
        if st.session_state.labels and st.session_state.labels[-1] == "anomaly":
            st.info(f"Active Alert Score: **{st.session_state.scores[-1]:.4f}**")
            fb_c1, fb_c2 = st.columns(2)
            with fb_c1:
                if st.button("✅ Confirm Fault", use_container_width=True):
                    _record_feedback("CONFIRMED_FAULT", st.session_state.scores[-1])
                    st.success("Logged as confirmed mechanical defect!")
            with fb_c2:
                if st.button("❌ False Alarm", use_container_width=True):
                    new_t = _record_feedback("FALSE_ALARM", st.session_state.scores[-1])
                    st.warning(f"Threshold self-tuned to {new_t:.4f} to suppress false alarm.")
        else:
            st.caption("No active alerts awaiting technician confirmation.")

    # Auto-refresh every 5 seconds
    time.sleep(5)
    st.rerun()


if __name__ == "__main__":
    main()
