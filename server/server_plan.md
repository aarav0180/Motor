# 🏛️ Tier 2: Server / Backend Architecture & Build Plan

> **Role**: Central communication, persistence, and coordination tier in the 3-Tier Decoupled IoT Architecture.
> Bridges the **Headless Edge Daemon (Tier 1 on Raspberry Pi)** with the **Remote Client Application (Tier 3 on Web/Mobile)**.

---

## 1. System Context & Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TIER 1: EDGE (Raspberry Pi 4)                         │
│                                                                             │
│  [ADXL345 / MPU6050] ──> [service/monitor.py] ──> [TorchScript 1D-CNN]     │
│                                  │                        ▲                 │
│         telemetry (motor/health) │                        │ remote config   │
│         alerts    (motor/alert)  │                        │ (motor/config/  │
│                                  │                        │  threshold)     │
└──────────────────────────────────┼────────────────────────┼─────────────────┘
                                   │ MQTT (QoS 1)           │ MQTT (QoS 1)
                                   ▼                        │
┌───────────────────────────────────────────────────────────┴─────────────────┐
│                    TIER 2: SERVER / BACKEND (FastAPI + MQTT)                │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        MQTT Bridge Service                          │   │
│   │   - Ingests `motor/health` & `motor/alert`                          │   │
│   │   - Publishes adapted thresholds to `motor/config/threshold`        │   │
│   └───────────────┬─────────────────────────────────────▲───────────────┘   │
│                   │                                     │                   │
│                   ▼                                     │                   │
│   ┌───────────────────────────────┐     ┌───────────────┴───────────────┐   │
│   │   Time-Series SQLite / WAL    │     │      Event Broadcaster        │   │
│   │   (Telemetry, Alerts, Feedbk) │     │      (In-Memory Queue/Bus)    │   │
│   └───────────────┬───────────────┘     └───────────────┬───────────────┘   │
│                   │                                     │                   │
│                   ▼                                     ▼                   │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      FastAPI Application Engine                     │   │
│   │                                                                     │   │
│   │   REST API (/api/v1)                    WebSocket Stream (/ws)      │   │
│   │   • GET  /telemetry/latest              • WS /telemetry             │   │
│   │   • GET  /telemetry/history               (Pushes live snapshots    │   │
│   │   • GET  /alerts                          to connected apps <10ms)  │   │
│   │   • POST /alerts/{id}/ack                                           │   │
│   │   • POST /feedback  ───[Adapts Threshold & Relays to MQTT]──────────┘   │
│   │   • GET  /config                                                        │
│   └─────────────────────────────────────┬───────────────────────────────────┘
└─────────────────────────────────────────┼───────────────────────────────────┘
                                          │
                         REST / WebSocket │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   TIER 3: CLIENT APP (Web / Mobile Frontend)                │
│                                                                             │
│   - Real-time rolling health score charts & FFT spectrum viewer             │
│   - Push alert banners & acoustic warnings                                  │
│   - 👨‍🔧 Human-in-the-Loop Feedback Controls:                                 │
│       [✅ Confirm Fault] ──► POST /api/feedback (Logged for retraining)      │
│       [❌ False Alarm]   ──► POST /api/feedback (Self-tunes threshold * 1.05)│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure & Modular Separation

All server components reside in `server/`, keeping the codebase decoupled from edge HAL and training scripts while sharing root configuration.

```
server/
├── __init__.py
├── server_plan.md           # Master architecture & plan (this document)
├── config.py                # Server settings (loads config.yaml + env overrides)
├── database.py              # SQLite (WAL mode) / PostgreSQL connection & ORM models
├── schemas.py               # Pydantic V2 data contracts (in/out validation)
├── mqtt_bridge.py           # Background MQTT subscriber/publisher worker
├── ws_manager.py            # WebSocket connection manager for live broadcasts
├── routes/                  # Modular REST routers
│   ├── __init__.py
│   ├── telemetry.py         # /api/telemetry (latest, history, metrics)
│   ├── alerts.py            # /api/alerts (list, acknowledge)
│   ├── feedback.py          # /api/feedback (human-in-the-loop adaptation)
│   └── config.py            # /api/config (view & update thresholds)
└── main.py                  # FastAPI app factory, CORS, and Lifespan lifecycle
```

---

## 3. Database Schema & Persistence

The server uses **SQLite with Write-Ahead Logging (WAL)** by default for zero-configuration, robust concurrency, and high throughput ($> 50,000\text{ writes/sec}$), with a clean transition to PostgreSQL/TimescaleDB via SQLAlchemy.

### Table 1: `telemetry`
Stores periodic 10-second health updates from the edge.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PRIMARY KEY | Auto-incrementing record ID |
| `timestamp` | REAL (INDEX) | UNIX epoch timestamp of window |
| `score` | REAL | Anomaly reconstruction error / probability |
| `label` | VARCHAR(16) | `'normal'` or `'anomaly'` |
| `detector` | VARCHAR(32) | Active detector (e.g., `'unsupervised_autoencoder'`) |
| `diagnosis` | VARCHAR(32) | Root cause (e.g., `'inner_race_fault'`, `'healthy'`) |
| `confidence` | REAL | Diagnostic confidence ($0.0 \text{ to } 1.0$) |
| `proc_time_s` | REAL | Edge processing latency in seconds |

### Table 2: `alerts`
Stores threshold breaches requiring technician attention.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PRIMARY KEY | Auto-incrementing alert ID |
| `timestamp` | REAL (INDEX) | Alert trigger timestamp |
| `message` | VARCHAR(256) | Human-readable alert summary |
| `score` | REAL | Anomaly score triggering alert |
| `diagnosis` | VARCHAR(32) | Mechanical diagnosis |
| `acknowledged` | BOOLEAN | `False` until technician reviews |
| `ack_timestamp` | REAL (NULLABLE)| Timestamp of review |

### Table 3: `human_feedback`
Logs all technician interventions and threshold adaptations.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PRIMARY KEY | Auto-incrementing record ID |
| `timestamp` | REAL (INDEX) | Time feedback was submitted |
| `event` | VARCHAR(32) | `'FALSE_ALARM'` or `'CONFIRMED_FAULT'` |
| `score` | REAL | Anomaly score at the time of alert |
| `adapted_threshold` | REAL (NULLABLE)| New self-tuned threshold (if false alarm) |
| `notes` | TEXT | Technician comments / repair details |

### Table 4: `system_config`
Maintains current operational threshold and system parameters.

| Column | Type | Description |
|---|---|---|
| `key` | VARCHAR(64) PRIMARY KEY | Config parameter key (e.g. `'alert_threshold'`) |
| `value` | VARCHAR(256) | Parameter value |
| `updated_at` | REAL | Last modification timestamp |

---

## 4. API Specification

### REST Endpoints (`/api/v1`)

#### 1. System Health & Heartbeat
- **`GET /health`**
  - **Response `200 OK`**:
    ```json
    {
      "status": "healthy",
      "mqtt_connected": true,
      "db_connected": true,
      "edge_online": true,
      "last_edge_heartbeat_s": 4.2,
      "uptime_seconds": 1840.5
    }
    ```

#### 2. Live & Historical Telemetry
- **`GET /api/v1/telemetry/latest`**
  - Returns the single most recent telemetry window received from the Pi.
  - **Response `200 OK`**:
    ```json
    {
      "timestamp": 1731550800.12,
      "score": 0.0028,
      "label": "normal",
      "detector": "hybrid_edge",
      "diagnosis": "healthy",
      "confidence": 0.9972,
      "proc_time_s": 0.002
    }
    ```

- **`GET /api/v1/telemetry/history?limit=60&start=...&end=...`**
  - Returns rolling array of telemetry records for frontend charting.

#### 3. Alerts Management
- **`GET /api/v1/alerts?unacknowledged_only=true&limit=20`**
  - Returns recent alerts sorted in reverse chronological order.
- **`POST /api/v1/alerts/{alert_id}/acknowledge`**
  - Marks an active alert as reviewed by the technician.

#### 4. 👨‍🔧 Human-in-the-Loop Feedback & Adaptation
- **`POST /api/v1/feedback`**
  - Core endpoint implementing human-in-the-loop learning.
  - **Request Body**:
    ```json
    {
      "event": "FALSE_ALARM",
      "score": 0.0482,
      "notes": "Transient line voltage fluctuation caused temporary anomaly."
    }
    ```
  - **Server Action**:
    1. If `event == "FALSE_ALARM"`:
       $$\tau_{\text{new}} = \text{score} \times 1.05$$
       - Updates `system_config` table with `alert_threshold = tau_new`.
       - Logs event in `human_feedback` table.
       - Publishes update payload down to Edge via MQTT:
         ```json
         {
           "threshold": 0.0506,
           "reason": "human_false_alarm",
           "ts": 1731550850.12
         }
         ```
    2. If `event == "CONFIRMED_FAULT"`:
       - Logs confirmed failure in DB for offline supervised fine-tuning.
  - **Response `200 OK`**:
    ```json
    {
      "status": "success",
      "event": "FALSE_ALARM",
      "adapted_threshold": 0.0506,
      "mqtt_broadcast": true
    }
    ```

#### 5. Configuration & Threshold Management
- **`GET /api/v1/config`**: Returns active threshold and motor parameters.
- **`POST /api/v1/config/threshold`**: Manually calibrate threshold and broadcast to Pi.

---

### WebSocket Real-Time Stream

- **`WS /ws/telemetry`**
  - Real-time full-duplex stream for frontend dashboards.
  - Broadcasts whenever new telemetry or an alert is ingested via MQTT.
  - Eliminates frontend HTTP polling, reducing latency to $< 10\text{ ms}$.
  - **Outgoing Message**:
    ```json
    {
      "type": "TELEMETRY_UPDATE",
      "data": {
        "timestamp": 1731550800.12,
        "score": 0.0028,
        "label": "normal",
        "diagnosis": "healthy",
        "confidence": 0.9972
      }
    }
    ```

---

## 5. Hybrid Pub/Sub & Web Push Architecture (Disconnected Client Resilience)

### The Problem
In real-world predictive maintenance, technicians do **not** keep a dashboard open on a screen 24/7. Mobile phones sleep in pockets, browser tabs get closed, and Wi-Fi drops.
- **WebSocket fragility**: WebSockets are ephemeral. When the browser tab or mobile screen is closed, the TCP socket is terminated by the OS to save battery. Any alert sent via WebSocket to a closed app evaporates.

### The Solution: 3-Pillar Hybrid Pub/Sub
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HYBRID PUB/SUB ENGINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Background Web Push (W3C Push API / RFC 8030 VAPID)                      │
│    - When app is CLOSED/SLEEPING: Server dispatches cryptographically signed│
│      push message through browser push service (Chrome FCM / Mozilla / etc).│
│    - Triggers system lock screen banner & acoustic notification.             │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. Catch-Up Sync on App Resume                                              │
│    - When technician taps notification and app opens, it fetches:           │
│      • GET /api/v1/alerts?unacknowledged_only=true (pending fault alerts)   │
│      • GET /api/v1/telemetry/latest (current motor state)                   │
│    - Immediately populates dashboard state without waiting for next window. │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. Foreground Streaming (WebSocket & MQTT over WebSockets)                  │
│    - While app is OPEN: WebSocket /ws/telemetry streams 10-second updates   │
│      at <10ms latency.                                                      │
│    - MQTT retained topic (`motor/status`) ensures zero connection lag.      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### New Database Table: `push_subscriptions`
Stores device push subscription tokens:
| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PRIMARY KEY | Unique subscription ID |
| `endpoint` | TEXT UNIQUE NOT NULL | Browser push gateway endpoint URL |
| `p256dh` | TEXT NOT NULL | Client P-256 public encryption key |
| `auth` | TEXT NOT NULL | Authentication secret key |
| `client_id` | TEXT | Optional client / device identifier |
| `created_at` | REAL NOT NULL | Subscription timestamp |

### New Push Notification Service (`server/push_notifier.py`)
- Manages server VAPID key pairs (`private_key.pem`, `public_key`).
- Exposes `broadcast_alert(alert_payload)`: Iterates over registered subscriptions in `push_subscriptions`, encrypts payload with AES-128-GCM, and dispatches via HTTP/2 Web Push.
- Cleans up dead/expired subscriptions (HTTP 404/410 Gone).

### New API Endpoints (`server/routes/push.py`)
- `GET /api/v1/push/key`: Returns VAPID public key in base64 URL-safe format.
- `POST /api/v1/push/subscribe`: Registers client subscription payload (`{endpoint, keys: {p256dh, auth}}`).
- `POST /api/v1/push/unsubscribe`: Deregisters an endpoint.
- `POST /api/v1/push/test`: Dispatches a simulated test alert push notification.

### MQTT Bridge Integration Hook
In `server/mqtt_bridge.py`:
```python
def _handle_alert_message(self, payload: Dict[str, Any]) -> None:
    # 1. Persist to DB
    insert_alert(payload)
    # 2. Broadcast to open WebSockets
    ws_manager.broadcast_sync({"type": "ALERT", "data": payload})
    # 3. Dispatch Web Push to closed/sleeping apps
    push_notifier.broadcast_alert(payload)
```

---

## 6. Code Reuse & Integration Strategy

1. **Configuration Reuse (`config.yaml`)**:
   - `server/config.py` directly parses the root `config.yaml` to read MQTT broker host, port, and topic names (`topic_health`, `topic_alert`, `topic_config`).
   - Supports environment variable overrides (`MQTT_BROKER`, `SERVER_PORT`, `DATABASE_URL`) for cloud container deployments (Docker / Railway / Render).

2. **Schema Uniformity with Edge**:
   - Pydantic models in `server/schemas.py` match the dictionary structure published by [`service/monitor.py`](file:///c:/Users/aarav/OneDrive/Desktop/ESI/service/monitor.py) and consumed by [`dashboard/app.py`](file:///c:/Users/aarav/OneDrive/Desktop/ESI/dashboard/app.py).

3. **MQTT Bridge Worker**:
   - Uses `paho-mqtt` in a dedicated background thread or async loop inside the FastAPI Lifespan context.
   - Automatically reconnects if broker drops, guaranteeing zero dropped messages.

---

## 6. Implementation Milestones

| Step | Milestone | Deliverables |
|:---:|---|---|
| **1** | **Configuration & Database** | `server/config.py`, `server/database.py` with SQLAlchemy ORM models and SQLite WAL initialization. |
| **2** | **Pydantic Schemas** | `server/schemas.py` for Telemetry, Alert, Feedback, and Config. |
| **3** | **MQTT Ingestion & Bridge** | `server/mqtt_bridge.py` subscribing to `motor/health` and publishing to `motor/config/threshold`. |
| **4** | **WebSocket Broadcaster** | `server/ws_manager.py` managing real-time broadcast to connected web/mobile clients. |
| **5** | **REST Routers** | `server/routes/` implementing all telemetry, alert, feedback, and config endpoints. |
| **6** | **FastAPI Application Entry** | `server/main.py` with lifespan manager and CORS middleware. |
| **7** | **Verification Suite** | `tests/test_server_api.py` and `tests/test_server_e2e.py` verifying full edge-to-server loop. |
