# 🛠️ Vibration-Based Motor Fault Detection — Full Build Plan & Architecture

> **Goal**: Build an edge-to-cloud predictive maintenance system for small electric motors (fans, pumps) using a low-cost MEMS accelerometer on a Raspberry Pi 4, a 1D-CNN Autoencoder trained on healthy baseline vibration, and a decoupled 3-tier IoT architecture with Human-in-the-Loop active feedback.

---

## 1. Decoupled 3-Tier IoT Architecture

The system is structured as three strictly decoupled tiers to guarantee high reliability, edge autonomy, and separation of concerns. The Raspberry Pi operates entirely headless without local GUI or browser overhead.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         TIER 1: EDGE LAYER (Raspberry Pi 4)                      │
│                                                                                  │
│   [Motor Rig] ──vibration──> [ADXL345 / MPU6050] (1000 Hz, ±16g)                 │
│                                      │ I2C (smbus2)                              │
│                               ┌──────▼──────┐                                    │
│                               │  HAL Sensor │ (PhysicalSensor / MockSensor)      │
│                               └──────┬──────┘                                    │
│                                      │ 10-second raw windows (10,000 samples)    │
│                         ┌────────────▼────────────┐                              │
│                         │ Signal Conditioning     │ (DC removal, bandpass, RMS)  │
│                         │ Feature & FFT Spectrum  │ (Order-tracked f/f_run)      │
│                         └────────────┬────────────┘                              │
│                                      │                                           │
│                         ┌────────────▼────────────┐                              │
│                         │ INT8 PyTorch Autoencoder│ (~3 ms latency on Pi CPU)    │
│                         │ Dynamic Anomaly Scoring │ (MSE > tau -> Alert)         │
│                         └───────┬──────────▲──────┘                              │
│         telemetry (health/alert)│          │ remote threshold update             │
│         JSON over MQTT          │          │ (motor/config/threshold)            │
└─────────────────────────────────┼──────────┼─────────────────────────────────────┘
                                  │          │
                     MQTT (QoS 1) │          │ MQTT (QoS 1)
                                  ▼          │
┌────────────────────────────────────────────┴─────────────────────────────────────┐
│                    TIER 2: SERVER / BACKEND LAYER (Cloud / Gateway)              │
│                                                                                  │
│   ┌───────────────────────────┐           ┌──────────────────────────────────┐   │
│   │    Central MQTT Broker    │           │    Backend Application Server    │   │
│   │   (Mosquitto / EMQX)      │◄─────────►│        (FastAPI / Node.js)       │   │
│   └─────────────┬─────────────┘           └─────────┬──────────────▲─────────┘   │
│                 │                                   │              │             │
│                 ▼                                   ▼              │             │
│   ┌───────────────────────────┐           ┌──────────────────┐     │             │
│   │ Time-Series Database      │           │ WebSocket / REST │     │             │
│   │ (TimescaleDB / InfluxDB)  │           │ Telemetry Feeds  │     │             │
│   └───────────────────────────┘           └─────────┬────────┘     │             │
└─────────────────────────────────────────────────────┼──────────────┼─────────────┘
                                                      │              │
                                   WebSocket / REST   │              │ User Feedback
                                   Telemetry Stream   │              │ [Confirm / False]
                                                      ▼              │
┌────────────────────────────────────────────────────────────────────┴─────────────┐
│             TIER 3: CLIENT APP LAYER (Web / Mobile Frontend — DEFERRED)          │
│                                                                                  │
│   - Remote Web / Mobile UI (React / Next.js / Streamlit Client / Mobile App)     │
│   - Runs on user's laptop, tablet, or smartphone (Completely detached from Pi)   │
│   - Live health score charts, spectrum inspector, and alert notifications        │
│   - 👨‍🔧 Human-in-the-Loop Feedback:                                              │
│       • [✅ Confirm Fault] -> Flags window for offline retraining                 │
│       • [❌ False Alarm]   -> Computes tau_new = score * 1.05 -> Pushes to Server │
│                                                                                  │
│   *Current Status*: Deferred to App Stage after edge inference is validated.     │
│   *Interim Dev Harness*: `dashboard/app.py` acts as a local developer mock.      │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 📌 Core Architectural Tenets ("Things to Remember")

1. **Strict Decoupled Tier Boundary**:
   - The Raspberry Pi (Tier 1) **never** communicates directly with the Client App (Tier 3).
   - All telemetry flows upward: $\text{Edge} \xrightarrow{\text{MQTT}} \text{Server Broker} \xrightarrow{\text{API}} \text{App}$.
   - All configurations flow downward: $\text{App} \xrightarrow{\text{REST}} \text{Server API} \xrightarrow{\text{MQTT}} \text{Edge}$.

2. **Zero GUI on the Edge Device**:
   - The Raspberry Pi operates 100% headless. It runs `service/monitor.py` as a systemd background service.
   - Do not install browsers, desktop environments, or heavy UI frameworks on the Pi.

3. **Human-in-the-Loop Remote Adaptation**:
   - When a false alarm occurs, the technician clicks `[False Alarm]` in the Client App.
   - The App instructs Tier 2, which publishes an adapted threshold ($\tau_{\text{new}} = \text{score} \times 1.05$) to MQTT topic `motor/config/threshold`.
   - The Edge Daemon dynamically updates its in-memory threshold without restarting the service or losing sensor sync.

4. **Colab-First Model Pipeline**:
   - Heavy training, hyperparameter exploration, and model calibration take place in Google Colab (`colab/motor_anomaly_autoencoder.ipynb`) on GPU.
   - Genuine CWRU `.mat` files are downloaded directly from `engineering.case.edu`.
   - Trained models are dynamically quantized to INT8 and exported as `autoencoder_best.pt` for Pi deployment.

5. **Speed-Invariance (Order Tracking)**:
   - Variable motor speeds (Low: 25 Hz, Med: 40 Hz, High: 55 Hz) shift harmonic peaks and cause false anomalies if unaddressed.
   - FFT spectra are normalized relative to running speed ($f / f_{\text{run}}$) or conditioned by operating frequency to guarantee speed-invariance.

6. **Edge Resource Guardrails**:
   - CPU usage on Raspberry Pi 4 must remain $< 5\%$.
   - Inference latency on Pi CPU must be $< 10\text{ ms}$ per 10-second window (target: ~3 ms with INT8).
   - RAM footprint must remain $< 50\text{ MB}$.

7. **Staged Implementation Strategy**:
   - **Stage 1 (Current Focus)**: HAL, data pipelines, signal conditioning, Colab model training, INT8 quantization, and live edge daemon.
   - **Stage 2 (App Stage)**: Tier 2 Cloud/Server Backend API and standalone Tier 3 Web/Mobile Client App. In the interim, `dashboard/app.py` serves strictly as a local test harness.

---

## 3. Folder Structure

```
ESI/
├── models/                          # Pre-trained TorchScript Edge Models
│   ├── cwru_unsupervised_xiong_edge.pt # 1D-CNN Autoencoder (healthy data baseline)
│   └── cwru_advanced_edge.pt        # 4-class Fault Classifier (healthy + faults)
│
├── colab/
│   ├── model.md                     # Model architecture, equations, INT8 quantization plan
│   └── motor_anomaly_autoencoder.ipynb # One-click Colab training notebook with real CWRU data
│
├── hal/                             # Tier 1: Hardware Abstraction Layer
│   ├── __init__.py
│   ├── base.py                      # VibrationSensor ABC (get_10s_window)
│   ├── physical_sensor.py           # ADXL345 / MPU6050 over I2C (smbus2)
│   └── mock_sensor.py               # Dataset replay sensor (CWRU / rig .npy)
│
├── data/                            # Signal processing & datasets
│   ├── __init__.py
│   ├── loader.py                    # CWRU .mat parser (numeric & descriptive stems)
│   ├── cleaner.py                   # SignalCleaner (DC removal, bandpass, RMS norm)
│   ├── splitter.py                  # Splitter (train_healthy, mock, eval)
│   ├── pipeline.py                  # DataPipeline (DATA_MODE = MOCK | REAL)
│   ├── cwru/                        # Real CWRU .mat files (97, 105, 118, 130)
│   ├── raw/                         # Raw .npy windows recorded from rig
│   └── processed/                   # Preprocessed feature tables
│
├── features/                        # Feature Extraction
│   ├── __init__.py
│   └── extractor.py                 # RMS, crest factor, kurtosis, FFT harmonics
│
├── detectors/                       # Anomaly Detectors & Deep Learning Inference
│   ├── __init__.py
│   ├── base.py                      # BaseDetector ABC (fit, predict, save, load)
│   ├── threshold.py                 # Baseline RMS threshold (mu + k*sigma)
│   ├── isolation_forest.py          # Scikit-learn Isolation Forest
│   ├── autoencoder.py               # PyTorch 1D-CNN Autoencoder
│   ├── edge_models.py               # TorchScript inference (Unsupervised, Classifier, Hybrid)
│   └── saved/                       # Exported model weights (.pkl, .pt)
│
├── service/                         # Tier 1: Live Edge Service
│   ├── __init__.py
│   ├── monitor.py                   # Headless daemon (sensor -> features -> detector -> MQTT)
│   └── mqtt_publisher.py            # MQTT publisher (health/alerts) & config listener
│
├── client/                          # Tier 3: Web / Mobile Client Application
│   ├── client_plan.md               # Tier 3 architecture and data flow master plan
│   ├── ui_plan.md                   # UI/UX design specification & wireframes
│   ├── index.html                   # Industrial dark-mode Single-Page Dashboard
│   ├── manifest.json                # PWA metadata (standalone installable)
│   ├── sw.js                        # Service Worker (Web Push & notification routing)
│   ├── css/
│   │   └── styles.css               # Glassmorphism & cyber-physical animations
│   └── js/
│       ├── config.js                # Dynamic host/port resolution
│       ├── api.js                   # REST API client wrapper
│       ├── ws.js                    # Auto-reconnecting WebSocket client
│       ├── audio.js                 # Web Audio API synthesized harmonic chime
│       ├── charts.js                # Chart.js real-time streaming timeline
│       ├── push.js                  # Web Push VAPID subscription manager
│       └── app.js                   # Application orchestrator & HITL feedback
│
├── dashboard/                       # Tier 3 (Interim Local Prototyping Mock)
│   └── app.py                       # Streamlit dev mock (superseded by client/)
│
├── training/                        # Local training & calibration scripts
│   ├── train_rms_threshold.py
│   ├── train_isolation_forest.py
│   └── train_autoencoder.py
│
├── evaluation/                      # Offline benchmarks & metrics
│   ├── metrics.py                   # Precision, Recall, F1, ROC-AUC
│   ├── evaluate_rig.py              # Benchmark on local rig data
│   └── evaluate_cwru.py             # Cross-benchmark on CWRU bearing dataset
│
├── scripts/                         # Operational CLI scripts
│   ├── download_real_cwru.py        # Fetches real .mat from engineering.case.edu
│   ├── validate_data.py             # Signal health check CLI
│   ├── generate_sample_cwru.py      # Creates synthetic CWRU test files
│   ├── run_pipeline_demo.py         # End-to-end pipeline verification
│   └── record.py                    # Records live rig data to .npy
│
├── tests/                           # Comprehensive test suite (166 tests passing across 16 suites)
│   ├── test_cwru_loader.py
│   ├── test_data_validation.py
│   ├── test_data_pipeline.py
│   ├── test_pipeline_e2e.py
│   ├── test_features.py
│   ├── test_detectors.py
│   ├── test_edge_models.py
│   ├── test_hal.py
│   ├── test_app_service.py
│   ├── test_training_eval.py
│   ├── test_server_database.py
│   ├── test_server_api.py
│   ├── test_server_mqtt_bridge.py
│   ├── test_server_push.py
│   ├── test_client_serving.py
│   └── test_e2e_edge_server.py
│
├── config.yaml                      # Hardware, motor, detector, and MQTT configuration
├── requirements.txt                 # Full dev dependencies
├── requirements-pi.txt              # Ultra-light edge dependencies for Raspberry Pi
└── plan.md                          # ← Master architecture & plan (this file)
```

---

## 4. Tech Stack by Tier

### Tier 1: Edge Layer (Raspberry Pi 4)
- **OS**: Raspberry Pi OS Lite (64-bit Debian bookworm, headless, no desktop environment).
- **Runtime**: Python 3.11+.
- **I2C Driver**: `smbus2` interacting directly with sensor registers (`0x53` for ADXL345, `0x68` for MPU6050).
- **Numerics & Signal**: `numpy` (array ops), `scipy` (Butterworth bandpass, Welch PSD).
- **Edge ML**: `torch` CPU-only wheel with dynamic INT8 quantization (`torch.quantization.quantize_dynamic`).
- **Telemetry Client**: `paho-mqtt` communicating with Tier 2 broker.
- **Process Supervision**: `systemd` daemon (`motor-monitor.service`) with auto-restart.

### Tier 2: Server / Backend Layer (Cloud / Central Gateway)
- **Broker**: Eclipse Mosquitto or EMQX (MQTT v3.1.1/v5.0 with TLS).
- **Backend API**: Python FastAPI or Node.js / Express.
- **Database**: TimescaleDB / PostgreSQL or InfluxDB for time-series vibration history.
- **Communication**: WebSockets for live telemetry streaming to clients; REST for feedback and configuration APIs.

### Tier 3: Client App Layer (Deferred to App Stage)
- **Frontend Framework**: React / Next.js (Web) or Flutter / React Native (Mobile).
- **Visualization**: Plotly.js / Chart.js for real-time anomaly scores and spectral cascades.
- **Human-in-the-Loop Controls**: Real-time push alert handling, fault confirmation, and threshold adaptation triggers.

### Model Training Environment (Google Colab)
- **Hardware**: Colab NVIDIA T4 / V100 GPU.
- **Frameworks**: PyTorch, Scipy, Scikit-learn, Pandas, Matplotlib.
- **Dataset**: Direct HTTPS download from Case Western Reserve University (CWRU) Bearing Data Center.
- **Export Format**: Serialized state dict and TorchScript INT8 `.pt`.

---

## 5. Build Phases & Roadmap

### Phase 1: Data Pipeline & Signal Cleaning (Completed ✅)
- [x] Implement `CWRULoader` supporting both descriptive filenames and official numeric IDs (`97`, `105`, `118`, `130`).
- [x] Build `SignalCleaner` with DC bias removal, 4th-order Butterworth bandpass filter, $3\sigma$ outlier clipping, and unit-RMS normalization.
- [x] Create `DataPipeline` with clean `MOCK` $\leftrightarrow$ `REAL` sensor mode toggle.
- [x] Implement CLI data validation script `scripts/validate_data.py`.
- [x] Download real CWRU benchmark files from `engineering.case.edu`.

### Phase 2: HAL & Feature Extraction (Completed ✅)
- [x] Build `VibrationSensor` ABC with `get_10s_window()`.
- [x] Implement `PhysicalSensor` for ADXL345 / MPU6050 over I2C with register-level configuration.
- [x] Implement `MockSensor` with looping dataset replay and simulated time delay.
- [x] Build `FeatureExtractor` extracting RMS, peak, crest factor, kurtosis, spectral entropy, and harmonic band energies.

### Phase 3: Model Architecture & Colab Pipeline (Completed ✅)
- [x] Document mathematical formulation, 1D-CNN architecture, and INT8 quantization in `colab/model.md`.
- [x] Create self-contained `colab/motor_anomaly_autoencoder.ipynb` with automated dataset download, training, threshold tuning, and INT8 export.
- [x] Implement local detector baseline: `RMSThresholdDetector`, `IsolationForestDetector`, `AutoencoderDetector`.
- [x] Establish speed-invariance via order-tracked spectral bins ($f / f_{\text{run}}$).

### Phase 4: Headless Edge Service & Remote Config (Completed ✅)
- [x] Build `service/monitor.py` headless daemon for Raspberry Pi.
- [x] Build `service/mqtt_publisher.py` with QoS 1 telemetry and alert dispatch.
- [x] Implement remote configuration subscriber on `motor/config/threshold`.
- [x] Enable dynamic threshold adaptation on the edge without restarting the service.
- [x] Verify full test suite (117+ unit and integration tests passing).

### Phase 5: Hardware Verification on Physical Motor Rig (Current Stage 🚀)
- [ ] Wire ADXL345 / MPU6050 to Raspberry Pi 4 GPIO (SDA: GPIO 2, SCL: GPIO 3, 3.3V, GND).
- [ ] Verify I2C connectivity using `i2cdetect -y 1`.
- [ ] Measure and log the actual achieved I2C sample rate over 10-second continuous windows.
- [ ] Collect healthy baseline vibration windows across all three speed settings (Low, Medium, High).
- [ ] Introduce controlled mechanical faults (calibrated adhesive mass on blades, loosened mounting screws) and record fault signatures.
- [ ] Deploy quantized `autoencoder_best.pt` on the Pi and verify CPU latency $< 5\text{ ms}$.

### Phase 6: Cloud Server API & Human-in-the-Loop Backend (Tier 2 Completed ✅)
- [x] Create Tier 2 Server architecture plan (`server/server_plan.md`).
- [x] Implement FastAPI backend application (`server/main.py`) with CORS and lifespan.
- [x] Implement SQLite WAL time-series database (`server/database.py`) with telemetry, alerts, human_feedback, and system_config tables.
- [x] Implement Pydantic V2 data contracts (`server/schemas.py`).
- [x] Implement real-time WebSocket connection manager (`server/ws_manager.py`).
- [x] Implement MQTT Bridge background worker (`server/mqtt_bridge.py`) for two-way communication:
  - Ingests `motor/health` and `motor/alert` into DB and broadcasts to WebSocket.
  - Subscribes to dynamic threshold feedback and publishes updates to `motor/config/threshold`.
- [x] Implement REST API routes (`server/routes/`):
  - `GET /health` (server, MQTT status, edge heartbeat)
  - `GET /api/v1/telemetry/latest`, `GET /api/v1/telemetry/history`
  - `GET /api/v1/alerts`, `POST /api/v1/alerts/{alert_id}/acknowledge`
  - `POST /api/v1/feedback` (Human-in-the-Loop threshold self-tuning & MQTT push), `GET /api/v1/feedback`
  - `GET /api/v1/config`, `POST /api/v1/config/threshold`
- [x] Verify complete test suite (146 tests passing across 14 test files, including full end-to-end edge-to-server cycle).

### Phase 7: Tier 3 Client App & Production Deployment (Current Stage 🚀)
- [x] Create Tier 3 architecture and data flow specification (`client/client_plan.md`).
- [x] Create industrial UI/UX design tokens and wireframe specification (`client/ui_plan.md`).
- [x] Implement modern zero-build Single Page Application (`client/index.html`, `client/css/styles.css`).
- [x] Implement modular client architecture:
  - `client/js/config.js`: Dynamic endpoint & origin resolution.
  - `client/js/api.js`: REST API client for telemetry, alerts, HITL feedback, and config.
  - `client/js/ws.js`: WebSocket client with automatic exponential backoff reconnection.
  - `client/js/audio.js`: Web Audio API synthesized harmonic alarm chime (zero external assets).
  - `client/js/charts.js`: Real-time streaming Chart.js timeline with dynamic threshold animation.
  - `client/js/push.js`: Web Push VAPID subscription manager.
  - `client/js/app.js`: Main state orchestrator with Catch-Up Sync on resume.
- [x] Implement Service Worker (`client/sw.js`) and PWA manifest (`client/manifest.json`).
- [x] Mount client static files directly in FastAPI (`server/main.py`) for same-origin execution at `http://localhost:8000/`.
- [x] Verify complete test suite (163 tests passing across 16 test files).

### Phase 8: 100% Free Production Deployment & Edge Kit (Completed ✅)
- [x] Configure Render.com 1-click free cloud blueprint (`render.yaml`).
- [x] Create production containerfile (`server/Dockerfile`) and all-in-one local stack (`docker-compose.yml`).
- [x] Support TLS (port 8883) and user/pass authentication for free managed brokers (HiveMQ Cloud).
- [x] Create Raspberry Pi 4 automated setup script (`scripts/pi_setup.sh`) and systemd unit (`scripts/motor-monitor.service`).
- [x] Set up free GitHub Actions CI workflow (`.github/workflows/ci.yml`).
- [x] Document complete step-by-step 100% free deployment guide (`deployment_guide.md`).
- [x] Verify complete test suite (166 tests passing across 16 test files).

### Phase 9: Hardware Verification & Live Field Run (When Hardware Arrives)
- [ ] Connect ADXL345 / MPU6050 to Pi 4 GPIO (Pins: SDA GPIO2, SCL GPIO3, 3.3V, GND).
- [ ] Run `i2cdetect -y 1` and verify detection at address 0x53 or 0x68.
- [ ] Start background service `sudo systemctl start motor-monitor.service`.
- [ ] Verify live telemetry streaming to Render dashboard and test lockscreen Web Push alerts.

---

## 6. Mathematical Formulation

### 1. Signal Normalization
Raw vibration signals $x[n]$ are conditioned to remove DC bias and scaled to unit-RMS:
$$\tilde{x}[n] = \frac{x[n] - \mu_x}{\text{RMS}(x - \mu_x)}$$

### 2. Spectral Reconstruction Error
The autoencoder maps the 512-point order-tracked FFT magnitude spectrum $S$ to a latent representation $z$ and reconstructs $\hat{S}$:
$$z = \text{Encoder}(S), \quad \hat{S} = \text{Decoder}(z)$$
$$\mathcal{L}_{\text{MSE}}(S, \hat{S}) = \frac{1}{K} \sum_{k=1}^K \left( S_k - \hat{S}_k \right)^2$$

### 3. Statistical Anomaly Threshold
Trained strictly on healthy baseline data, the decision threshold $\tau$ is calibrated using extreme value statistics:
$$\tau = \mu_{\text{train}} + 3.5 \cdot \sigma_{\text{train}}$$
$$\text{State}(S) = \begin{cases} \text{Normal}, & \text{if } \mathcal{L}_{\text{MSE}}(S, \hat{S}) \le \tau \\ \text{Anomaly}, & \text{if } \mathcal{L}_{\text{MSE}}(S, \hat{S}) > \tau \end{cases}$$

### 4. Human-in-the-Loop Dynamic Calibration
When a technician identifies a false alarm on sample score $s_{\text{false}} > \tau$, the threshold adapts immediately with a $5\%$ engineering margin:
$$\tau_{\text{new}} = \max\left(\tau, \; 1.05 \cdot s_{\text{false}}\right)$$

---

## 7. MQTT Message Specifications

### 1. Telemetry (`motor/health`) — Published by Edge Daemon every 10s
```json
{
  "score": 0.0412,
  "label": "normal",
  "ts": 1731518400.12,
  "detector": "autoencoder",
  "proc_time_s": 0.003
}
```

### 2. Fault Alert (`motor/alert`) — Published on Threshold Breach
```json
{
  "message": "FAULT DETECTED — score=0.8924",
  "score": 0.8924,
  "ts": 1731518410.15
}
```

### 3. Remote Configuration (`motor/config/threshold`) — Received by Edge Daemon
```json
{
  "threshold": 0.4250,
  "reason": "human_false_alarm",
  "ts": 1731518450.00
}
```

---

## 8. Safety & Physical Rig Guardrails

> [!CAUTION]
> - **Always keep the protective fan guard firmly installed** during all powered motor tests.
> - **Unplug the motor from mains power** before touching blades, adding weights, or loosening bolts.
> - **Never touch spinning parts** — wait for complete rotation stop.
> - Clamp or bolt the motor test bench securely to a heavy surface to prevent bench rocking.
> - Begin imbalance testing with minute adhesive masses ($< 0.5\text{ g}$), increasing incrementally.
