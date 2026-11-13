# ⚡ VibraGuard // Edge AI Motor Sentinel

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-TorchScript_1D--CNN-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![MQTT](https://img.shields.io/badge/MQTT-Paho_v3.1.1-660066?style=for-the-badge&logo=eclipsemosquitto&logoColor=white)
![PWA](https://img.shields.io/badge/Frontend-PWA_Web_Push-5A0FC8?style=for-the-badge&logo=pwa&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-166_Passing-10B981?style=for-the-badge&logo=pytest&logoColor=white)

**Vibration-based predictive maintenance and real-time anomaly detection for rotating electric machinery (fans, pumps, blowers) using low-cost MEMS accelerometers, dynamic INT8 deep learning on edge CPU, and a decoupled 3-tier IoT architecture with Human-in-the-Loop active feedback.**

[Architecture](#-decoupled-3-tier-architecture) • [Technologies](#-technologies-used) • [Quick Start](#-quick-start) • [100% Free Deployment](#-100-free-cloud-deployment) • [Hardware Pinout](#-hardware-pinout--wiring) • [Tests](#-automated-tests)

</div>

---

## 📌 Problem & Motivation

Unplanned downtime in small-to-medium industrial motors costs thousands of dollars per incident. High-end enterprise condition monitoring systems (vibration analyzers, eddy-current probes) cost \$5,000–\$20,000+ per asset, pricing out small factories, workshops, and facilities.

**VibraGuard** solves this by combining:
1. **Low-Cost MEMS Sensors** (\$3–\$5 ADXL345 / MPU6050) sampled via I2C at 1000 Hz.
2. **Edge Deep Learning**: Pre-trained TorchScript 1D-CNN Autoencoders running on a headless Raspberry Pi 4 CPU ($<0.1\%$ CPU, $<35\text{ MB}$ RAM, $0.5\text{ ms}$ latency).
3. **Decoupled 3-Tier IoT Architecture**: Edge never touches the UI directly; all communication flows via secure MQTT and WebSockets through an independent cloud server.
4. **Hybrid Pub/Sub & Disconnected Resilience**: When the technician's phone is asleep or the web app is closed, W3C Web Push (VAPID) delivers lockscreen alert banners. When opened, it runs an instant "Catch-Up Sync".
5. **Human-in-the-Loop Active Adaptation**: When an operator marks an alert as **[False Alarm]**, the edge threshold adapts dynamically ($\tau_{\text{new}} = \text{score} \times 1.05$) via downward MQTT push without restarting the daemon.

---

## 🏛️ Decoupled 3-Tier Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                TIER 1: EDGE LAYER (Raspberry Pi 4)                               │
│                                                                                                  │
│   [Motor Rig] ──vibration──► [ADXL345 / MPU6050] (1000 Hz, ±16g, I2C /dev/i2c-1)                 │
│                                       │                                                          │
│                               ┌───────▼───────┐                                                  │
│                               │  HAL Sensor   │ (PhysicalSensor / MockSensor with Web Hot-Swap)  │
│                               └───────┬───────┘                                                  │
│                                       │ 10-second continuous windows (10,000 samples)            │
│                         ┌─────────────▼─────────────┐                                            │
│                         │   Signal Conditioning     │ (DC removal, Butterworth bandpass, RMS)    │
│                         │   Order-Tracked Spectrum  │ (f / f_run speed-invariant FFT bins)       │
│                         └─────────────┬─────────────┘                                            │
│                                       │                                                          │
│                         ┌─────────────▼─────────────┐                                            │
│                         │  Two-Stage Deep Learning  │ Stage 1: 1D-CNN Autoencoder (MSE > τ)      │
│                         │  Edge Inference (Torch)   │ Stage 2: 4-Class Fault Diagnosis Classifier│
│                         └─────────────┬─────────────┘ (< 0.5 ms inference latency on Pi CPU)     │
│                                       │                                                          │
│       telemetry: `motor/health` (10s) │   ▲ remote threshold: `motor/config/threshold`           │
│       alerts:    `motor/alert` (fault)│   │ dynamic sensor mode: `motor/config/sensor`           │
└───────────────────────────────────────┼───┼──────────────────────────────────────────────────────┘
                                        │   │
                     MQTT (QoS 1, TLS)  │   │ MQTT (QoS 1, TLS)
                                        ▼   │
┌───────────────────────────────────────────┴──────────────────────────────────────────────────────┐
│                       TIER 2: SERVER BACKEND LAYER (FastAPI + SQLite WAL)                        │
│                                                                                                  │
│   ┌───────────────────────────┐           ┌──────────────────────────────────────────────────┐   │
│   │    Central MQTT Broker    │           │            Bi-Directional MQTT Bridge            │   │
│   │  (HiveMQ Cloud / Mosquitto│◄─────────►│  • Ingests telemetry & alerts into SQLite WAL    │   │
│   │   TLS Port 8883)          │           │  • Dispatches threshold updates to Edge via MQTT │   │
│   └───────────────────────────┘           └─────────┬────────────────────────────┬───────────┘   │
│                                                     │                            │               │
│                                                     ▼                            ▼               │
│                        ┌────────────────────────────────────────┐   ┌────────────────────────┐   │
│                        │       SQLite (WAL Mode Engine)         │   │  Web Push Dispatcher   │   │
│                        │  • telemetry  • alerts                 │   │  (VAPID / RFC 8030)    │   │
│                        │  • human_feedback  • system_config     │   │  Alerts sleeping apps  │   │
│                        └────────────────────────────────────────┘   └────────────┬───────────┘   │
│                                                                                  │               │
│                                        ┌─────────────────────────────────────────┘               │
│                                        │ Web Push Notification                                   │
│                                        │ (When device is asleep/closed)                          │
│                                        ▼                                                         │
│   ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                        FastAPI Application Engine (Single Port 8000)                     │   │
│   │                                                                                          │   │
│   │   REST API (/api/v1)                                  WebSocket Stream (/ws/telemetry)   │   │
│   │   • GET  /telemetry/latest & /history                 • Pushes live snapshots <10ms      │   │
│   │   • GET  /alerts & POST /alerts/{id}/ack              • Dispatches live config updates   │   │
│   │   • POST /feedback (Human-in-the-Loop Adaptation)     • Static asset delivery (Client)   │   │
│   │   • POST /config/sensor-mode (Mock <-> Physical swap)                                    │   │
│   └─────────────────────────────────────────┬────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────┼────────────────────────────────────────────────────┘
                                              │
                             WebSocket / REST │
                                              ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                            TIER 3: CLIENT APP LAYER (PWA Dashboard)                              │
│                                                                                                  │
│   • Runs in any desktop or mobile browser (Chrome, Edge, Firefox, Safari iOS)                    │
│   • Cyber-Physical Dark Mode UI with tabular jitter-free telemetry numbers                       │
│   • Rolling 10-Minute Real-Time Anomaly Score Timeline (Chart.js) with dynamic threshold curve   │
│   • Service Worker (`sw.js`): W3C Web Push event listener + background lockscreen alert banner   │
│   • Native Web Audio API synthesized harmonic alert chimes (880 Hz -> 1174 Hz)                   │
│   • 👨‍🔧 Human-in-the-Loop Decision Center:                                                        │
│       [❌ False Alarm] ──► Calculates τ_new = score * 1.05 ──► Updates Edge live over MQTT        │
│       [✅ Confirm Fault] ──► Logs verified defect to DB for offline supervised retraining          │
│   • Sensor Mode Switch: 1-click toggle between [MOCK (CWRU Replay)] and [PHYSICAL (I2C)]         │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technologies Used

| Tier / Subsystem | Technology | Purpose |
|---|---|---|
| **Tier 1 (Edge Device)** | **Raspberry Pi 4B (64-bit Lite)** | Low-power headless field processor |
| | **ADXL345 / MPU6050** | 3-axis MEMS accelerometer ($\pm 16g$, $1000\text{ Hz}$) |
| | **Linux `smbus2`** | Kernel-level I2C register configuration & burst read |
| | **PyTorch (TorchScript CPU)** | Dynamic INT8 quantized 1D-CNN Autoencoder & 4-class classifier |
| | **SciPy / NumPy** | 4th-order Butterworth bandpass, FFT spectra, and order tracking |
| | **Paho MQTT (Python)** | QoS 1 telemetry publisher and remote config subscriber |
| | **Linux `systemd`** | Background service management, auto-restart, and boot supervision |
| **Tier 2 (Server Backend)**| **FastAPI + Uvicorn** | High-performance asynchronous REST & WebSocket backend |
| | **SQLite (WAL Mode)** | Concurrency-safe time-series storage ($>50,000\text{ writes/sec}$) |
| | **Pydantic V2** | Strict bidirectional data contract validation |
| | **PyWebPush & PyVAPID** | W3C Push API / RFC 8030 encrypted lockscreen alert dispatch |
| | **HiveMQ Cloud (Free)** | Fully managed, TLS-encrypted central MQTT broker (Port 8883) |
| **Tier 3 (Web Client)** | **HTML5 + ES6+ JavaScript** | Zero-build modern frontend (no brittle npm/Webpack toolchain) |
| | **Tailwind CSS (CDN)** | Cyber-physical dark industrial design with glassmorphism |
| | **Chart.js (CDN)** | Real-time rolling timeline with dynamic threshold animation |
| | **Web Audio API** | Dual-harmonic synthesized warning chime (zero audio assets) |
| | **Service Worker (`sw.js`)** | Background push notification handling and window focus |
| | **Web App Manifest** | PWA standalone installation on Android / iOS home screen |

---

## 🧠 Deep Learning & Mathematical Formulation

### 1. Signal Conditioning & Speed-Invariance
Raw vibration signals $x[n]$ are conditioned to remove DC bias and normalized to unit-RMS:
$$\tilde{x}[n] = \frac{x[n] - \mu_x}{\text{RMS}(x - \mu_x)}$$
To ensure variable motor speeds (Low: 25 Hz, Medium: 40 Hz, High: 55 Hz) do not trigger false anomalies, FFT magnitude spectra are order-tracked relative to the motor running frequency ($f / f_{\text{run}}$).

### 2. Autoencoder Anomaly Scoring
The 1D-CNN autoencoder maps the 512-point order-tracked FFT magnitude spectrum $S$ to a compact latent space $z$ and reconstructs $\hat{S}$:
$$z = \text{Encoder}(S), \quad \hat{S} = \text{Decoder}(z)$$
$$\mathcal{L}_{\text{MSE}}(S, \hat{S}) = \frac{1}{K} \sum_{k=1}^K \left( S_k - \hat{S}_k \right)^2$$

### 3. Extreme Value Decision Threshold
Calibrated strictly on healthy baseline data using extreme value statistics:
$$\tau = \mu_{\text{train}} + 3.5 \cdot \sigma_{\text{train}}$$
$$\text{State}(S) = \begin{cases} \text{Normal}, & \text{if } \mathcal{L}_{\text{MSE}} \le \tau \\ \text{Anomaly}, & \text{if } \mathcal{L}_{\text{MSE}} > \tau \end{cases}$$

### 4. Human-in-the-Loop Active Calibration
When an operator flags a false alarm on an anomalous window with score $s_{\text{false}} > \tau$, the system immediately adapts with an engineering margin:
$$\tau_{\text{new}} = \max\left(\tau, \; 1.05 \cdot s_{\text{false}}\right)$$
The server publishes $\tau_{\text{new}}$ down to the edge via MQTT, and the Pi updates its in-memory threshold without restarting the service or dropping a sample.

---

## 📂 Project Structure

```
ESI/
├── client/                          # Tier 3: Web / Mobile Client Application
│   ├── index.html                   # Cyber-physical Single-Page Dashboard
│   ├── manifest.json                # PWA standalone installable metadata
│   ├── sw.js                        # Service Worker for background Web Push alerts
│   ├── client_plan.md               # Tier 3 architectural plan
│   ├── ui_plan.md                   # UI/UX design tokens & wireframes
│   ├── css/styles.css               # Glassmorphism cards & glowing beacon pulses
│   └── js/
│       ├── config.js                # Dynamic host/port resolution
│       ├── api.js                   # REST API client wrapper
│       ├── ws.js                    # Auto-reconnecting WebSocket client
│       ├── audio.js                 # Web Audio API harmonic alarm synthesizer
│       ├── charts.js                # Real-time streaming Chart.js timeline
│       ├── push.js                  # Web Push VAPID subscription manager
│       └── app.js                   # UI state orchestrator & HITL feedback loop
│
├── server/                          # Tier 2: Cloud / Gateway Backend
│   ├── main.py                      # FastAPI app factory, CORS, static mount, and lifespan
│   ├── database.py                  # SQLite WAL time-series storage & push token CRUD
│   ├── schemas.py                   # Pydantic V2 data validation contracts
│   ├── push_notifier.py             # RFC 8030 Web Push alert broadcaster
│   ├── ws_manager.py                # Real-time WebSocket connection manager
│   ├── mqtt_bridge.py               # Bi-directional MQTT bridge (Edge <-> DB <-> Web Push)
│   ├── config.py                    # Server configuration & environment overrides
│   ├── mosquitto.conf               # Mosquitto config with TCP (1883) & WebSockets (9001)
│   ├── Dockerfile                   # Production container definition
│   └── routes/                      # Modular REST routers
│       ├── health.py                # Server heartbeat & edge latency
│       ├── telemetry.py             # Latest & historical telemetry queries
│       ├── alerts.py                # Alert log & technician acknowledgment
│       ├── feedback.py              # Human-in-the-Loop threshold self-tuning
│       ├── config.py                # Remote threshold & sensor mode calibration
│       └── push.py                  # VAPID public key & push subscription endpoints
│
├── service/                         # Tier 1: Headless Edge Service (Raspberry Pi)
│   ├── monitor.py                   # Live edge daemon (Sensor -> Features -> ML -> MQTT)
│   └── mqtt_publisher.py            # Resilient MQTT publisher & remote config subscriber
│
├── hal/                             # Hardware Abstraction Layer
│   ├── base.py                      # VibrationSensor ABC (get_10s_window)
│   ├── physical_sensor.py           # ADXL345 / MPU6050 over I2C (smbus2)
│   └── mock_sensor.py               # In-memory dataset replay sensor
│
├── models/                          # Pre-trained TorchScript Deep Learning Models
│   ├── cwru_unsupervised_xiong_edge.pt # 1D-CNN Autoencoder (healthy data baseline)
│   └── cwru_advanced_edge.pt        # 4-class Fault Classifier (healthy + defects)
│
├── detectors/                       # Anomaly Detectors & Inference Wrappers
│   ├── base.py                      # BaseDetector ABC
│   ├── edge_models.py               # TorchScript inference (Autoencoder, Classifier, Hybrid)
│   ├── threshold.py                 # Statistical RMS threshold detector
│   ├── isolation_forest.py          # Scikit-learn Isolation Forest
│   └── autoencoder.py               # PyTorch training baseline
│
├── colab/                           # GPU Training & Architecture Documentation
│   ├── model.md                     # Mathematical specification & INT8 quantization guide
│   └── motor_anomaly_autoencoder.ipynb # Self-contained Google Colab GPU training notebook
│
├── scripts/                         # CLI & Deployment Scripts
│   ├── pi_setup.sh                  # Automated Raspberry Pi 4 setup script
│   ├── motor-monitor.service        # systemd background service configuration
│   ├── download_real_cwru.py        # Fetches genuine benchmark data from engineering.case.edu
│   ├── validate_data.py             # Signal health check CLI
│   └── record.py                    # Records vibration windows from rig to .npy
│
├── tests/                           # Comprehensive automated test suite (166 tests passing)
│   ├── test_cwru_loader.py          # CWRU dataset parsing & loading
│   ├── test_data_pipeline.py        # Signal cleaning & window splitting
│   ├── test_data_validation.py      # Data integrity checks
│   ├── test_pipeline_e2e.py         # Mock vs Real mode switch tests
│   ├── test_features.py             # Time & frequency domain feature extraction
│   ├── test_detectors.py            # Detector baseline tests
│   ├── test_edge_models.py          # Deep learning TorchScript inference tests
│   ├── test_hal.py                  # HAL & MockSensor tests
│   ├── test_app_service.py          # Edge monitor daemon & remote MQTT config tests
│   ├── test_training_eval.py        # Benchmark evaluation tests
│   ├── test_server_database.py      # SQLite WAL transactions & CRUD tests
│   ├── test_server_api.py           # REST API endpoints & sensor mode tests
│   ├── test_server_mqtt_bridge.py   # MQTT bridge ingestion & threshold dispatch tests
│   ├── test_server_push.py          # Web Push VAPID subscriptions & alert tests
│   ├── test_client_serving.py       # PWA static asset delivery & SW header tests
│   └── test_e2e_edge_server.py      # Full 3-tier end-to-end feedback loop test
│
├── config.yaml                      # Root configuration (sensor, motor, detector, MQTT)
├── render.yaml                      # 1-click Render.com deployment blueprint
├── docker-compose.yml               # Local/VPS Docker stack
├── deployment_guide.md              # 100% Free Cloud Deployment walkthrough
└── pyproject.toml                   # Project metadata & pytest configuration
```

---

## ⚡ Quick Start (Local Development)

### 1. Clone & Install Dependencies
```bash
# Clone the repository
git clone https://github.com/your-username/ESI.git
cd ESI

# Install development dependencies
pip install -r requirements.txt
```

### 2. Download Real Benchmark Data
```bash
python scripts/download_real_cwru.py
# Downloads 97.mat (healthy), 105.mat (inner race), 118.mat (ball), 130.mat (outer race)
```

### 3. Run the Backend & Web App
```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000
```
Open **`http://localhost:8000/`** in your browser. The live dashboard will load with full interactive charts!

### 4. Run the Edge Daemon (Mock Mode on Laptop)
In a second terminal window:
```bash
python -m service.monitor --mock
```
Watch the live anomaly timeline stream updates every 10 seconds. You can switch between **MOCK** and **PHYSICAL** sensor mode directly with the toggle in the dashboard header!

---

## 🧪 Automated Tests

The repository includes **166 automated unit, integration, and end-to-end tests** covering all three tiers:

```bash
pytest -v
```

```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.2.0
collected 166 items

tests/test_app_service.py ....................                          [ 12%]
tests/test_client_serving.py ......                                     [ 15%]
tests/test_cwru_loader.py ...........................                   [ 31%]
tests/test_data_pipeline.py ..........................                  [ 47%]
tests/test_data_validation.py ..........                                [ 53%]
tests/test_detectors.py ..........                                      [ 59%]
tests/test_edge_models.py ..........                                    [ 65%]
tests/test_features.py ..............                                    [ 74%]
tests/test_hal.py ..........                                            [ 80%]
tests/test_pipeline_e2e.py ......                                       [ 83%]
tests/test_server_api.py ........                                       [ 88%]
tests/test_server_database.py .....                                     [ 91%]
tests/test_server_mqtt_bridge.py .....                                  [ 94%]
tests/test_server_push.py ...........                                   [100%]
tests/test_training_eval.py ........                                    [100%]
tests/test_e2e_edge_server.py .                                         [100%]

====================== 166 passed, 12 warnings in 22.47s ======================
```

---

## ☁️ 100% Free Cloud Deployment

For detailed step-by-step instructions with credentials and screenshots, see [`deployment_guide.md`](deployment_guide.md).

```
[ Raspberry Pi 4 ] ──(TLS 8883)──► [ HiveMQ Cloud (Free) ] ◄──(TLS 8883)──► [ Render.com (Free) ]
 (Factory Floor)                      (100 Devices Free)                    (FastAPI + Web App)
```

1. **Free Cloud Broker**: Create a free forever cluster on **[HiveMQ Cloud](https://www.hivemq.com/pricing/)** (100 devices, 10 GB/month free, TLS encrypted on port 8883).
2. **Free Web Server**: Push code to GitHub and connect to **[Render.com](https://render.com/)**. Render automatically reads [`render.yaml`](render.yaml) and gives you a free HTTPS address (e.g. `https://vibraguard-sentinel.onrender.com`).
3. **Edge Daemon on Pi**: Run `bash scripts/pi_setup.sh`, set HiveMQ credentials in `config.yaml`, and start the background daemon:
   ```bash
   sudo systemctl start motor-monitor.service
   ```

---

## 🔌 Hardware Pinout & Wiring

When connecting your MEMS sensor to the Raspberry Pi 4 GPIO header:

| Accelerometer Pin | Signal | Raspberry Pi 4 Physical Pin | Pin Function |
|---|---|---|---|
| **VCC / VIN** | Power | **Pin 1** | 3.3V Power |
| **GND** | Ground | **Pin 6** | Ground |
| **SDA** | I2C Data | **Pin 3** | GPIO 2 (I2C1 SDA) |
| **SCL** | I2C Clock | **Pin 5** | GPIO 3 (I2C1 SCL) |
| **CS** *(ADXL345 only)*| Bus Select | Tie to **Pin 1 (3.3V)** | Selects I2C Mode |
| **SDO/AD0** | Address Select | Tie to **Pin 6 (GND)** | Address `0x53` (ADXL) / `0x68` (MPU) |

Verify hardware connectivity on the Pi:
```bash
i2cdetect -y 1
# Expected output: 53 (ADXL345) or 68 (MPU6050)
```

---

## 📄 License & Attribution

* **Dataset**: Case Western Reserve University (CWRU) Bearing Data Center.
* **Deep Learning Baseline**: Xiong et al. 1D-CNN Autoencoder architecture.
* **License**: MIT License. Open source and free for commercial and academic use.
