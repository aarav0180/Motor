# 🚀 100% Free Cloud & Edge Deployment Guide

> **Stack**: HiveMQ Cloud (Free Forever) + Render.com (Free Forever) + Raspberry Pi 4 (systemd)  
> **Total Monthly Cost**: **$0.00**

---

## 1. Step 1: Set Up Free Cloud MQTT Broker (HiveMQ Cloud)

**Time required**: 2 minutes.

1. Go to **[hivemq.com/pricing](https://www.hivemq.com/pricing/)** and click **"Sign Up Free"** under HiveMQ Cloud Free (100 devices, 10 GB/month free forever).
2. Create a free cluster (select AWS Frankfurt, Oregon, or Ohio).
3. Once created, click on **"Access Management"** $\to$ **"Add Credentials"**:
   * **Username**: `vibraguard_user` (or your choice)
   * **Password**: `your_strong_password`
4. Copy your **Cluster URL** from the overview page:
   * Example: `a1b2c3d4e5f6.s1.eu.hivemq.cloud`
   * **Port**: `8883` (TLS)

---

## 2. Step 2: Deploy Server & Web App (Render.com)

**Time required**: 3 minutes.

1. Push your code to your GitHub repository.
2. Sign in to **[render.com](https://render.com/)** with your GitHub account.
3. Click **"New +"** $\to$ **"Web Service"** $\to$ Connect your repository.
4. Render will automatically read [`render.yaml`](render.yaml) or you can configure manually:
   * **Runtime**: Python 3
   * **Build Command**: `pip install -r requirements-server.txt`
   * **Start Command**: `uvicorn server.main:app --host 0.0.0.0 --port $PORT`
   * **Plan**: Free
5. In the **Environment Variables** section, add your HiveMQ Cloud credentials:
   | Key | Value |
   |---|---|
   | `MQTT_BROKER` | `a1b2c3d4e5f6.s1.eu.hivemq.cloud` (your cluster URL) |
   | `MQTT_PORT` | `8883` |
   | `MQTT_USE_TLS` | `true` |
   | `MQTT_USERNAME` | `vibraguard_user` |
   | `MQTT_PASSWORD` | `your_strong_password` |
6. Click **"Deploy Web Service"**.
7. Once deployed, Render will provide a free HTTPS URL:
   * Example: `https://vibraguard-sentinel.onrender.com`
   * Open this URL in any browser or phone to see your live dashboard!

---

## 3. Step 3: Deploy to Raspberry Pi 4 (Edge Layer)

**Time required**: 5 minutes.

### 1. Hardware Pinout (ADXL345 / MPU6050 to Pi 4 GPIO)
```
Accelerometer Pin       Raspberry Pi 4 Physical Pin
─────────────────       ───────────────────────────
VCC / VIN         ───►  Pin 1  (3.3V Power)
GND               ───►  Pin 6  (Ground)
SDA               ───►  Pin 3  (GPIO 2 / I2C1 SDA)
SCL               ───►  Pin 5  (GPIO 3 / I2C1 SCL)
CS (if ADXL345)   ───►  Tie to 3.3V (Selects I2C mode)
SDO/AD0           ───►  Tie to GND (Selects address 0x53 or 0x68)
```

### 2. Run Automated Setup on Pi
Clone your repository onto the Pi and run the installer:
```bash
git clone https://github.com/your-username/ESI.git /home/pi/ESI
cd /home/pi/ESI
bash scripts/pi_setup.sh
```

### 3. Configure `config.yaml` on the Pi
Open `config.yaml`:
```yaml
sensor:
  type: physical          # live I2C sensor
  bus: 1
  address: "0x53"         # or 0x68 for MPU6050

mqtt:
  broker: "a1b2c3d4e5f6.s1.eu.hivemq.cloud"
  port: 8883
  username: "vibraguard_user"
  password: "your_strong_password"
  use_tls: true
```

### 4. Start & Verify the Service
```bash
# Verify sensor is detected on I2C bus:
i2cdetect -y 1
# Expected: 53 (ADXL345) or 68 (MPU6050)

# Start background systemd daemon:
sudo systemctl start motor-monitor.service

# View live stream logs:
sudo journalctl -u motor-monitor.service -f
```

---

## 4. Live Operational Checklist

1. The Raspberry Pi samples 10-second vibration windows, computes 1D-CNN anomaly scores, and publishes to HiveMQ Cloud over port 8883 (TLS).
2. Render server ingests telemetry, updates SQLite WAL, and streams to your open web browser via WebSockets.
3. If an anomaly occurs while your phone is asleep, the **Service Worker triggers a Web Push notification** on your lock screen.
4. Tapping the notification opens the dashboard, allowing you to review the vibration snapshot and click **[False Alarm]** (dynamically adapts edge threshold $\tau_{\text{new}} = \text{score} \times 1.05$) or **[Confirm Fault]**.
