#!/usr/bin/env bash
# ==============================================================================
# scripts/pi_setup.sh — Automated Raspberry Pi 4 Edge Daemon Installer
# Installs dependencies, enables I2C hardware, sets up venv & systemd service.
# ==============================================================================

set -euo pipefail

echo "=========================================================="
echo "   VibraGuard Edge Sentinel — Raspberry Pi Setup Script    "
echo "=========================================================="

# 1. Update OS packages and install I2C development tools
echo "[1/5] Updating system packages & installing I2C tools..."
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv python3-dev i2c-tools libatlas-base-dev

# 2. Enable Hardware I2C interface
echo "[2/5] Enabling hardware I2C interface..."
sudo raspi-config nonint do_i2c 0 || true

# 3. Create dedicated Python virtual environment
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "[3/5] Setting up virtual environment in $PROJECT_DIR/venv..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# 4. Install ultra-light edge dependencies
echo "[4/5] Installing edge dependencies from requirements-pi.txt..."
pip install --upgrade pip
pip install -r requirements-pi.txt

# 5. Configure and enable systemd background service
echo "[5/5] Installing systemd daemon..."
SERVICE_FILE="$PROJECT_DIR/scripts/motor-monitor.service"
if [ -f "$SERVICE_FILE" ]; then
    sudo cp "$SERVICE_FILE" /etc/systemd/system/motor-monitor.service
    sudo systemctl daemon-reload
    sudo systemctl enable motor-monitor.service
    echo "Service installed! Control with:"
    echo "  sudo systemctl start motor-monitor.service"
    echo "  sudo journalctl -u motor-monitor.service -f"
fi

echo "=========================================================="
echo "   Setup Complete! To verify I2C sensor wiring run:       "
echo "   i2cdetect -y 1                                         "
echo "   Expected address: 0x53 (ADXL345) or 0x68 (MPU6050)     "
echo "=========================================================="
