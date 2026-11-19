#!/usr/bin/env python3
"""
scripts/calibrate_sensor.py — Physical Accelerometer & Baseline Calibration Tool.

Analyzes raw live X, Y, Z accelerometer data from the Raspberry Pi,
removes static 1g gravity offset, determines dynamic vibration levels,
runs the baseline signal through the anomaly model, and calculates
the optimal alarm threshold.

Usage:
------
    python scripts/calibrate_sensor.py
    python scripts/calibrate_sensor.py --duration 10 --save
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import yaml

from detectors.edge_models import CWRUUnsupervisedDetector, CWRUAdvancedClassifierDetector

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger("calibrate")


def run_calibration(duration: int = 10, save_to_config: bool = False, config_path: str = "config.yaml") -> None:
    cfg_file = Path(config_path)
    cfg = {}
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    sensor_cfg = cfg.get("sensor", {})
    bus_num = int(sensor_cfg.get("bus", 1))
    addr_val = sensor_cfg.get("address", "0x53")
    address = int(addr_val, 16) if isinstance(addr_val, str) else int(addr_val)
    sample_rate = int(sensor_cfg.get("sample_rate", 1000))

    print("\n" + "=" * 65)
    print("      VibraGuard Physical Sensor & Threshold Calibration      ")
    print("=" * 65)
    print(f"I2C Bus: {bus_num} | Address: 0x{address:02X} | Target Rate: {sample_rate} Hz")
    print(f"Calibration Duration: {duration} seconds")
    print("-" * 65)
    print("👉 Ensure your motor is running normally under HEALTHY conditions.")
    input("Press [ENTER] to start sampling...")

    try:
        import smbus2
        bus = smbus2.SMBus(bus_num)
    except ImportError:
        print("\n❌ Error: smbus2 is not installed. Install via: pip install smbus2")
        return
    except Exception as exc:
        print(f"\n❌ Error opening I2C bus {bus_num}: {exc}")
        return

    # Configure ADXL345: full resolution, +/-16g range to avoid clipping
    try:
        bus.write_byte_data(address, 0x2C, 0x0C)  # BW_RATE: 400 Hz
        bus.write_byte_data(address, 0x2D, 0x08)  # POWER_CTL: Measure mode
        bus.write_byte_data(address, 0x31, 0x0B)  # DATA_FORMAT: Full-Res, +/- 16g
        time.sleep(0.05)
    except Exception as exc:
        print(f"\n❌ Failed to configure ADXL345: {exc}")
        return

    print("\n⏳ Collecting baseline accelerometer data...")
    x_raw, y_raw, z_raw = [], [], []
    t_start = time.perf_counter()
    target_samples = sample_rate * duration
    delay = 1.0 / sample_rate

    scale = 0.00390625  # 3.9 mg/LSB in full-resolution mode

    for _ in range(target_samples):
        try:
            data = bus.read_i2c_block_data(address, 0x32, 6)
            x = np.int16((data[1] << 8) | data[0]) * scale
            y = np.int16((data[3] << 8) | data[2]) * scale
            z = np.int16((data[5] << 8) | data[4]) * scale
            x_raw.append(x)
            y_raw.append(y)
            z_raw.append(z)
        except Exception:
            pass
        time.sleep(delay)

    bus.close()
    elapsed = time.perf_counter() - t_start
    n_samples = len(x_raw)
    achieved_hz = n_samples / max(elapsed, 1e-4)

    print(f"✅ Captured {n_samples} samples in {elapsed:.2f} s ({achieved_hz:.1f} Hz actual rate)")

    ax = np.array(x_raw, dtype=np.float64)
    ay = np.array(y_raw, dtype=np.float64)
    az = np.array(z_raw, dtype=np.float64)

    # 1. Static Gravity Analysis
    mean_x, mean_y, mean_z = float(np.mean(ax)), float(np.mean(ay)), float(np.mean(az))
    static_mag = float(np.sqrt(mean_x**2 + mean_y**2 + mean_z**2))

    print("\n1. Static Gravity Vector (DC Bias):")
    print(f"   X: {mean_x:+.2f} g")
    print(f"   Y: {mean_y:+.2f} g")
    print(f"   Z: {mean_z:+.2f} g")
    print(f"   Total Static Gravity |A|: {static_mag:.2f} g  (Expected ~1.0 g)")

    # 2. Dynamic Vibration (Pure AC Mechanical Motion)
    dx = ax - mean_x
    dy = ay - mean_y
    dz = az - mean_z
    dyn_mag = np.sqrt(dx**2 + dy**2 + dz**2)

    rms_x = float(np.sqrt(np.mean(dx**2)))
    rms_y = float(np.sqrt(np.mean(dy**2)))
    rms_z = float(np.sqrt(np.mean(dz**2)))
    rms_dyn = float(np.sqrt(np.mean(dyn_mag**2)))
    peak_dyn = float(np.max(dyn_mag))

    print("\n2. Dynamic Vibration Metrics (Mechanical Motion without Gravity):")
    print(f"   X-Axis Dynamic RMS: {rms_x:.4f} g  (Peak: {np.max(np.abs(dx)):.2f} g)")
    print(f"   Y-Axis Dynamic RMS: {rms_y:.4f} g  (Peak: {np.max(np.abs(dy)):.2f} g)")
    print(f"   Z-Axis Dynamic RMS: {rms_z:.4f} g  (Peak: {np.max(np.abs(dz)):.2f} g)")
    print(f"   3-Axis Vector RMS : {rms_dyn:.4f} g  (Peak: {peak_dyn:.2f} g)")

    # Identify dominant vibration axis
    axis_rms = {"x": rms_x, "y": rms_y, "z": rms_z}
    dominant_axis = max(axis_rms, key=axis_rms.get)
    print(f"   👉 Primary Vibration Axis: {dominant_axis.upper()}-Axis")

    # 3. Model Anomaly Score Evaluation
    anom_path = cfg.get("detector", {}).get("unsupervised_path", "models/best_anomaly_model.pt")
    multi_path = cfg.get("detector", {}).get("classifier_path", "models/best_multiclass_model.pt")

    print("\n3. Baseline AI Model Evaluation:")
    try:
        det = CWRUUnsupervisedDetector(model_path=anom_path, threshold=0.08)
        # Score over consecutive 1000-sample windows
        sub_scores = []
        step = min(1000, len(dyn_mag))
        for i in range(0, len(dyn_mag) - step + 1, step):
            sub = dyn_mag[i : i + step]
            feat = det.prepare_feature(sub)
            sub_scores.append(det.score(feat))

        mean_score = float(np.mean(sub_scores))
        max_score = float(np.max(sub_scores))
        std_score = float(np.std(sub_scores))
        print(f"   Model Anomaly Score Mean: {mean_score:.4f}")
        print(f"   Model Anomaly Score Max : {max_score:.4f}")
        print(f"   Model Score Std Dev (σ) : {std_score:.4f}")

        # Optimal threshold = mean + 3.5*std (or max * 1.25, whichever is safer)
        opt_thresh = float(round(max(0.08, max_score * 1.25, mean_score + 3.5 * std_score), 4))
    except Exception as exc:
        logger.warning("Could not evaluate anomaly model: %s", exc)
        opt_thresh = 0.08
        mean_score = 0.04

    try:
        multi_det = CWRUAdvancedClassifierDetector(model_path=multi_path)
        diag = multi_det.diagnose(multi_det.prepare_feature(dyn_mag[:1000]))
        print(f"   Multiclass Model Diagnosis : {diag['label'].upper()} (Confidence: {diag['confidence']*100:.1f}%)")
    except Exception as exc:
        logger.warning("Could not evaluate multiclass model: %s", exc)

    print("\n" + "=" * 65)
    print("              CALIBRATION RECOMMENDATIONS                     ")
    print("=" * 65)
    print(f"• Recommended Model Anomaly Threshold: {opt_thresh:.4f}")
    print(f"• Recommended RMS Threshold          : {rms_dyn * 1.8:.4f} g  (1.8x healthy baseline)")
    print(f"• Recommended Sensor Axis Mode       : magnitude (or '{dominant_axis}')")
    print(f"• Recommended ADXL345 Range          : 16g (to prevent saturation)")
    print("=" * 65)

    if save_to_config:
        should_save = True
    else:
        ans = input(f"\nApply threshold ({opt_thresh:.4f}) to config.yaml? [Y/n]: ").strip().lower()
        should_save = ans in ("y", "yes", "")

    if should_save and cfg_file.exists():
        cfg.setdefault("detector", {})["alert_threshold"] = opt_thresh
        cfg.setdefault("sensor", {})["axis"] = "magnitude"
        cfg.setdefault("sensor", {})["range_g"] = 16
        with open(cfg_file, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"🎉 Updated {cfg_file} with alert_threshold={opt_thresh:.4f}, axis=magnitude, range_g=16g!")
        print("You can now restart your monitor service: sudo systemctl restart motor-monitor.service")


def main():
    parser = argparse.ArgumentParser(description="Calibrate physical sensor baseline and compute optimal threshold")
    parser.add_argument("--duration", type=int, default=10, help="Calibration duration in seconds (default: 10)")
    parser.add_argument("--save", action="store_true", help="Auto-save recommended threshold to config.yaml")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    run_calibration(duration=args.duration, save_to_config=args.save, config_path=args.config)


if __name__ == "__main__":
    main()
