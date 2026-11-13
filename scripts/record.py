"""
scripts/record.py — CLI tool to record vibration windows from the sensor.

Saves each 10-second window as a .npy file with a timestamped filename.

Usage
-----
    python scripts/record.py --label healthy --speed medium --duration 60
    python scripts/record.py --label imbalance_5g --speed high --duration 120
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure repo root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import yaml

from hal import PhysicalSensor, MockSensor

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger("record")


def main(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build sensor
    if args.mock:
        mock_data = np.random.randn(120_000).astype(np.float32)   # placeholder
        sensor = MockSensor(mock_data, sample_rate=1000, simulate_delay=False)
        logger.warning("Using MockSensor — data is random noise.  "
                       "Replace with real CWRU data for meaningful tests.")
    else:
        sensor = PhysicalSensor(
            bus_num=args.bus,
            address=int(args.address, 16),
            sample_rate=args.sample_rate,
        )

    total_windows = max(1, args.duration // 10)
    logger.info(
        "Recording %d x 10 s windows  label=%r  speed=%r",
        total_windows, args.label, args.speed,
    )

    for i in range(total_windows):
        window    = sensor.get_10s_window()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"{args.label}_{args.speed}_{timestamp}.npy"
        save_path = output_dir / filename
        np.save(str(save_path), window)
        logger.info(
            "[%d/%d] Saved %s  (samples=%d, rms=%.4f)",
            i + 1, total_windows, filename,
            len(window), float(np.sqrt(np.mean(window ** 2))),
        )

    logger.info("Recording complete.  Files written to %s", output_dir)


def _parse_args():
    p = argparse.ArgumentParser(description="Record vibration windows from sensor")
    p.add_argument("--label",       required=True,
                   help="e.g. healthy, imbalance_5g, loose_mount")
    p.add_argument("--speed",       required=True,
                   help="e.g. low, medium, high")
    p.add_argument("--duration",    type=int, default=60,
                   help="Total recording duration in seconds (rounded to 10 s)")
    p.add_argument("--output-dir",  default="data/raw")
    p.add_argument("--sample-rate", type=int, default=1000)
    p.add_argument("--bus",         type=int, default=1)
    p.add_argument("--address",     default="0x53")
    p.add_argument("--mock",        action="store_true",
                   help="Use MockSensor instead of physical hardware")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
