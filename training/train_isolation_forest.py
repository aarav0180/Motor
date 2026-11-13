"""
training/train_isolation_forest.py

Trains the Isolation Forest detector on healthy windows and saves it.

Usage
-----
    python -m training.train_isolation_forest
    python -m training.train_isolation_forest --data-dir data/raw --speed medium
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure repo root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from features import FeatureExtractor
from detectors import IsolationForestDetector

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger("train_iso_forest")


def load_healthy_windows(data_dir: str, speed: str = "all") -> list:
    """
    Load all healthy .npy files from data_dir.

    File naming convention: healthy_<speed>_<timestamp>.npy
    """
    data_path = Path(data_dir)
    pattern   = f"healthy_{speed}_*.npy" if speed != "all" else "healthy_*.npy"
    files     = sorted(data_path.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No healthy files matching '{pattern}' in {data_dir}"
        )

    windows = [np.load(str(f)) for f in files]
    logger.info("Loaded %d healthy windows from %s", len(windows), data_dir)
    return windows


def main(args) -> None:
    sample_rate = args.sample_rate
    run_freq    = args.run_freq

    windows   = load_healthy_windows(args.data_dir, args.speed)
    extractor = FeatureExtractor(sample_rate=sample_rate, run_freq_hz=run_freq)

    logger.info("Extracting features from %d windows …", len(windows))
    feat_dicts = [extractor.transform(w).to_dict() for w in windows]

    # Train/validation split (80/20)
    split_idx = int(len(feat_dicts) * 0.8)
    train_feats = feat_dicts[:split_idx]
    val_feats   = feat_dicts[split_idx:]

    detector = IsolationForestDetector(contamination=0.05, n_estimators=100)
    detector.fit(train_feats)

    if val_feats:
        detector.set_threshold_from_healthy(val_feats, sigma=3.0)

    save_path = Path(args.output_dir) / "iso_forest.pkl"
    detector.save(str(save_path))
    logger.info("Done.  Model saved to %s", save_path)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",    default="data/raw")
    p.add_argument("--output-dir",  default="detectors/saved")
    p.add_argument("--speed",       default="all",
                   help="'low', 'medium', 'high', or 'all'")
    p.add_argument("--sample-rate", type=int, default=1000)
    p.add_argument("--run-freq",    type=float, default=40.0,
                   help="Motor running frequency in Hz")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
