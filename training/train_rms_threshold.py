"""
training/train_rms_threshold.py

Trains the RMS threshold baseline detector on healthy windows and saves it.

Usage
-----
    python -m training.train_rms_threshold
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

from features import FeatureExtractor
from detectors import RMSThresholdDetector

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger("train_rms_threshold")


def load_healthy_windows(data_dir: str, speed: str = "all") -> list:
    data_path = Path(data_dir)
    pattern   = f"healthy_{speed}_*.npy" if speed != "all" else "healthy_*.npy"
    files     = sorted(data_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No healthy files in {data_dir} matching {pattern}")
    return [np.load(str(f)) for f in files]


def main(args) -> None:
    windows   = load_healthy_windows(args.data_dir, args.speed)
    extractor = FeatureExtractor(sample_rate=args.sample_rate, run_freq_hz=args.run_freq)
    feat_dicts = [extractor.transform(w).to_dict() for w in windows]

    split_idx   = int(len(feat_dicts) * 0.8)
    train_feats = feat_dicts[:split_idx]
    val_feats   = feat_dicts[split_idx:]

    detector = RMSThresholdDetector(sigma=args.sigma)
    detector.fit(train_feats)

    if val_feats:
        detector.set_threshold_from_healthy(val_feats, sigma=args.sigma)

    save_path = Path(args.output_dir) / "rms_threshold.pkl"
    detector.save(str(save_path))
    logger.info("Done.  Model saved to %s", save_path)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",    default="data/raw")
    p.add_argument("--output-dir",  default="detectors/saved")
    p.add_argument("--speed",       default="all")
    p.add_argument("--sample-rate", type=int,   default=1000)
    p.add_argument("--run-freq",    type=float, default=40.0)
    p.add_argument("--sigma",       type=float, default=3.0)
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
