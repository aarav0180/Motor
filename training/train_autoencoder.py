"""
training/train_autoencoder.py

Trains the FFT-spectrum autoencoder on healthy windows and saves it.

Usage
-----
    python -m training.train_autoencoder
    python -m training.train_autoencoder --epochs 100 --speed medium
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

from detectors import AutoencoderDetector

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger("train_autoencoder")


def load_healthy_windows(data_dir: str, speed: str = "all") -> list:
    data_path = Path(data_dir)
    pattern   = f"healthy_{speed}_*.npy" if speed != "all" else "healthy_*.npy"
    files     = sorted(data_path.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No healthy files matching '{pattern}' in {data_dir}"
        )

    windows = [np.load(str(f)) for f in files]
    logger.info("Loaded %d healthy windows", len(windows))
    return windows


def main(args) -> None:
    windows = load_healthy_windows(args.data_dir, args.speed)

    # Train/validation split (80/20)
    split_idx    = int(len(windows) * 0.8)
    train_windows = windows[:split_idx]
    val_windows   = windows[split_idx:]

    detector = AutoencoderDetector(
        n_fft_bins=args.n_fft_bins,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    logger.info("Training autoencoder on %d healthy windows …", len(train_windows))
    detector.fit(features=[], raw_windows=train_windows)

    if val_windows:
        # Recompute threshold from held-out validation set
        val_errors = [
            detector._reconstruction_error(
                AutoencoderDetector.compute_fft_spectrum(w, args.n_fft_bins)
            )
            for w in val_windows
        ]
        import numpy as np
        mu, std = np.mean(val_errors), np.std(val_errors)
        detector.threshold = float(mu + 3 * std)
        logger.info(
            "Validation threshold (mu+3σ): %.6f (mu=%.6f, σ=%.6f)",
            detector.threshold, mu, std,
        )

    save_path = Path(args.output_dir) / "autoencoder.pt"
    detector.save(str(save_path))
    logger.info("Done.  Model saved to %s", save_path)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",    default="data/raw")
    p.add_argument("--output-dir",  default="detectors/saved")
    p.add_argument("--speed",       default="all")
    p.add_argument("--n-fft-bins",  type=int, default=512)
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--batch-size",  type=int, default=32)
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
