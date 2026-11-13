"""
evaluation/evaluate_cwru.py — Cross-dataset validation on CWRU bearing data.

Loads CWRU .mat files, segments into 10-second windows (at 12 kHz),
extracts features, and runs the pre-trained detectors.
Healthy = 'Normal_*.mat', Faulty = any file with 'IR', 'OR', 'B' in name.

Usage
-----
    python -m evaluation.evaluate_cwru
    python -m evaluation.evaluate_cwru --cwru-dir data/cwru --models-dir detectors/saved
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure repo root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import scipy.io

from features import FeatureExtractor
from detectors import (
    RMSThresholdDetector,
    IsolationForestDetector,
    AutoencoderDetector,
)
from evaluation.metrics import compute_metrics, print_metrics_table

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger("evaluate_cwru")

CWRU_SAMPLE_RATE = 12_000   # Hz
CWRU_RUN_FREQ    = 29.95    # Hz — CWRU motor runs at ~1797 rpm


def load_cwru_signal(mat_path: Path) -> np.ndarray:
    """Return the drive-end vibration channel from a CWRU .mat file."""
    mat = scipy.io.loadmat(str(mat_path))
    # DE channel key looks like 'X097_DE_time'
    de_key = next(
        (k for k in mat if k.endswith("_DE_time") and not k.startswith("_")),
        None,
    )
    if de_key is None:
        raise KeyError(f"No DE channel found in {mat_path}.  Keys: {list(mat.keys())}")
    return mat[de_key].flatten().astype(np.float32)


def segment_signal(
    signal: np.ndarray,
    sample_rate: int,
    window_seconds: int = 10,
) -> List[np.ndarray]:
    """Split a long signal into non-overlapping 10-second windows."""
    window_size = sample_rate * window_seconds
    n_windows   = len(signal) // window_size
    return [signal[i * window_size : (i + 1) * window_size] for i in range(n_windows)]


from data.loader import CWRULoader, LABEL_HEALTHY


def collect_windows(cwru_dir: Path) -> Tuple[List[np.ndarray], List[int]]:
    """
    Walk the CWRU directory, label files, segment, and return (windows, labels).
    Label 0 = healthy, 1 = faulty.
    """
    windows, labels = [], []
    for mat_path in sorted(cwru_dir.glob("*.mat")):
        fault_type, _, _ = CWRULoader._parse_filename(mat_path.stem)
        label = 0 if fault_type == LABEL_HEALTHY else 1
        try:
            signal = load_cwru_signal(mat_path)
        except Exception as exc:
            logger.warning("Skipping %s: %s", mat_path.name, exc)
            continue
        segs = segment_signal(signal, CWRU_SAMPLE_RATE)
        windows.extend(segs)
        labels.extend([label] * len(segs))
        logger.info(
            "Loaded %s — label=%d  windows=%d", mat_path.name, label, len(segs)
        )
    return windows, labels


def run_evaluation(cwru_dir: str, models_dir: str) -> None:
    cwru_path   = Path(cwru_dir)
    models_path = Path(models_dir)
    extractor   = FeatureExtractor(
        sample_rate=CWRU_SAMPLE_RATE, run_freq_hz=CWRU_RUN_FREQ
    )

    windows, y_true = collect_windows(cwru_path)
    if not windows:
        logger.error("No .mat files found in %s", cwru_dir)
        return

    logger.info(
        "Total windows: %d  (healthy=%d, faulty=%d)",
        len(windows), y_true.count(0), y_true.count(1),
    )

    feat_dicts = [extractor.transform(w).to_dict() for w in windows]

    # Load detectors
    detectors = {}
    for name, fname, cls in [
        ("RMS Threshold",   "rms_threshold.pkl", RMSThresholdDetector),
        ("Isolation Forest","iso_forest.pkl",     IsolationForestDetector),
    ]:
        p = models_path / fname
        if p.exists():
            detectors[name] = cls.load(str(p))

    ae_path = models_path / "autoencoder.pt"
    if ae_path.exists():
        ae = AutoencoderDetector.load(str(ae_path))
        # Add fft_spectrum to feature dicts for the autoencoder
        ae_feats = []
        for w, fd in zip(windows, feat_dicts):
            fd_copy = dict(fd)
            fd_copy.update(ae.prepare_feature(w))
            ae_feats.append(fd_copy)
        detectors["Autoencoder"] = (ae, ae_feats)

    results = {}
    for name, det_or_pair in detectors.items():
        if isinstance(det_or_pair, tuple):
            det, feats = det_or_pair
        else:
            det, feats = det_or_pair, feat_dicts
        y_pred = [1 if det.predict(f)[1] == "anomaly" else 0 for f in feats]
        results[name] = compute_metrics(y_true, y_pred)

    print_metrics_table(results, title="CWRU Bearing Dataset — Cross-Dataset Results")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cwru-dir",   default="data/cwru")
    p.add_argument("--models-dir", default="detectors/saved")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_evaluation(args.cwru_dir, args.models_dir)
