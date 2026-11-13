"""
evaluation/evaluate_rig.py — Evaluate all detectors on the rig dataset.

Loads labelled windows from data/processed/rig_features.csv, runs each
pre-trained detector, and prints a comparison table.

Usage
-----
    python -m evaluation.evaluate_rig
    python -m evaluation.evaluate_rig --features data/processed/rig_features.csv
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure repo root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import numpy as np

from detectors import (
    RMSThresholdDetector,
    IsolationForestDetector,
    AutoencoderDetector,
)
from evaluation.metrics import compute_metrics, print_metrics_table

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger("evaluate_rig")


def run_evaluation(features_path: str, models_dir: str = "detectors/saved") -> None:
    df = pd.read_csv(features_path)

    # Expect columns: label, rms, peak, crest_factor, kurtosis, ...
    # label: 'healthy' → 0, anything else → 1
    y_true = (df["label"] != "healthy").astype(int).tolist()
    feat_dicts = df.drop(columns=["label"]).to_dict(orient="records")

    detectors = {}

    rms_path = Path(models_dir) / "rms_threshold.pkl"
    if rms_path.exists():
        detectors["RMS Threshold"] = RMSThresholdDetector.load(str(rms_path))

    iso_path = Path(models_dir) / "iso_forest.pkl"
    if iso_path.exists():
        detectors["Isolation Forest"] = IsolationForestDetector.load(str(iso_path))

    ae_path = Path(models_dir) / "autoencoder.pt"
    if ae_path.exists():
        fft_cols = [c for c in df.columns if c.startswith("fft_bin_")]
        if "fft_spectrum" in feat_dicts[0]:
            detectors["Autoencoder"] = AutoencoderDetector.load(str(ae_path))
        elif fft_cols:
            ae = AutoencoderDetector.load(str(ae_path))
            for f, row in zip(feat_dicts, df[fft_cols].values):
                f["fft_spectrum"] = row.tolist()
            detectors["Autoencoder"] = ae
        else:
            logger.info(
                "Autoencoder model found, but rig_features.csv contains scalar features only. "
                "Skipping Autoencoder on scalar CSV (use evaluate_cwru for raw window evaluation)."
            )

    if not detectors:
        logger.error("No trained models found in %s.  Run training first.", models_dir)
        return

    results = {}
    for name, det in detectors.items():
        y_pred = []
        for f in feat_dicts:
            _, label_str = det.predict(f)
            y_pred.append(1 if label_str == "anomaly" else 0)
        results[name] = compute_metrics(y_true, y_pred)

    print_metrics_table(results, title="Rig Dataset — Detector Comparison")

    # Per-fault-type breakdown
    for fault in df["label"].unique():
        if fault == "healthy":
            continue
        mask     = (df["label"] == fault) | (df["label"] == "healthy")
        sub_true = (df.loc[mask, "label"] != "healthy").astype(int).tolist()
        sub_feat = df.loc[mask].drop(columns=["label"]).to_dict(orient="records")
        sub_res  = {}
        for name, det in detectors.items():
            y_pred = [1 if det.predict(f)[1] == "anomaly" else 0 for f in sub_feat]
            sub_res[name] = compute_metrics(sub_true, y_pred)
        print_metrics_table(sub_res, title=f"Fault: {fault}")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features",   default="data/processed/rig_features.csv")
    p.add_argument("--models-dir", default="detectors/saved")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_evaluation(args.features, args.models_dir)
