"""
scripts/run_pipeline_demo.py
----------------------------
Demonstrates the full data pipeline with the single-switch pattern.

Change DATA_MODE at the top to switch between mock and real sensor.

Usage
-----
    python scripts/run_pipeline_demo.py
    python scripts/run_pipeline_demo.py --mode real
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so we can import without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from data.pipeline import DataPipeline, DataMode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline_demo")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ★★★  THE ONE SWITCH  ★★★
#  Change this to DataMode.REAL when running on the Raspberry Pi
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA_MODE = DataMode.MOCK
# DATA_MODE = DataMode.REAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main(mode: DataMode, cwru_dir: str, n_live_windows: int) -> None:
    log.info("Starting pipeline demo  mode=%s", mode.name)

    # ── 1. Build and set up the pipeline ──────────────────────────────
    pipeline = DataPipeline(
        mode=mode,
        cwru_dir=cwru_dir,
        window_seconds=10,
        mock_fraction=0.20,
        train_fraction=0.80,
    )

    if not list(Path(cwru_dir).glob("*.mat")):
        log.warning(
            "No CWRU .mat files found in '%s'.\n"
            "  Using synthetic random data for demonstration.\n"
            "  Download real data from: "
            "https://engineering.case.edu/bearingdatacenter",
            cwru_dir,
        )
        _inject_synthetic_data(pipeline)
    else:
        pipeline.setup()

    # ── 2. Training data ──────────────────────────────────────────────
    X_train, train_feats = pipeline.get_train_data()
    log.info(
        "Training data: %d healthy windows  shape=%s",
        len(train_feats), X_train.shape,
    )
    _print_feature_sample(train_feats[0], label="Sample training feature")

    # ── 3. Evaluation data ────────────────────────────────────────────
    X_eval, eval_feats, y_true = pipeline.get_eval_data()
    n_healthy = sum(1 for y in y_true if y == 0)
    n_faulty  = sum(1 for y in y_true if y == 1)
    log.info(
        "Eval data: %d total  (healthy=%d, faulty=%d)",
        len(eval_feats), n_healthy, n_faulty,
    )

    # ── 4. Simulate the live sensing loop ─────────────────────────────
    sensor    = pipeline.get_sensor()
    extractor = pipeline.get_extractor()

    log.info(
        "Sensor: %r  — running %d windows …", sensor, n_live_windows
    )

    for i in range(n_live_windows):
        window   = sensor.get_10s_window()
        feat_dict = pipeline.extract_live_features(window)
        rms_val  = feat_dict["rms"]
        log.info(
            "  [window %d/%d]  len=%d  rms=%.4f  peak=%.4f  crest=%.4f",
            i + 1, n_live_windows,
            len(window), rms_val,
            feat_dict["peak"], feat_dict["crest_factor"],
        )

    log.info("Demo complete.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_feature_sample(feat: dict, label: str = "Features") -> None:
    print(f"\n  {label}:")
    for k, v in feat.items():
        print(f"    {k:<20} = {v:.6f}" if isinstance(v, float) else f"    {k:<20} = {v}")
    print()


def _inject_synthetic_data(pipeline: DataPipeline) -> None:
    """
    When no CWRU files exist, fill the pipeline with synthetic random data
    so the demo still runs end-to-end.
    """
    from data.windows  import LabelledWindow
    from data.splitter import Split
    from hal           import MockSensor
    from features      import FeatureExtractor
    from data.loader   import CWRU_SAMPLE_RATE_HZ, CWRU_RUN_FREQ_HZ
    from data.cleaner  import SignalCleaner

    rng = np.random.default_rng(99)
    FS  = CWRU_SAMPLE_RATE_HZ
    WS  = FS * 10

    def _win(label, seed):
        s = rng.standard_normal(WS).astype(np.float32)
        if label != "healthy":
            s[::200] += 5.0  # inject impulses for faulty windows
        return LabelledWindow(signal=s, label=label, source="synthetic")

    healthy = [_win("healthy", i)       for i in range(50)]
    faulty  = [_win("inner_race", i+100) for i in range(20)]

    from data.splitter import DataSplitter
    splitter = DataSplitter(mock_fraction=0.2, train_fraction=0.8, seed=42)
    split    = splitter.split(healthy + faulty)

    combined = np.concatenate([w.signal for w in split.mock_sensor])
    sensor   = MockSensor(combined, sample_rate=FS, simulate_delay=False)

    pipeline._records   = []
    pipeline._split     = split
    pipeline._sensor    = sensor
    pipeline._cleaner   = SignalCleaner(sample_rate=FS)
    pipeline._extractor = FeatureExtractor(sample_rate=FS, run_freq_hz=CWRU_RUN_FREQ_HZ)
    pipeline._is_ready  = True
    split.summary()


def _parse_args():
    p = argparse.ArgumentParser(description="Data pipeline demo")
    p.add_argument(
        "--mode", choices=["mock", "real"], default="mock",
        help="'mock' uses CWRU data, 'real' uses physical sensor",
    )
    p.add_argument("--cwru-dir",      default="data/cwru")
    p.add_argument("--n-live-windows", type=int, default=3,
                   help="Number of windows to pull from the sensor in the demo")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    mode = DataMode.REAL if args.mode == "real" else DataMode.MOCK
    main(mode, args.cwru_dir, args.n_live_windows)
