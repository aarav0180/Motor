"""
data/pipeline.py  —  Single-switch data pipeline orchestrator.

╔══════════════════════════════════════════════════════════════════╗
║                  THE ONE SWITCH                                  ║
║                                                                  ║
║   DATA_MODE = DataMode.MOCK    →  use CWRU subset as sensor     ║
║   DATA_MODE = DataMode.REAL    →  use physical ADXL345/MPU6050  ║
║                                                                  ║
║   Everything else stays identical.                               ║
╚══════════════════════════════════════════════════════════════════╝

In MOCK mode
------------
  - 20 % of CWRU healthy windows are served through MockSensor
    (same interface as PhysicalSensor — the rest of the code is blind
    to the difference)
  - 80 % of healthy windows are used to train detectors
  - ALL CWRU faulty windows + held-out healthy are used for evaluation

In REAL mode
------------
  - PhysicalSensor (live I2C) provides the stream of windows
  - ALL CWRU data (healthy + faulty) is used for cross-dataset evaluation
  - Detector training data must be collected separately with scripts/record.py

Usage
-----
    from data.pipeline import DataPipeline, DataMode

    # ── CHANGE THIS ONE LINE ──
    DATA_MODE = DataMode.MOCK
    # DATA_MODE = DataMode.REAL

    pipeline = DataPipeline(mode=DATA_MODE)
    pipeline.setup()

    sensor   = pipeline.get_sensor()          # MockSensor or PhysicalSensor
    split    = pipeline.get_split()           # Split (training / eval / mock)

    # Training data for detectors
    train_windows    = split.train_healthy    # List[LabelledWindow]
    train_feat_dicts = pipeline.extract_features(train_windows)

    # Evaluation set
    eval_windows     = split.all_eval
    eval_feat_dicts  = pipeline.extract_features(eval_windows)
    eval_labels      = split.eval_labels      # List[int]  0=healthy 1=faulty
"""

import logging
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional

import numpy as np

from .loader   import CWRULoader, CWRURecord, CWRU_SAMPLE_RATE_HZ, CWRU_RUN_FREQ_HZ
from .cleaner  import SignalCleaner, CleanerConfig
from .splitter import DataSplitter, Split
from .windows  import LabelledWindow, windows_to_arrays
from features  import FeatureExtractor
from hal       import MockSensor, PhysicalSensor, VibrationSensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ★  THE SWITCH  ★
# ---------------------------------------------------------------------------

class DataMode(Enum):
    """
    Controls where the live sensor data comes from.

    MOCK  →  CWRU healthy subset replayed through MockSensor
    REAL  →  Live ADXL345/MPU6050 via PhysicalSensor (Raspberry Pi)
    """
    MOCK = auto()
    REAL = auto()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class DataPipeline:
    """
    Orchestrates loading, cleaning, splitting, and feature extraction.

    Parameters
    ----------
    mode : DataMode
        DataMode.MOCK or DataMode.REAL  — the single switch.
    cwru_dir : str
        Path to the directory containing CWRU .mat files.
    sample_rate : int
        Sample rate for the live sensor (ADXL345 ≈ 1 000 Hz).
        CWRU data is always treated at CWRU_SAMPLE_RATE_HZ (12 000 Hz).
    window_seconds : int
        Window length in seconds (must match what the sensor returns).
    run_freq_hz : float
        Motor running frequency in Hz (for the live sensor / mock sensor).
        CWRU cross-dataset evaluation always uses CWRU_RUN_FREQ_HZ.
    cleaner_config : CleanerConfig, optional
        Cleaning parameters.  Defaults chosen automatically per mode.
    mock_fraction : float
        Fraction of healthy CWRU windows reserved for MockSensor.
    train_fraction : float
        Fraction of remaining healthy windows used for detector training.
    seed : int
        Random seed for reproducible splits.
    pi_bus / pi_address / pi_sample_rate :
        Physical sensor parameters (used only in REAL mode).
    """

    def __init__(
        self,
        mode:            DataMode  = DataMode.MOCK,
        cwru_dir:        str       = "data/cwru",
        sample_rate:     int       = 1_000,
        window_seconds:  int       = 10,
        run_freq_hz:     float     = 40.0,
        cleaner_config:  Optional[CleanerConfig] = None,
        mock_fraction:   float     = 0.20,
        train_fraction:  float     = 0.80,
        seed:            int       = 42,
        pi_bus:          int       = 1,
        pi_address:      int       = 0x53,
        pi_sample_rate:  int       = 1_000,
    ) -> None:
        self.mode           = mode
        self.cwru_dir       = Path(cwru_dir)
        self.sample_rate    = sample_rate
        self.window_seconds = window_seconds
        self.run_freq_hz    = run_freq_hz
        self.mock_fraction  = mock_fraction
        self.train_fraction = train_fraction
        self.seed           = seed
        self.pi_bus         = pi_bus
        self.pi_address     = pi_address
        self.pi_sample_rate = pi_sample_rate

        # Auto-select cleaner preset
        if cleaner_config is not None:
            self.cleaner_config = cleaner_config
        elif mode == DataMode.MOCK:
            self.cleaner_config = CleanerConfig.for_cwru()
        else:
            self.cleaner_config = CleanerConfig.for_adxl345()

        # Internals set by setup()
        self._records:   Optional[List[CWRURecord]] = None
        self._split:     Optional[Split]            = None
        self._sensor:    Optional[VibrationSensor]  = None
        self._cleaner:   Optional[SignalCleaner]    = None
        self._extractor: Optional[FeatureExtractor] = None
        self._is_ready:  bool                       = False

    # ------------------------------------------------------------------
    # Setup — call this once before using the pipeline
    # ------------------------------------------------------------------

    def setup(self) -> "DataPipeline":
        """
        Load, clean and split the CWRU dataset.

        Always called regardless of mode so CWRU data is always available
        for cross-dataset evaluation.

        Returns self for chaining.
        """
        logger.info("DataPipeline.setup() — mode=%s", self.mode.name)

        # 1. Load CWRU
        self._records = self._load_cwru()

        # 2. Create cleaner (CWRU rate)
        self._cleaner = SignalCleaner(self.cleaner_config, sample_rate=CWRU_SAMPLE_RATE_HZ)

        # 3. Segment into windows and split
        window_size = CWRU_SAMPLE_RATE_HZ * self.window_seconds
        splitter    = DataSplitter(
            mock_fraction=self.mock_fraction,
            train_fraction=self.train_fraction,
            seed=self.seed,
        )
        self._split = splitter.split_records(
            self._records,
            window_size=window_size,
            cleaner=self._cleaner,
        )

        # 4. Build sensor
        self._sensor = self._build_sensor()

        # 5. Build feature extractor
        #    For mock/CWRU data use CWRU rate + CWRU run freq
        #    For real sensor use the configured rate + run freq
        if self.mode == DataMode.MOCK:
            self._extractor = FeatureExtractor(
                sample_rate=CWRU_SAMPLE_RATE_HZ,
                run_freq_hz=CWRU_RUN_FREQ_HZ,
            )
        else:
            self._extractor = FeatureExtractor(
                sample_rate=self.pi_sample_rate,
                run_freq_hz=self.run_freq_hz,
            )

        self._is_ready = True
        self._split.summary()
        logger.info(
            "DataPipeline ready — mode=%s  sensor=%r",
            self.mode.name, self._sensor,
        )
        return self

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_sensor(self) -> VibrationSensor:
        """
        Return the active VibrationSensor.

        MOCK  →  MockSensor backed by CWRU healthy windows
        REAL  →  PhysicalSensor (connects to I2C on first call)
        """
        self._require_ready()
        return self._sensor

    def get_split(self) -> Split:
        """Return the dataset split (training / mock / eval groups)."""
        self._require_ready()
        return self._split

    def get_extractor(self) -> FeatureExtractor:
        """Return the FeatureExtractor configured for the current mode."""
        self._require_ready()
        return self._extractor

    def get_cleaner(self) -> SignalCleaner:
        """Return the SignalCleaner."""
        self._require_ready()
        return self._cleaner

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    def extract_features(
        self, windows: List[LabelledWindow]
    ) -> List[dict]:
        """
        Extract feature dicts from a list of LabelledWindows.

        Uses the extractor configured for CWRU rate when the windows come
        from CWRU data (they always do in MOCK mode), and the pi_sample_rate
        extractor in REAL mode (for live windows).

        Returns
        -------
        List of feature dicts (output of FeatureVector.to_dict())
        """
        self._require_ready()
        # Always use CWRU extractor for CWRU-sourced LabelledWindows
        cwru_extractor = FeatureExtractor(
            sample_rate=CWRU_SAMPLE_RATE_HZ,
            run_freq_hz=CWRU_RUN_FREQ_HZ,
        )
        return [cwru_extractor.transform(w.signal).to_dict() for w in windows]

    def extract_live_features(self, window: np.ndarray) -> dict:
        """
        Extract features from one raw window returned by the live sensor.

        Uses the extractor tuned for the current mode (CWRU rate for mock,
        pi sample_rate for real).
        """
        self._require_ready()
        return self._extractor.transform(window).to_dict()

    # ------------------------------------------------------------------
    # Convenience: get numpy arrays ready for sklearn/torch
    # ------------------------------------------------------------------

    def get_train_data(self):
        """
        Return (X_train, feature_dicts) for the healthy training windows.

        X_train : np.ndarray  shape (N, window_size) — raw signals
        feat_dicts : List[dict]                       — extracted features
        """
        self._require_ready()
        windows    = self._split.train_healthy
        X, _       = windows_to_arrays(windows)
        feat_dicts = self.extract_features(windows)
        return X, feat_dicts

    def get_eval_data(self):
        """
        Return (X_eval, feat_dicts, y_true) for the full evaluation set.

        y_true : List[int]  0 = healthy, 1 = faulty
        """
        self._require_ready()
        windows    = self._split.all_eval
        X, _       = windows_to_arrays(windows)
        feat_dicts = self.extract_features(windows)
        y_true     = self._split.eval_labels
        return X, feat_dicts, y_true

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_cwru(self) -> List[CWRURecord]:
        """Load CWRU .mat files, or return empty list if dir is missing."""
        if not self.cwru_dir.exists() or not any(self.cwru_dir.glob("*.mat")):
            logger.warning(
                "CWRU directory is empty or missing: %s\n"
                "  → Download from https://engineering.case.edu/bearingdatacenter\n"
                "  → Place .mat files in %s",
                self.cwru_dir, self.cwru_dir,
            )
            return []
        loader  = CWRULoader(self.cwru_dir)
        records = loader.load_all()
        return records

    def _build_sensor(self) -> VibrationSensor:
        """Instantiate the correct sensor for the current mode."""
        if self.mode == DataMode.MOCK:
            # Concatenate all mock_sensor windows into one long signal
            mock_windows = self._split.mock_sensor
            if not mock_windows:
                raise RuntimeError(
                    "No mock sensor windows available.  "
                    "Check CWRU directory and mock_fraction."
                )
            combined = np.concatenate([w.signal for w in mock_windows])
            logger.info(
                "MockSensor: %d windows → %.1f min of simulated healthy data",
                len(mock_windows), len(combined) / CWRU_SAMPLE_RATE_HZ / 60,
            )
            return MockSensor(
                data=combined,
                sample_rate=CWRU_SAMPLE_RATE_HZ,
                simulate_delay=False,
                label="cwru_healthy_mock",
            )
        else:
            logger.info(
                "PhysicalSensor: bus=%d addr=0x%02X rate=%d Hz",
                self.pi_bus, self.pi_address, self.pi_sample_rate,
            )
            return PhysicalSensor(
                bus_num=self.pi_bus,
                address=self.pi_address,
                sample_rate=self.pi_sample_rate,
            )

    def _require_ready(self) -> None:
        if not self._is_ready:
            raise RuntimeError(
                "DataPipeline has not been set up yet.  Call pipeline.setup() first."
            )

    def __repr__(self) -> str:
        status = "ready" if self._is_ready else "not set up"
        return f"DataPipeline(mode={self.mode.name}, status={status})"
