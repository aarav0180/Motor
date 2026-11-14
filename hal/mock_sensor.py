"""
hal/mock_sensor.py — Dataset-backed sensor for local development & CI.

Feeds pre-recorded or public-dataset data through the same interface as
the physical sensor, allowing the entire pipeline to run offline on a
laptop without any hardware.

Supports:
  - Any numpy array (loaded from .npy or .mat files)
  - Optional simulated real-time delay (default: disabled for fast testing)
  - Automatic looping when the dataset is exhausted
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from .base import VibrationSensor

logger = logging.getLogger(__name__)


class MockSensor(VibrationSensor):
    """
    Replay a pre-recorded vibration signal through the sensor interface.

    Supports:
      - Any numpy array (loaded from .npy or .mat files)
      - Dynamic mode: automatically alternates between healthy baseline operation
        and realistic bearing fault shocks (inner race, outer race, ball defect)
      - Stochastic physical noise injection to ensure continuous realistic variation
      - Optional simulated real-time delay (default: disabled for fast testing)
      - Automatic looping when the dataset is exhausted
    """

    def __init__(
        self,
        data: np.ndarray,
        sample_rate: int = 12_000,
        simulate_delay: bool = False,
        label: str = "unknown",
        fault_data: Optional[Dict[str, np.ndarray]] = None,
        dynamic_mode: bool = False,
        add_noise: bool = False,
        fault_every_n: int = 5,
    ) -> None:
        if data.ndim != 1:
            raise ValueError(f"data must be 1-D, got shape {data.shape}")

        self.data = data.astype(np.float32)
        self.sample_rate = sample_rate
        self.simulate_delay = simulate_delay
        self.label = label
        self.fault_data = fault_data or {}
        self.dynamic_mode = dynamic_mode
        self.add_noise = add_noise
        self.fault_every_n = fault_every_n
        self.current_label = label

        self._cursor = 0           # current position in the data array
        self._window_count = 0     # how many windows have been served

        logger.info(
            "MockSensor ready — label=%r  total_samples=%d  rate=%d Hz  dynamic=%s",
            label, len(data), sample_rate, dynamic_mode,
        )

    # ------------------------------------------------------------------
    # VibrationSensor interface
    # ------------------------------------------------------------------

    def get_10s_window(self) -> np.ndarray:
        """
        Return the next 10-second slice of the dataset.

        In dynamic mode, periodically introduces bearing fault bursts from the fault
        dataset or synthetic shocks to exercise real-time anomaly detection and alerts.
        """
        window_size = self.sample_rate * self.window_seconds

        inject_fault = False
        current_fault_name = None

        if self.dynamic_mode and self.fault_data:
            # Inject fault every `fault_every_n` windows (e.g., window 4, 9, 14...)
            if (self._window_count + 1) % self.fault_every_n == 0:
                inject_fault = True
                fault_keys = list(self.fault_data.keys())
                current_fault_name = fault_keys[(self._window_count // self.fault_every_n) % len(fault_keys)]

        if inject_fault and current_fault_name and current_fault_name in self.fault_data:
            fdata = self.fault_data[current_fault_name]
            cursor = (self._window_count * window_size) % max(1, len(fdata) - window_size)
            window = fdata[cursor : cursor + window_size].copy()
            self.current_label = current_fault_name
            logger.info("MockSensor: served FAULT window [%s] (#%d)", current_fault_name, self._window_count + 1)
        else:
            if self._cursor + window_size > len(self.data):
                logger.debug(
                    "MockSensor reached end of data — looping back to start (window #%d)",
                    self._window_count,
                )
                self._cursor = 0

            window = self.data[self._cursor : self._cursor + window_size].copy()
            self._cursor += window_size
            self.current_label = self.label

        self._window_count += 1

        if self.add_noise:
            # Inject subtle physical sensor noise (jitter & minor drift)
            noise_sigma = 0.02 * max(float(np.std(window)), 0.05)
            noise = np.random.normal(0.0, noise_sigma, size=window.shape).astype(np.float32)
            window = window + noise

        if self.simulate_delay:
            time.sleep(self.window_seconds)

        self.validate_window(window)
        return window

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Rewind the cursor to the beginning of the dataset."""
        self._cursor = 0
        self._window_count = 0

    @property
    def windows_served(self) -> int:
        """Total number of windows returned since creation (or last reset)."""
        return self._window_count

    @classmethod
    def from_npy(cls, path: str, **kwargs) -> "MockSensor":
        """Load a .npy file and wrap it in a MockSensor."""
        data = np.load(path)
        logger.info("Loaded .npy file: %s  shape=%s", path, data.shape)
        return cls(data.flatten(), **kwargs)

    @classmethod
    def from_mat(cls, path: str, key: str, **kwargs) -> "MockSensor":
        """
        Load a CWRU-style .mat file.

        Parameters
        ----------
        path : str
            Path to the .mat file.
        key : str
            Variable name inside the file, e.g. 'X097_DE_time'.
        """
        import scipy.io
        mat = scipy.io.loadmat(path)
        if key not in mat:
            available = [k for k in mat if not k.startswith("_")]
            raise KeyError(
                f"Key '{key}' not found in {path}.  "
                f"Available keys: {available}"
            )
        data = mat[key].flatten()
        logger.info(
            "Loaded .mat file: %s  key=%s  samples=%d", path, key, len(data)
        )
        return cls(data, **kwargs)


def create_dynamic_mock_sensor(
    cwru_dir: str = "data/cwru",
    sample_rate: int = 12_000,
    simulate_delay: bool = False,
    fault_every_n: int = 4,
) -> MockSensor:
    """
    Factory creating a realistic MockSensor using available CWRU bearing dataset files
    or synthetic mechanical vibration signals.

    Features:
      - Realistic baseline operation (normal motor humming, low RMS)
      - Dynamic fault injection bursts (inner race, outer race, ball defect)
      - Stochastic sensor noise
    """
    cwru_path = Path(cwru_dir)
    healthy_signal: Optional[np.ndarray] = None
    fault_signals: Dict[str, np.ndarray] = {}

    if cwru_path.exists():
        try:
            import scipy.io

            def _extract_de(filepath: Path) -> Optional[np.ndarray]:
                try:
                    mat = scipy.io.loadmat(str(filepath), squeeze_me=True, struct_as_record=False)
                    for k in mat:
                        if "DE_time" in k or k.startswith("X") and "DE" in k:
                            arr = np.asarray(mat[k]).flatten().astype(np.float32)
                            if len(arr) >= sample_rate * 10:
                                return arr
                except Exception:
                    pass
                return None

            # Look for normal baseline
            for fname in ("Normal_0.mat", "Normal_1.mat", "97.mat"):
                p = cwru_path / fname
                if p.exists():
                    arr = _extract_de(p)
                    if arr is not None:
                        healthy_signal = arr
                        break

            # Look for fault files
            fault_map = {
                "inner_race_fault": ["IR007_0.mat", "IR014_1.mat", "105.mat"],
                "outer_race_fault": ["OR007@6_0.mat", "OR014@6_1.mat", "130.mat"],
                "ball_fault": ["B007_0.mat", "B014_1.mat", "118.mat"],
            }
            for fault_name, fnames in fault_map.items():
                for fn in fnames:
                    p = cwru_path / fn
                    if p.exists():
                        arr = _extract_de(p)
                        if arr is not None:
                            fault_signals[fault_name] = arr
                            break
        except Exception as exc:
            logger.warning("Could not load CWRU .mat files (%s); falling back to synthetic generator.", exc)

    # If no CWRU files found, synthesize realistic physical signals
    if healthy_signal is None:
        fs = sample_rate
        duration_s = 60
        t = np.linspace(0, duration_s, fs * duration_s, endpoint=False)
        # Healthy: low RMS, dominant 40 Hz rotational frequency + 2nd harmonic
        healthy_signal = (
            0.12 * np.sin(2 * np.pi * 40.0 * t)
            + 0.03 * np.sin(2 * np.pi * 80.0 * t)
            + np.random.normal(0, 0.04, len(t))
        ).astype(np.float32)

    if not fault_signals:
        fs = sample_rate
        duration_s = 60
        t = np.linspace(0, duration_s, fs * duration_s, endpoint=False)
        # Inner race fault: periodic impact shocks (BPFI ~156 Hz) with ringing
        impact_train = np.zeros_like(t)
        period_idx = int(fs / 156.0)
        decay_len = min(int(fs * 0.005), period_idx)
        decay_curve = np.exp(-np.linspace(0, 5, decay_len)) * np.sin(2 * np.pi * 3200 * np.linspace(0, 0.005, decay_len))
        for idx in range(0, len(t) - decay_len, period_idx):
            impact_train[idx : idx + decay_len] += decay_curve * 0.9

        inner_fault = (
            healthy_signal[: len(t)]
            + impact_train
            + np.random.normal(0, 0.08, len(t))
        ).astype(np.float32)
        fault_signals["inner_race_fault"] = inner_fault

    return MockSensor(
        data=healthy_signal,
        sample_rate=sample_rate,
        simulate_delay=simulate_delay,
        label="healthy",
        fault_data=fault_signals,
        dynamic_mode=True,
        add_noise=True,
        fault_every_n=fault_every_n,
    )
