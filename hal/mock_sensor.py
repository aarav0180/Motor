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
import numpy as np

from .base import VibrationSensor

logger = logging.getLogger(__name__)


class MockSensor(VibrationSensor):
    """
    Replay a pre-recorded vibration signal through the sensor interface.

    Parameters
    ----------
    data : np.ndarray
        1-D array of vibration samples (e.g. CWRU drive-end channel).
    sample_rate : int
        Sample rate of the source data in Hz (CWRU DE channel = 12 000 Hz).
    simulate_delay : bool
        If True, sleep for `window_seconds` to mimic real acquisition time.
        Set False for fast unit-testing or batch evaluation.
    label : str
        Human-readable label (e.g. "healthy", "inner_race_fault") used in
        logging and optional metadata returns.
    """

    def __init__(
        self,
        data: np.ndarray,
        sample_rate: int = 12_000,
        simulate_delay: bool = False,
        label: str = "unknown",
    ) -> None:
        if data.ndim != 1:
            raise ValueError(f"data must be 1-D, got shape {data.shape}")

        self.data = data.astype(np.float32)
        self.sample_rate = sample_rate
        self.simulate_delay = simulate_delay
        self.label = label
        self._cursor = 0           # current position in the data array
        self._window_count = 0     # how many windows have been served

        logger.info(
            "MockSensor ready — label=%r  total_samples=%d  rate=%d Hz  "
            "approx %.1f windows available",
            label, len(data), sample_rate,
            len(data) / (sample_rate * self.window_seconds),
        )

    # ------------------------------------------------------------------
    # VibrationSensor interface
    # ------------------------------------------------------------------

    def get_10s_window(self) -> np.ndarray:
        """
        Return the next 10-second slice of the dataset.

        Loops back to the start if less than a full window remains.
        """
        window_size = self.sample_rate * self.window_seconds

        # Loop back if we have run out of data
        if self._cursor + window_size > len(self.data):
            logger.debug(
                "MockSensor reached end of data — looping back to start "
                "(window #%d)", self._window_count,
            )
            self._cursor = 0

        window = self.data[self._cursor : self._cursor + window_size]
        self._cursor += window_size
        self._window_count += 1

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
