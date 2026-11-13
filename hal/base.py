"""
hal/base.py — Abstract base class for all vibration sensors.

Every concrete sensor (physical or mock) must implement `get_10s_window`.
This is the contract the rest of the pipeline depends on; nothing else in
the codebase should import sensor-specific code directly.
"""

from abc import ABC, abstractmethod
import numpy as np


class VibrationSensor(ABC):
    """
    Hardware Abstraction Layer base class.

    Subclasses represent either a physical I2C accelerometer or a mock
    data feeder backed by a dataset file.  The pipeline only ever calls
    `get_10s_window()` and never touches hardware registers directly.
    """

    # ------------------------------------------------------------------
    # Abstract interface — subclasses MUST implement this
    # ------------------------------------------------------------------

    @abstractmethod
    def get_10s_window(self) -> np.ndarray:
        """
        Collect (or simulate) 10 seconds of vibration data.

        Returns
        -------
        np.ndarray
            1-D array of acceleration samples (Z-axis, in raw ADC counts
            or g-units depending on the sensor).  Length = sample_rate * 10.
        """

    # ------------------------------------------------------------------
    # Shared helpers (available to all concrete sensors)
    # ------------------------------------------------------------------

    @property
    def window_seconds(self) -> int:
        """Fixed window length in seconds."""
        return 10

    def validate_window(self, window: np.ndarray) -> None:
        """
        Raise ValueError if the returned window looks wrong.
        Called optionally by concrete sensors before returning data.
        """
        if not isinstance(window, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(window)}")
        if window.ndim != 1:
            raise ValueError(f"Window must be 1-D, got shape {window.shape}")
        if len(window) == 0:
            raise ValueError("Window must not be empty")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
