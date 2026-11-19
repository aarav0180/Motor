"""
hal/physical_sensor.py — Live ADXL345 / MPU6050 sensor over I2C.

Requires `smbus2` (pip install smbus2) and runs only on a Raspberry Pi
(or any Linux board with an I2C bus).  On a laptop this module can still
be imported; the error surfaces only when PhysicalSensor() is instantiated.

ADXL345 register map (used here)
---------------------------------
0x2C  BW_RATE      — output data rate
0x2D  POWER_CTL    — measurement mode
0x31  DATA_FORMAT  — range / resolution
0x32  DATAX0       — X-axis LSB
0x33  DATAX1       — X-axis MSB
0x36  DATAZ0       — Z-axis LSB
0x37  DATAZ1       — Z-axis MSB
"""

import time
import logging
from typing import Tuple
import numpy as np

from .base import VibrationSensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ADXL345 register constants
# ---------------------------------------------------------------------------
_REG_BW_RATE    = 0x2C   # Output data rate register
_REG_POWER_CTL  = 0x2D   # Power control register
_REG_DATA_FORMAT= 0x31   # Data format register
_REG_DATAX0     = 0x32   # X-axis data LSB
_REG_DATAX1     = 0x33   # X-axis data MSB
_REG_DATAZ0     = 0x36   # Z-axis data LSB
_REG_DATAZ1     = 0x37   # Z-axis data MSB

_VAL_ODR_100HZ  = 0x0A   # BW_RATE value for 100 Hz output data rate
_VAL_ODR_400HZ  = 0x0C   # BW_RATE value for 400 Hz output data rate
_VAL_MEASURE    = 0x08   # POWER_CTL: set Measure bit
_VAL_FULL_RES_16G = 0x0B # DATA_FORMAT: full-resolution mode + 16g range (0x08 | 0x03)
_VAL_FULL_RES   = 0x08   # Legacy 2g full-resolution mode


class PhysicalSensor(VibrationSensor):
    """
    Live 3-axis accelerometer sensor using smbus2 over I2C.

    Supports ADXL345 with automatic 3-axis burst reading, gravity de-biasing,
    and dynamic magnitude vector calculation.

    Parameters
    ----------
    bus_num : int
        I2C bus number.  Raspberry Pi defaults to 1.
    address : int
        I2C address of the accelerometer (default 0x53).
    sample_rate : int
        Target samples per second.
    axis : str
        Measurement axis: 'magnitude' (3-axis dynamic vector), 'x', 'y', or 'z'.
    range_g : int
        Accelerometer measurement range: 2, 4, 8, or 16 (default 16g).
    """

    def __init__(
        self,
        bus_num: int = 1,
        address: int = 0x53,
        sample_rate: int = 1000,
        axis: str = "magnitude",
        range_g: int = 16,
    ) -> None:
        try:
            import smbus2
            self._bus = smbus2.SMBus(bus_num)
        except ImportError as exc:
            raise ImportError(
                "smbus2 is required for PhysicalSensor.  "
                "Install it with: pip install smbus2"
            ) from exc

        self.address = address
        self.sample_rate = sample_rate
        self.axis = axis.lower().strip()
        self.range_g = range_g

        self._init_adxl345()
        logger.info(
            "PhysicalSensor ready — bus=%d addr=0x%02X axis=%s range=%dg target_rate=%d Hz",
            bus_num, address, self.axis, self.range_g, sample_rate,
        )

    # ------------------------------------------------------------------
    # Sensor initialisation
    # ------------------------------------------------------------------

    def _init_adxl345(self) -> None:
        """Write initialisation bytes to the ADXL345."""
        # High-speed data rate (400 Hz or 100 Hz depending on target)
        odr_val = _VAL_ODR_400HZ if self.sample_rate >= 400 else _VAL_ODR_100HZ
        self._write_reg(_REG_BW_RATE, odr_val)
        self._write_reg(_REG_POWER_CTL, _VAL_MEASURE)

        # Range configuration (full resolution: 3.9 mg/LSB)
        range_bits = {2: 0x00, 4: 0x01, 8: 0x02, 16: 0x03}.get(self.range_g, 0x03)
        self._write_reg(_REG_DATA_FORMAT, 0x08 | range_bits)
        time.sleep(0.05)   # Let the sensor settle

    # ------------------------------------------------------------------
    # Low-level I2C helpers
    # ------------------------------------------------------------------

    def _write_reg(self, register: int, value: int) -> None:
        self._bus.write_byte_data(self.address, register, value)

    def _read_reg(self, register: int) -> int:
        return self._bus.read_byte_data(self.address, register)

    def _read_xyz_raw(self) -> Tuple[float, float, float]:
        """
        Read all 6 bytes (X, Y, Z) in a single burst I2C transaction.
        Returns acceleration in g units.
        """
        data = self._bus.read_i2c_block_data(self.address, _REG_DATAX0, 6)
        x_raw = np.int16((data[1] << 8) | data[0])
        y_raw = np.int16((data[3] << 8) | data[2])
        z_raw = np.int16((data[5] << 8) | data[4])
        # In Full-Res mode, scale factor is 3.9 mg/LSB = 0.00390625 g/count
        scale = 0.00390625
        return float(x_raw * scale), float(y_raw * scale), float(z_raw * scale)

    # ------------------------------------------------------------------
    # VibrationSensor interface
    # ------------------------------------------------------------------

    def get_10s_window(self) -> np.ndarray:
        """
        Poll the accelerometer for a window, remove DC gravity bias,
        and return the dynamic vibration waveform.
        """
        target_samples = self.sample_rate * self.window_seconds
        delay = 1.0 / self.sample_rate
        x_list, y_list, z_list = [], [], []

        t_start = time.perf_counter()
        for _ in range(target_samples):
            try:
                x, y, z = self._read_xyz_raw()
                x_list.append(x)
                y_list.append(y)
                z_list.append(z)
            except Exception:
                # Handle transient I2C read error gracefully
                if x_list:
                    x_list.append(x_list[-1])
                    y_list.append(y_list[-1])
                    z_list.append(z_list[-1])
            time.sleep(delay)
        elapsed = time.perf_counter() - t_start

        achieved_rate = len(x_list) / elapsed
        logger.debug(
            "Collected %d samples in %.2f s — achieved rate: %.1f Hz",
            len(x_list), elapsed, achieved_rate,
        )

        ax = np.array(x_list, dtype=np.float32)
        ay = np.array(y_list, dtype=np.float32)
        az = np.array(z_list, dtype=np.float32)

        # Remove static DC gravity vector (demean)
        dx = ax - float(np.mean(ax))
        dy = ay - float(np.mean(ay))
        dz = az - float(np.mean(az))

        if self.axis == "x":
            window = dx
        elif self.axis == "y":
            window = dy
        elif self.axis == "z":
            window = dz
        else:
            # Default: 3-axis dynamic vibration magnitude vector
            window = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2).astype(np.float32)

        self.validate_window(window)
        return window

    def close(self) -> None:
        """Release the I2C bus."""
        self._bus.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
