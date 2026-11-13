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
_VAL_MEASURE    = 0x08   # POWER_CTL: set Measure bit
_VAL_FULL_RES   = 0x08   # DATA_FORMAT: full-resolution mode


class PhysicalSensor(VibrationSensor):
    """
    Live accelerometer sensor using smbus2 over I2C.

    Parameters
    ----------
    bus_num : int
        I2C bus number.  Raspberry Pi defaults to 1.
    address : int
        I2C address of the accelerometer.
        ADXL345: 0x53 (CS→VCC) or 0x1D (CS→3.3V).
        MPU6050: 0x68 (AD0→GND) or 0x69 (AD0→VCC).
    sample_rate : int
        Target samples per second for the polling loop.
        Note: actual achieved rate may be lower — always measure and log it.
    """

    def __init__(
        self,
        bus_num: int = 1,
        address: int = 0x53,
        sample_rate: int = 1000,
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

        self._init_adxl345()
        logger.info(
            "PhysicalSensor ready — bus=%d addr=0x%02X target_rate=%d Hz",
            bus_num, address, sample_rate,
        )

    # ------------------------------------------------------------------
    # Sensor initialisation
    # ------------------------------------------------------------------

    def _init_adxl345(self) -> None:
        """Write initialisation bytes to the ADXL345."""
        self._write_reg(_REG_BW_RATE,     _VAL_ODR_100HZ)  # 100 Hz ODR
        self._write_reg(_REG_POWER_CTL,   _VAL_MEASURE)    # Measurement mode
        self._write_reg(_REG_DATA_FORMAT, _VAL_FULL_RES)   # Full resolution
        time.sleep(0.1)   # Let the sensor settle

    # ------------------------------------------------------------------
    # Low-level I2C helpers
    # ------------------------------------------------------------------

    def _write_reg(self, register: int, value: int) -> None:
        self._bus.write_byte_data(self.address, register, value)

    def _read_reg(self, register: int) -> int:
        return self._bus.read_byte_data(self.address, register)

    def _read_raw_axis(self, lsb_reg: int, msb_reg: int) -> int:
        """
        Read a signed 10-bit value from two consecutive registers.

        Returns
        -------
        int
            Signed integer in the range [-512, 511].
        """
        lsb = self._read_reg(lsb_reg)
        msb = self._read_reg(msb_reg)
        raw = ((msb & 0x03) << 8) | lsb   # 10-bit unsigned
        if raw > 511:
            raw -= 1024                    # Two's complement → signed
        return raw

    # ------------------------------------------------------------------
    # VibrationSensor interface
    # ------------------------------------------------------------------

    def get_10s_window(self) -> np.ndarray:
        """
        Poll the ADXL345 Z-axis for 10 seconds and return all samples.

        Also logs the achieved sample rate so discrepancies from the
        target can be detected and documented.
        """
        target_samples = self.sample_rate * self.window_seconds
        delay = 1.0 / self.sample_rate
        samples = []

        t_start = time.perf_counter()
        for _ in range(target_samples):
            z = self._read_raw_axis(_REG_DATAZ0, _REG_DATAZ1)
            samples.append(z)
            time.sleep(delay)
        elapsed = time.perf_counter() - t_start

        achieved_rate = len(samples) / elapsed
        logger.debug(
            "Collected %d samples in %.2f s — achieved rate: %.1f Hz",
            len(samples), elapsed, achieved_rate,
        )

        window = np.array(samples, dtype=np.float32)
        self.validate_window(window)
        return window

    def close(self) -> None:
        """Release the I2C bus."""
        self._bus.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
