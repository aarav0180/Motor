"""
HAL (Hardware Abstraction Layer)
---------------------------------
Exports the sensor base class and both concrete implementations so the rest
of the codebase only ever imports from `hal`:

    from hal import PhysicalSensor, MockSensor
"""

from .base import VibrationSensor
from .physical_sensor import PhysicalSensor
from .mock_sensor import MockSensor

__all__ = ["VibrationSensor", "PhysicalSensor", "MockSensor"]
