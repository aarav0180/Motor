"""
Detectors package.
All detectors share the BaseDetector interface.
"""

from .base import BaseDetector
from .threshold import RMSThresholdDetector
from .isolation_forest import IsolationForestDetector
from .autoencoder import AutoencoderDetector
from .edge_models import (
    CWRUUnsupervisedDetector,
    CWRUAdvancedClassifierDetector,
    CWRUHybridDetector,
)

__all__ = [
    "BaseDetector",
    "RMSThresholdDetector",
    "IsolationForestDetector",
    "AutoencoderDetector",
    "CWRUUnsupervisedDetector",
    "CWRUAdvancedClassifierDetector",
    "CWRUHybridDetector",
]
