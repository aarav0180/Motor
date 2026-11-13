"""
data/windows.py  —  Windowing utilities shared across the project.

Converts long continuous signals into fixed-length windows (overlapping or
not) and assigns each window a label from its source CWRURecord.

Used by both the DataSplitter (offline) and the MockSensor (online).
"""

import logging
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LabelledWindow:
    """
    One fixed-length vibration window with its ground-truth label.

    Attributes
    ----------
    signal : np.ndarray  shape (window_size,)
    label  : str         e.g. 'healthy', 'inner_race', 'ball'
    source : str         filename or sensor identifier
    load_hp: int         motor load in HP (CWRU convention)
    fault_diameter_thou : Optional[int]
    """
    signal:              np.ndarray
    label:               str
    source:              str           = "unknown"
    load_hp:             int           = 0
    fault_diameter_thou: Optional[int] = None

    @property
    def is_healthy(self) -> bool:
        return self.label == "healthy"

    def __repr__(self) -> str:
        return (
            f"LabelledWindow(label={self.label!r}, "
            f"len={len(self.signal)}, source={self.source!r})"
        )


# ---------------------------------------------------------------------------
# Core windowing functions
# ---------------------------------------------------------------------------

def segment_signal(
    signal: np.ndarray,
    window_size: int,
    hop_size: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Slice a 1-D signal into fixed-length windows.

    Parameters
    ----------
    signal : np.ndarray  1-D array
    window_size : int    samples per window
    hop_size : int       step between windows; equals window_size if None
                         (non-overlapping).  Use window_size // 2 for 50 % overlap.

    Returns
    -------
    List of np.ndarray, each of length window_size.
    Tail samples that don't fill a full window are dropped.
    """
    if hop_size is None:
        hop_size = window_size

    n = len(signal)
    windows = []
    start = 0
    while start + window_size <= n:
        windows.append(signal[start : start + window_size].astype(np.float32))
        start += hop_size

    logger.debug(
        "segment_signal: %d samples → %d windows "
        "(window=%d, hop=%d, dropped=%d)",
        n, len(windows), window_size, hop_size,
        n - (start - hop_size + window_size) if windows else n,
    )
    return windows


def windows_from_record(
    record,                   # CWRURecord — avoid circular import with string hint
    window_size: int,
    hop_size: Optional[int] = None,
    cleaner=None,             # Optional[SignalCleaner]
) -> List[LabelledWindow]:
    """
    Segment a CWRURecord's DE signal into LabelledWindows.

    Parameters
    ----------
    record : CWRURecord
    window_size : int
    hop_size : int, optional
    cleaner : SignalCleaner, optional
        If provided, cleaning is applied before segmentation.
    """
    signal = record.de_signal.copy()
    if cleaner is not None:
        signal = cleaner.clean(signal)

    segments = segment_signal(signal, window_size, hop_size)
    labelled = [
        LabelledWindow(
            signal=seg,
            label=record.fault_type,
            source=record.path.name,
            load_hp=record.load_hp,
            fault_diameter_thou=record.fault_diameter_thou,
        )
        for seg in segments
    ]
    logger.debug(
        "windows_from_record: %s → %d windows (label=%s)",
        record.path.name, len(labelled), record.fault_type,
    )
    return labelled


def windows_from_records(
    records: list,
    window_size: int,
    hop_size: Optional[int] = None,
    cleaner=None,
) -> List[LabelledWindow]:
    """Convenience wrapper: flatten windows from multiple records."""
    all_windows = []
    for rec in records:
        all_windows.extend(
            windows_from_record(rec, window_size, hop_size, cleaner)
        )
    return all_windows


def windows_to_arrays(
    windows: List[LabelledWindow],
) -> Tuple[np.ndarray, List[str]]:
    """
    Stack windows into a 2-D matrix and return (X, labels).

    Returns
    -------
    X : np.ndarray  shape (n_windows, window_size)
    labels : list of str
    """
    X      = np.stack([w.signal for w in windows])   # (N, window_size)
    labels = [w.label for w in windows]
    return X, labels
