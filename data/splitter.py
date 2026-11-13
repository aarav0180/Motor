"""
data/splitter.py  —  Dataset splitting for training, mock-sensor feed, and evaluation.

One CWRU dataset is used for three purposes simultaneously:

  ┌────────────────────────────────────────────────────────────────┐
  │  CWRU records                                                  │
  │                                                                │
  │  Healthy windows ─────────────────────┬──────────────────────┐│
  │                                        │                      ││
  │                           MOCK_SENSOR  │    TRAIN (detector)  ││
  │                           (20 % of     │    (80 % of healthy) ││
  │                            healthy)    │                      ││
  │                                        └──────────────────────┘│
  │  Faulty windows  ───────────────────── EVAL (all fault types)  │
  │  + remaining healthy                   + held-out healthy      │
  └────────────────────────────────────────────────────────────────┘

The `DataSplitter` returns a `Split` namedtuple with four groups so the
rest of the codebase never touches raw splits directly.
"""

import logging
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .windows import LabelledWindow, windows_to_arrays

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class Split:
    """
    Four non-overlapping groups derived from the full CWRU dataset.

    Attributes
    ----------
    train_healthy : List[LabelledWindow]
        Healthy windows used to train the anomaly detectors.
    mock_sensor   : List[LabelledWindow]
        Healthy windows reserved for the MockSensor stream.
        When DATA_MODE = "mock", these are served one-by-one through the
        same VibrationSensor interface as the physical sensor.
    eval_healthy  : List[LabelledWindow]
        Held-out healthy windows for threshold calibration + FAR evaluation.
    eval_faulty   : List[LabelledWindow]
        All faulty windows for detection-rate evaluation.
    """
    train_healthy: List[LabelledWindow]
    mock_sensor:   List[LabelledWindow]
    eval_healthy:  List[LabelledWindow]
    eval_faulty:   List[LabelledWindow]

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def all_eval(self) -> List[LabelledWindow]:
        """Healthy + faulty eval windows (the full test set)."""
        return self.eval_healthy + self.eval_faulty

    @property
    def eval_labels(self) -> List[int]:
        """Binary labels for all_eval: 0 = healthy, 1 = faulty."""
        return [0 if w.is_healthy else 1 for w in self.all_eval]

    def summary(self) -> None:
        print(f"\n{'='*55}")
        print("  Dataset Split Summary")
        print(f"{'='*55}")
        print(f"  train_healthy : {len(self.train_healthy):>5} windows")
        print(f"  mock_sensor   : {len(self.mock_sensor):>5} windows "
              f"(≈ {len(self.mock_sensor) * 10 / 60:.0f} min of simulated data)")
        print(f"  eval_healthy  : {len(self.eval_healthy):>5} windows")
        print(f"  eval_faulty   : {len(self.eval_faulty):>5} windows")

        # Fault breakdown
        fault_counts: dict = {}
        for w in self.eval_faulty:
            fault_counts[w.label] = fault_counts.get(w.label, 0) + 1
        print(f"\n  Fault types in eval:")
        for ft, cnt in sorted(fault_counts.items()):
            print(f"    {ft:<20}: {cnt:>5} windows")
        print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Splitter class
# ---------------------------------------------------------------------------

class DataSplitter:
    """
    Split a list of LabelledWindows into the four groups described above.

    Parameters
    ----------
    mock_fraction : float
        Fraction of healthy windows reserved for the MockSensor stream.
        Default 0.20 (20 %).
    train_fraction : float
        Fraction of remaining healthy windows used for training.
        The rest go to eval_healthy for threshold calibration / FAR check.
        Default 0.80 (80 % of the non-mock healthy windows).
    seed : int
        Random seed for reproducible shuffling.
    """

    def __init__(
        self,
        mock_fraction:  float = 0.20,
        train_fraction: float = 0.80,
        seed:           int   = 42,
    ) -> None:
        if not (0.0 < mock_fraction < 1.0):
            raise ValueError("mock_fraction must be in (0, 1)")
        if not (0.0 < train_fraction < 1.0):
            raise ValueError("train_fraction must be in (0, 1)")

        self.mock_fraction  = mock_fraction
        self.train_fraction = train_fraction
        self.rng = random.Random(seed)

    def split(self, windows: List[LabelledWindow]) -> Split:
        """
        Partition all windows into the four groups.

        Parameters
        ----------
        windows : List[LabelledWindow]
            All windows from the full CWRU dataset (healthy + faulty).

        Returns
        -------
        Split
        """
        healthy = [w for w in windows if w.is_healthy]
        faulty  = [w for w in windows if not w.is_healthy]

        if not healthy:
            raise ValueError("No healthy windows found in the dataset.")
        if not faulty:
            logger.warning("No faulty windows found — eval_faulty will be empty.")

        # Shuffle for randomness
        self.rng.shuffle(healthy)

        # 1. Carve out mock-sensor portion
        n_mock  = max(1, int(len(healthy) * self.mock_fraction))
        mock    = healthy[:n_mock]
        rest    = healthy[n_mock:]

        # 2. Split remainder into train / eval_healthy
        n_train       = max(1, int(len(rest) * self.train_fraction))
        train_healthy = rest[:n_train]
        eval_healthy  = rest[n_train:]

        split = Split(
            train_healthy = train_healthy,
            mock_sensor   = mock,
            eval_healthy  = eval_healthy,
            eval_faulty   = faulty,
        )

        logger.info(
            "DataSplitter: total=%d  train_healthy=%d  mock=%d  "
            "eval_healthy=%d  eval_faulty=%d",
            len(windows),
            len(train_healthy), len(mock),
            len(eval_healthy),  len(faulty),
        )
        return split

    def split_records(
        self,
        records: list,          # List[CWRURecord]
        window_size: int,
        hop_size: Optional[int] = None,
        cleaner=None,           # Optional[SignalCleaner]
    ) -> Split:
        """
        Convenience wrapper: segment CWRURecords into windows then split.

        Parameters
        ----------
        records : List[CWRURecord]
        window_size : int
        hop_size : int, optional
        cleaner : SignalCleaner, optional
        """
        from .windows import windows_from_records
        all_windows = windows_from_records(records, window_size, hop_size, cleaner)
        return self.split(all_windows)
