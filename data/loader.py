"""
data/loader.py  —  CWRU Bearing Dataset loader.

The Case Western Reserve University (CWRU) dataset is the gold-standard
public benchmark for bearing fault detection.  Each .mat file contains
one or more time-series channels sampled at 12 kHz (or 48 kHz for some
files).  This module handles:

  - Discovering all .mat files in a directory
  - Parsing the filename convention to extract fault type, diameter, load
  - Loading the drive-end (DE) and/or fan-end (FE) channel
  - Returning typed CWRURecord dataclasses — no raw dicts leaking out

CWRU filename conventions
--------------------------
  Normal_0.mat         → healthy, load 0
  IR007_0.mat          → inner-race fault, 0.007-inch diameter, load 0
  OR007@6_0.mat        → outer-race fault, 0.007-inch, 6 o'clock, load 0
  B007_0.mat           → ball fault, 0.007-inch, load 0
  (loads: 0 = 0 HP, 1 = 1 HP, 2 = 2 HP, 3 = 3 HP)

Download from: https://engineering.case.edu/bearingdatacenter/download-data-file
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CWRU_SAMPLE_RATE_HZ = 12_000   # Drive-end channel standard rate
CWRU_RUN_FREQ_HZ    = 29.95    # ~1797 rpm ÷ 60

# Canonical fault labels used throughout the project
LABEL_HEALTHY     = "healthy"
LABEL_INNER_RACE  = "inner_race"
LABEL_OUTER_RACE  = "outer_race"
LABEL_BALL        = "ball"

# Map filename prefix → fault label
_FAULT_MAP = {
    "normal": LABEL_HEALTHY,
    "ir":     LABEL_INNER_RACE,
    "or":     LABEL_OUTER_RACE,
    "b":      LABEL_BALL,
}

# Standard Case Western Reserve University Bearing Data Center ID mapping
# Format: file_id -> (fault_type, diameter_thousandths, load_hp)
_CWRU_ID_MAP = {
    # Normal baseline (12k drive end)
    97:  (LABEL_HEALTHY, None, 0),
    98:  (LABEL_HEALTHY, None, 1),
    99:  (LABEL_HEALTHY, None, 2),
    100: (LABEL_HEALTHY, None, 3),
    # 0.007" Inner Race
    105: (LABEL_INNER_RACE, 7, 0),
    106: (LABEL_INNER_RACE, 7, 1),
    107: (LABEL_INNER_RACE, 7, 2),
    108: (LABEL_INNER_RACE, 7, 3),
    # 0.014" Inner Race
    169: (LABEL_INNER_RACE, 14, 0),
    170: (LABEL_INNER_RACE, 14, 1),
    171: (LABEL_INNER_RACE, 14, 2),
    172: (LABEL_INNER_RACE, 14, 3),
    # 0.021" Inner Race
    209: (LABEL_INNER_RACE, 21, 0),
    210: (LABEL_INNER_RACE, 21, 1),
    211: (LABEL_INNER_RACE, 21, 2),
    212: (LABEL_INNER_RACE, 21, 3),
    # 0.007" Ball
    118: (LABEL_BALL, 7, 0),
    119: (LABEL_BALL, 7, 1),
    120: (LABEL_BALL, 7, 2),
    121: (LABEL_BALL, 7, 3),
    # 0.014" Ball
    185: (LABEL_BALL, 14, 0),
    186: (LABEL_BALL, 14, 1),
    187: (LABEL_BALL, 14, 2),
    188: (LABEL_BALL, 14, 3),
    # 0.021" Ball
    222: (LABEL_BALL, 21, 0),
    223: (LABEL_BALL, 21, 1),
    224: (LABEL_BALL, 21, 2),
    225: (LABEL_BALL, 21, 3),
    # 0.007" Outer race @6:00
    130: (LABEL_OUTER_RACE, 7, 0),
    131: (LABEL_OUTER_RACE, 7, 1),
    132: (LABEL_OUTER_RACE, 7, 2),
    133: (LABEL_OUTER_RACE, 7, 3),
    # 0.014" Outer race @6:00
    197: (LABEL_OUTER_RACE, 14, 0),
    198: (LABEL_OUTER_RACE, 14, 1),
    199: (LABEL_OUTER_RACE, 14, 2),
    200: (LABEL_OUTER_RACE, 14, 3),
    # 0.021" Outer race @6:00
    234: (LABEL_OUTER_RACE, 21, 0),
    235: (LABEL_OUTER_RACE, 21, 1),
    236: (LABEL_OUTER_RACE, 21, 2),
    237: (LABEL_OUTER_RACE, 21, 3),
}

# Regex: captures (prefix, diameter_thou, load_hp)
# Handles: Normal_0, IR007_0, OR007@6_0, B021_3
_FILENAME_RE = re.compile(
    r"^(?P<prefix>[A-Za-z]+)"          # fault prefix (Normal, IR, OR, B)
    r"(?P<diam>\d{3})?"                # optional 3-digit diameter (thousandths)
    r"(?:@\d+)?"                       # optional position (@6)
    r"_(?P<load>\d)"                   # load suffix
    r"$",
    re.IGNORECASE,
)

# DE channel key pattern inside the .mat file
_DE_KEY_RE  = re.compile(r"X\d+_DE_time",  re.IGNORECASE)
_FE_KEY_RE  = re.compile(r"X\d+_FE_time",  re.IGNORECASE)
_RPM_KEY_RE = re.compile(r"X\d+_RPM",      re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------

@dataclass
class CWRURecord:
    """
    One loaded CWRU .mat file.

    Attributes
    ----------
    path : Path
        Source file path.
    fault_type : str
        One of 'healthy', 'inner_race', 'outer_race', 'ball'.
    fault_diameter_thou : Optional[int]
        Fault diameter in thousandths of an inch (None for healthy).
    load_hp : int
        Motor load in horsepower (0, 1, 2, 3).
    de_signal : np.ndarray
        Drive-end vibration channel (primary signal used for detection).
    fe_signal : Optional[np.ndarray]
        Fan-end channel (may be absent in some files).
    sample_rate : int
        Sample rate in Hz (almost always 12 000 Hz for DE channel).
    rpm : Optional[float]
        Measured RPM from the .mat file (if available).
    is_healthy : bool
        Convenience flag.
    """
    path:                Path
    fault_type:          str
    fault_diameter_thou: Optional[int]
    load_hp:             int
    de_signal:           np.ndarray
    fe_signal:           Optional[np.ndarray]
    sample_rate:         int
    rpm:                 Optional[float]

    @property
    def is_healthy(self) -> bool:
        return self.fault_type == LABEL_HEALTHY

    @property
    def label_str(self) -> str:
        """Human-readable label for plots / CSVs."""
        if self.is_healthy:
            return LABEL_HEALTHY
        diam = f"_{self.fault_diameter_thou}thou" if self.fault_diameter_thou else ""
        return f"{self.fault_type}{diam}_load{self.load_hp}"

    def __repr__(self) -> str:
        return (
            f"CWRURecord(fault={self.fault_type!r}, "
            f"diam={self.fault_diameter_thou}, "
            f"load={self.load_hp}, "
            f"samples={len(self.de_signal)})"
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class CWRULoader:
    """
    Discover and load all CWRU .mat files from a directory.

    Parameters
    ----------
    cwru_dir : str | Path
        Directory containing the downloaded .mat files.
    sample_rate : int
        Expected sample rate (used in logging; actual rate from file is preferred).
    load_fe : bool
        Whether to also load the fan-end channel (slower, uses more RAM).

    Example
    -------
    >>> loader = CWRULoader("data/cwru")
    >>> records = loader.load_all()
    >>> healthy = [r for r in records if r.is_healthy]
    """

    def __init__(
        self,
        cwru_dir: str | Path,
        sample_rate: int = CWRU_SAMPLE_RATE_HZ,
        load_fe: bool = False,
    ) -> None:
        self.cwru_dir    = Path(cwru_dir)
        self.sample_rate = sample_rate
        self.load_fe     = load_fe

        if not self.cwru_dir.exists():
            raise FileNotFoundError(f"CWRU directory not found: {self.cwru_dir}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover(self) -> List[Path]:
        """Return all .mat files sorted by name."""
        files = sorted(self.cwru_dir.glob("*.mat"))
        logger.info("Found %d .mat files in %s", len(files), self.cwru_dir)
        return files

    def load_all(self) -> List[CWRURecord]:
        """Load every .mat file and return a list of CWRURecords."""
        records = []
        for path in self.discover():
            try:
                rec = self._load_one(path)
                records.append(rec)
                logger.debug("Loaded %s", rec)
            except Exception as exc:
                logger.warning("Skipping %s: %s", path.name, exc)
        logger.info(
            "Loaded %d records  (healthy=%d, faulty=%d)",
            len(records),
            sum(r.is_healthy for r in records),
            sum(not r.is_healthy for r in records),
        )
        return records

    def load_healthy(self) -> List[CWRURecord]:
        """Load only files whose names start with 'Normal'."""
        return [r for r in self.load_all() if r.is_healthy]

    def load_by_fault(self, fault_type: str) -> List[CWRURecord]:
        """
        Load records for a specific fault type.

        Parameters
        ----------
        fault_type : str
            One of 'healthy', 'inner_race', 'outer_race', 'ball'.
        """
        return [r for r in self.load_all() if r.fault_type == fault_type]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_one(self, path: Path) -> CWRURecord:
        """Parse one .mat file into a CWRURecord."""
        import scipy.io

        mat = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)

        fault_type, diam, load = self._parse_filename(path.stem)

        de_signal = self._extract_channel(mat, _DE_KEY_RE)
        fe_signal = self._extract_channel(mat, _FE_KEY_RE) if self.load_fe else None
        rpm_val   = self._extract_scalar(mat, _RPM_KEY_RE)

        return CWRURecord(
            path=path,
            fault_type=fault_type,
            fault_diameter_thou=diam,
            load_hp=load,
            de_signal=de_signal.astype(np.float32),
            fe_signal=fe_signal.astype(np.float32) if fe_signal is not None else None,
            sample_rate=self.sample_rate,
            rpm=rpm_val,
        )

    @staticmethod
    def _parse_filename(stem: str) -> Tuple[str, Optional[int], int]:
        """
        Parse a CWRU filename stem into (fault_type, diameter_thou, load_hp).

        Supports:
          - Official numeric IDs: "97", "105", "118", "130"
          - Descriptive filenames: "Normal_0", "IR007_0", "OR007@6_0", "B007_0"
          - Fallback heuristics for custom test runs
        """
        # 1. Check if numeric CWRU ID (e.g. 97.mat, 105.mat)
        if stem.isdigit():
            cwru_id = int(stem)
            if cwru_id in _CWRU_ID_MAP:
                return _CWRU_ID_MAP[cwru_id]

        # 2. Check standard regex for descriptive names (Normal_0, IR007_0, etc.)
        m = _FILENAME_RE.match(stem)
        if not m:
            # Try heuristic fallback
            lower = stem.lower()
            if "normal" in lower:
                return LABEL_HEALTHY, None, 0
            for key, label in _FAULT_MAP.items():
                if lower.startswith(key):
                    return label, None, 0
            logger.warning("Cannot parse filename: %s — marking as 'unknown'", stem)
            return "unknown", None, 0

        prefix = m.group("prefix").lower()
        fault_type = _FAULT_MAP.get(prefix, "unknown")
        diam = int(m.group("diam")) if m.group("diam") else None
        load = int(m.group("load")) if m.group("load") else 0
        return fault_type, diam, load

    @staticmethod
    def _extract_channel(
        mat: Dict, pattern: re.Pattern
    ) -> Optional[np.ndarray]:
        """Find the first key matching `pattern` and return its data as 1-D array."""
        for key, val in mat.items():
            if pattern.match(key):
                arr = np.array(val).flatten()
                if arr.size > 0:
                    return arr
        return None

    @staticmethod
    def _extract_scalar(mat: Dict, pattern: re.Pattern) -> Optional[float]:
        for key, val in mat.items():
            if pattern.match(key):
                try:
                    return float(np.array(val).flatten()[0])
                except Exception:
                    pass
        return None

    def summary(self) -> None:
        """Print a human-readable table of all records."""
        records = self.load_all()
        print(f"\n{'='*60}")
        print(f"  CWRU Dataset Summary — {self.cwru_dir}")
        print(f"{'='*60}")
        print(f"  {'File':<25} {'Fault':<15} {'Diam':>5} {'Load':>5} {'Samples':>10}")
        print(f"  {'-'*55}")
        for r in records:
            print(
                f"  {r.path.name:<25} {r.fault_type:<15} "
                f"{str(r.fault_diameter_thou or '—'):>5} "
                f"{r.load_hp:>5} {len(r.de_signal):>10,}"
            )
        print(f"{'='*60}\n")
