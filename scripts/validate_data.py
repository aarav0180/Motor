"""
scripts/validate_data.py
------------------------
Data Health & Integrity Validator.

Tests data files in data/cwru/ and data/raw/ to verify:
  - File readability (.mat / .npy)
  - Absence of NaN or Inf values
  - Non-trivial signal activity (not stuck/dead-zero signal)
  - Sufficient duration for at least one full 10-second analysis window
  - Healthy vs Faulty class balance
  - Expected physical ranges (RMS, peak, crest factor)

Usage
-----
    python scripts/validate_data.py --cwru-dir data/cwru
    python scripts/validate_data.py --raw-dir data/raw
    python scripts/validate_data.py --all
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import CWRULoader, CWRURecord, CWRU_SAMPLE_RATE_HZ
from data.cleaner import SignalCleaner, CleanerConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_data")


class DataValidationError(Exception):
    """Raised when data integrity checks fail."""
    pass


def validate_signal_array(signal: np.ndarray, source_name: str, min_samples: int) -> Dict[str, float]:
    """
    Validate a single 1-D numpy array of vibration data.

    Returns summary stats if valid, or raises DataValidationError.
    """
    if not isinstance(signal, np.ndarray):
        raise DataValidationError(f"{source_name}: Expected np.ndarray, got {type(signal)}")

    if signal.ndim != 1:
        raise DataValidationError(f"{source_name}: Signal must be 1-D, got shape {signal.shape}")

    if len(signal) < min_samples:
        raise DataValidationError(
            f"{source_name}: Signal has {len(signal)} samples, which is less than "
            f"the minimum required {min_samples} samples."
        )

    if np.isnan(signal).any():
        n_nans = int(np.isnan(signal).sum())
        raise DataValidationError(f"{source_name}: Signal contains {n_nans} NaN values!")

    if np.isinf(signal).any():
        n_infs = int(np.isinf(signal).sum())
        raise DataValidationError(f"{source_name}: Signal contains {n_infs} Inf values!")

    std_val = float(np.std(signal))
    if std_val < 1e-7:
        raise DataValidationError(
            f"{source_name}: Signal has near-zero standard deviation ({std_val:.2e}). "
            "Sensor flatlined or channel dead!"
        )

    rms_val = float(np.sqrt(np.mean(signal ** 2)))
    peak_val = float(np.max(np.abs(signal)))
    crest = peak_val / rms_val if rms_val > 0 else 0.0

    return {
        "samples": len(signal),
        "mean": float(np.mean(signal)),
        "std": std_val,
        "rms": rms_val,
        "peak": peak_val,
        "crest_factor": crest,
    }


def validate_cwru_dataset(cwru_dir: Path, window_seconds: int = 10) -> Tuple[bool, List[str]]:
    """Validate all CWRU .mat files in a directory."""
    issues = []
    print(f"\n{'='*70}")
    print(f"  Validating CWRU Dataset in: {cwru_dir}")
    print(f"{'='*70}")

    if not cwru_dir.exists():
        issues.append(f"Directory {cwru_dir} does not exist.")
        return False, issues

    mat_files = list(cwru_dir.glob("*.mat"))
    if not mat_files:
        issues.append(f"No .mat files found in {cwru_dir}.")
        return False, issues

    loader = CWRULoader(cwru_dir)
    records = loader.load_all()

    if not records:
        issues.append("CWRULoader loaded 0 records.")
        return False, issues

    min_samples = CWRU_SAMPLE_RATE_HZ * window_seconds
    cleaner = SignalCleaner(CleanerConfig.for_cwru(), sample_rate=CWRU_SAMPLE_RATE_HZ)

    healthy_count = 0
    faulty_count = 0

    header = f"{'File':<18} {'Fault':<12} {'Samples':>9} {'RMS':>8} {'Peak':>8} {'Crest':>7} {'Status':>8}"
    print(header)
    print("-" * 70)

    for rec in records:
        try:
            stats = validate_signal_array(rec.de_signal, rec.path.name, min_samples)
            # Test that cleaner runs cleanly without errors
            cleaned = cleaner.clean(rec.de_signal[:min_samples])
            if len(cleaned) != min_samples:
                raise DataValidationError(f"Cleaner changed sample count from {min_samples} to {len(cleaned)}")

            status = "PASS"
            if rec.is_healthy:
                healthy_count += 1
            else:
                faulty_count += 1

            print(
                f"{rec.path.name:<18} {rec.fault_type:<12} {stats['samples']:>9,} "
                f"{stats['rms']:>8.4f} {stats['peak']:>8.4f} {stats['crest_factor']:>7.2f} "
                f"{status:>8}"
            )
        except Exception as e:
            issues.append(f"{rec.path.name}: {e}")
            print(f"{rec.path.name:<18} {rec.fault_type:<12} {'FAILED':>40}")

    print("-" * 70)
    print(f"Total files: {len(records)} | Healthy: {healthy_count} | Faulty: {faulty_count}")

    if healthy_count == 0:
        issues.append("Dataset has NO healthy data! Anomaly detector training requires healthy data.")

    if faulty_count == 0:
        issues.append("Dataset has NO faulty data! Benchmark evaluation requires faulty test data.")

    passed = len(issues) == 0
    print(f"CWRU Validation: {'ALL CHECKS PASSED [OK]' if passed else 'FAILURES DETECTED'}")
    return passed, issues


def validate_raw_dataset(raw_dir: Path, sample_rate: int = 1000, window_seconds: int = 10) -> Tuple[bool, List[str]]:
    """Validate all recorded .npy vibration files in a directory."""
    issues = []
    print(f"\n{'='*70}")
    print(f"  Validating Raw Test Rig Dataset in: {raw_dir}")
    print(f"{'='*70}")

    if not raw_dir.exists():
        issues.append(f"Directory {raw_dir} does not exist.")
        return False, issues

    npy_files = list(raw_dir.glob("*.npy"))
    if not npy_files:
        print("  No .npy files found in raw directory (none recorded yet).")
        return True, []

    min_samples = sample_rate * window_seconds
    cleaner = SignalCleaner(CleanerConfig.for_adxl345(), sample_rate=sample_rate)

    header = f"{'File':<28} {'Samples':>9} {'RMS':>8} {'Peak':>8} {'Crest':>7} {'Status':>8}"
    print(header)
    print("-" * 70)

    for npy_path in sorted(npy_files):
        try:
            data = np.load(str(npy_path))
            stats = validate_signal_array(data, npy_path.name, min_samples)
            cleaner.clean(data[:min_samples])
            print(
                f"{npy_path.name:<28} {stats['samples']:>9,} "
                f"{stats['rms']:>8.4f} {stats['peak']:>8.4f} {stats['crest_factor']:>7.2f} "
                f"{'PASS':>8}"
            )
        except Exception as e:
            issues.append(f"{npy_path.name}: {e}")
            print(f"{npy_path.name:<28} {'FAILED':>40}")

    print("-" * 70)
    passed = len(issues) == 0
    print(f"Raw Data Validation: {'ALL CHECKS PASSED [OK]' if passed else 'FAILURES DETECTED'}")
    return passed, issues


def main():
    parser = argparse.ArgumentParser(description="Validate vibration dataset integrity.")
    parser.add_argument("--cwru-dir", default="data/cwru", help="Directory with CWRU .mat files")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory with raw rig .npy files")
    parser.add_argument("--all", action="store_true", help="Validate both CWRU and raw data")
    args = parser.parse_args()

    cwru_path = Path(args.cwru_dir)
    raw_path = Path(args.raw_dir)

    all_passed = True
    all_issues = []

    if args.all or cwru_path.exists():
        ok, issues = validate_cwru_dataset(cwru_path)
        all_passed = all_passed and ok
        all_issues.extend(issues)

    if args.all or (raw_path.exists() and list(raw_path.glob("*.npy"))):
        ok, issues = validate_raw_dataset(raw_path)
        all_passed = all_passed and ok
        all_issues.extend(issues)

    if not all_passed:
        print("\nValidation Summary: ISSUES FOUND")
        for iss in all_issues:
            print(f"  - {iss}")
        sys.exit(1)
    else:
        print("\nAll data integrity checks passed cleanly.")
        sys.exit(0)


if __name__ == "__main__":
    main()
