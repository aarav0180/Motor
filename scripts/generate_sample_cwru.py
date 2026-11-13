"""
scripts/generate_sample_cwru.py
-------------------------------
Generates synthetic CWRU-format .mat files for offline development and testing.

Simulates the physics of:
  - Normal (healthy) motor: running speed (29.95 Hz) + harmonics + low background noise
  - Inner race fault (IR): sharp periodic shock pulses at BPFI (Ball Pass Frequency Inner race)
  - Outer race fault (OR): periodic impacts at BPFO (Ball Pass Frequency Outer race)
  - Ball fault (B): impacts modulated at FTF (Fundamental Train Frequency)

Outputs standard CWRU .mat files to data/cwru/ (or specified directory)
matching the exact naming and internal variable schema:
  - Normal_0.mat  (variable: X097_DE_time, X097_RPM)
  - IR007_0.mat   (variable: X105_DE_time, X105_RPM)
  - OR007@6_0.mat (variable: X130_DE_time, X130_RPM)
  - B007_0.mat    (variable: X118_DE_time, X118_RPM)

Usage
-----
    python scripts/generate_sample_cwru.py
    python scripts/generate_sample_cwru.py --duration 30 --output-dir data/cwru
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import scipy.io

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_cwru")

SAMPLE_RATE_HZ = 12_000   # 12 kHz standard for CWRU drive-end channel
RUN_FREQ_HZ    = 29.95    # ~1797 RPM / 60

# Characteristic fault frequency multipliers for standard 6205-2RS bearing
BPFI_MULT = 5.415  # Inner race
BPFO_MULT = 3.585  # Outer race
BSF_MULT  = 2.357  # Ball spin frequency


def generate_healthy_signal(duration_s: float, sample_rate: int = SAMPLE_RATE_HZ) -> np.ndarray:
    """Generate healthy vibration signal with 1X, 2X, 3X running speed harmonics + Gaussian noise."""
    n_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n_samples, endpoint=False, dtype=np.float32)

    # Fundamental + harmonics
    sig = (
        0.30 * np.sin(2 * np.pi * RUN_FREQ_HZ * t)
        + 0.15 * np.sin(2 * np.pi * 2 * RUN_FREQ_HZ * t)
        + 0.08 * np.sin(2 * np.pi * 3 * RUN_FREQ_HZ * t)
    )
    # Background Gaussian noise
    noise = np.random.normal(0, 0.08, n_samples).astype(np.float32)
    return (sig + noise).astype(np.float32)


def generate_fault_signal(
    fault_type: str,
    duration_s: float,
    sample_rate: int = SAMPLE_RATE_HZ,
    severity: float = 1.0,
) -> np.ndarray:
    """
    Generate faulty vibration signal with periodic high-frequency impulse decays
    superimposed on the baseline healthy motor signal.
    """
    base_sig = generate_healthy_signal(duration_s, sample_rate)
    n_samples = len(base_sig)

    if fault_type == "inner_race":
        fault_freq = RUN_FREQ_HZ * BPFI_MULT
        impact_amp = 1.8 * severity
    elif fault_type == "outer_race":
        fault_freq = RUN_FREQ_HZ * BPFO_MULT
        impact_amp = 1.4 * severity
    elif fault_type == "ball":
        fault_freq = RUN_FREQ_HZ * BSF_MULT
        impact_amp = 1.2 * severity
    else:
        return base_sig

    # Generate impulse train with exponential ring-down at resonance (approx 3000 Hz)
    impact_interval = int(sample_rate / fault_freq)
    decay_samples = int(sample_rate * 0.005)  # 5 ms impact decay
    decay_t = np.linspace(0, 0.005, decay_samples, endpoint=False, dtype=np.float32)
    impact_wave = np.exp(-decay_t * 800) * np.sin(2 * np.pi * 3200 * decay_t)

    impulse_signal = np.zeros(n_samples, dtype=np.float32)
    for idx in range(0, n_samples - decay_samples, impact_interval):
        # Add slight jitter to simulate mechanical variation
        jitter = np.random.randint(-2, 3)
        pos = max(0, min(n_samples - decay_samples, idx + jitter))
        impulse_signal[pos : pos + decay_samples] += impact_amp * impact_wave

    return (base_sig + impulse_signal).astype(np.float32)


def create_cwru_dataset(output_dir: Path, duration_s: float = 25.0) -> None:
    """Create a suite of standard CWRU .mat files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    file_specs = [
        {"filename": "Normal_0.mat",  "prefix": "X097", "fault": "healthy",    "rpm": 1797.0},
        {"filename": "Normal_1.mat",  "prefix": "X098", "fault": "healthy",    "rpm": 1772.0},
        {"filename": "IR007_0.mat",   "prefix": "X105", "fault": "inner_race", "rpm": 1797.0},
        {"filename": "IR014_1.mat",   "prefix": "X119", "fault": "inner_race", "rpm": 1772.0},
        {"filename": "OR007@6_0.mat", "prefix": "X130", "fault": "outer_race", "rpm": 1797.0},
        {"filename": "OR014@6_1.mat", "prefix": "X144", "fault": "outer_race", "rpm": 1772.0},
        {"filename": "B007_0.mat",    "prefix": "X118", "fault": "ball",       "rpm": 1797.0},
        {"filename": "B014_1.mat",    "prefix": "X122", "fault": "ball",       "rpm": 1772.0},
    ]

    for spec in file_specs:
        file_path = output_dir / spec["filename"]
        prefix = spec["prefix"]

        if spec["fault"] == "healthy":
            de_data = generate_healthy_signal(duration_s)
        else:
            de_data = generate_fault_signal(spec["fault"], duration_s)

        # Build standard CWRU dictionary structure
        mat_dict = {
            f"{prefix}_DE_time": de_data.reshape(-1, 1),
            f"{prefix}_RPM": np.array([[spec["rpm"]]], dtype=np.float64),
        }

        scipy.io.savemat(str(file_path), mat_dict)
        logger.info(
            "Created %s — fault: %-10s samples: %d (%.1f s)",
            spec["filename"], spec["fault"], len(de_data), duration_s,
        )


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic CWRU bearing dataset files.")
    parser.add_argument("--output-dir", default="data/cwru", help="Directory where .mat files will be saved")
    parser.add_argument("--duration", type=float, default=25.0, help="Duration of signal per file in seconds")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    logger.info("Generating CWRU dataset in: %s", out_path.resolve())
    create_cwru_dataset(out_path, duration_s=args.duration)
    logger.info("Done! CWRU sample files successfully generated.")


if __name__ == "__main__":
    main()
