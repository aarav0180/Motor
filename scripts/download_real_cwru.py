"""
scripts/download_real_cwru.py
-----------------------------
Downloads official experimental benchmark datasets directly from the
Case Western Reserve University (CWRU) Bearing Data Center repository:
  https://engineering.case.edu/bearingdatacenter

Default set downloads 4 core benchmark conditions (0 HP load, 1797 RPM, 12 kHz):
  - 97.mat  : Normal Baseline (Healthy)
  - 105.mat : Inner Race Fault (0.007" diameter)
  - 118.mat : Ball Fault (0.007" diameter)
  - 130.mat : Outer Race Fault (0.007" diameter, 6 o'clock location)

Optionally downloads full 16-file benchmark suite across multiple motor loads (0-3 HP).

Usage
-----
    python scripts/download_real_cwru.py
    python scripts/download_real_cwru.py --full
    python scripts/download_real_cwru.py --output-dir data/cwru
"""

import argparse
import logging
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_cwru")

BASE_URL = "https://engineering.case.edu/sites/default/files"

# Core 4 benchmark files: 0 HP load, 1797 RPM
CORE_FILES = {
    "97.mat":  "Normal Baseline (Healthy)",
    "105.mat": "Inner Race Fault (0.007 in, 0 HP)",
    "118.mat": "Ball Fault (0.007 in, 0 HP)",
    "130.mat": "Outer Race Fault @6:00 (0.007 in, 0 HP)",
}

# Extended suite: multiple loads (0 HP to 3 HP)
EXTENDED_FILES = {
    **CORE_FILES,
    "98.mat":  "Normal Baseline (1 HP, 1772 RPM)",
    "99.mat":  "Normal Baseline (2 HP, 1750 RPM)",
    "100.mat": "Normal Baseline (3 HP, 1730 RPM)",
    "106.mat": "Inner Race Fault (0.007 in, 1 HP)",
    "107.mat": "Inner Race Fault (0.007 in, 2 HP)",
    "108.mat": "Inner Race Fault (0.007 in, 3 HP)",
    "119.mat": "Ball Fault (0.007 in, 1 HP)",
    "120.mat": "Ball Fault (0.007 in, 2 HP)",
    "121.mat": "Ball Fault (0.007 in, 3 HP)",
    "131.mat": "Outer Race Fault @6:00 (0.007 in, 1 HP)",
    "132.mat": "Outer Race Fault @6:00 (0.007 in, 2 HP)",
    "133.mat": "Outer Race Fault @6:00 (0.007 in, 3 HP)",
}


def download_file(filename: str, desc: str, output_dir: Path) -> bool:
    """Download one CWRU .mat file with progress reporting."""
    dest_path = output_dir / filename
    url = f"{BASE_URL}/{filename}"

    if dest_path.exists() and dest_path.stat().st_size > 100_000:
        logger.info("Already downloaded: %s (%s, %d bytes)", filename, desc, dest_path.stat().st_size)
        return True

    logger.info("Downloading %s (%s)...", filename, desc)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MotorFaultDetector/1.0"},
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()

        if len(content) < 1000 or b"<!DOCTYPE" in content[:20]:
            logger.error("Download failed or returned HTML instead of .mat for %s", filename)
            return False

        with open(dest_path, "wb") as f:
            f.write(content)

        logger.info("Saved %s (%.2f MB)", filename, len(content) / (1024 * 1024))
        return True
    except Exception as exc:
        logger.error("Error downloading %s from %s: %s", filename, url, exc)
        return False


def main():
    parser = argparse.ArgumentParser(description="Download real CWRU benchmark files.")
    parser.add_argument("--output-dir", default="data/cwru", help="Destination directory")
    parser.add_argument("--full", action="store_true", help="Download extended suite (16 files across 0-3 HP)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files_to_download = EXTENDED_FILES if args.full else CORE_FILES
    logger.info("Target directory: %s", out_dir.resolve())
    logger.info("Files to download: %d", len(files_to_download))

    successes = 0
    for filename, desc in files_to_download.items():
        ok = download_file(filename, desc, out_dir)
        if ok:
            successes += 1

    print("\n" + "=" * 65)
    print(f"  CWRU Real Data Download: {successes}/{len(files_to_download)} files downloaded successfully")
    print("=" * 65)

    if successes == len(files_to_download):
        logger.info("All files downloaded! Run 'python scripts/validate_data.py --cwru-dir %s' to verify.", out_dir)
    else:
        logger.warning("Some files could not be downloaded. Check internet connection.")


if __name__ == "__main__":
    main()
