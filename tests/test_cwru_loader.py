"""
tests/test_cwru_loader.py
-------------------------
Unit tests for data.loader (CWRULoader, CWRURecord, and filename parsing).
Tests both synthetic .mat structures and real generated CWRU datasets.
"""

from pathlib import Path
import numpy as np
import pytest
import scipy.io

from data.loader import (
    CWRULoader,
    CWRURecord,
    CWRU_SAMPLE_RATE_HZ,
    CWRU_RUN_FREQ_HZ,
    LABEL_HEALTHY,
    LABEL_INNER_RACE,
    LABEL_OUTER_RACE,
    LABEL_BALL,
)


# ---------------------------------------------------------------------------
# CWRURecord unit tests
# ---------------------------------------------------------------------------

class TestCWRURecord:
    def test_record_properties_healthy(self):
        rec = CWRURecord(
            path=Path("Normal_0.mat"),
            fault_type=LABEL_HEALTHY,
            fault_diameter_thou=None,
            load_hp=0,
            de_signal=np.zeros(100, dtype=np.float32),
            fe_signal=None,
            sample_rate=12_000,
            rpm=1797.0,
        )
        assert rec.is_healthy is True
        assert rec.label_str == LABEL_HEALTHY
        assert "healthy" in repr(rec)

    def test_record_properties_faulty(self):
        rec = CWRURecord(
            path=Path("IR007_1.mat"),
            fault_type=LABEL_INNER_RACE,
            fault_diameter_thou=7,
            load_hp=1,
            de_signal=np.ones(100, dtype=np.float32),
            fe_signal=None,
            sample_rate=12_000,
            rpm=1772.0,
        )
        assert rec.is_healthy is False
        assert rec.label_str == "inner_race_7thou_load1"
        assert "inner_race" in repr(rec)


# ---------------------------------------------------------------------------
# Filename parsing tests
# ---------------------------------------------------------------------------

class TestFilenameParsing:
    @pytest.mark.parametrize(
        "stem,expected_fault,expected_diam,expected_load",
        [
            ("Normal_0", LABEL_HEALTHY, None, 0),
            ("Normal_3", LABEL_HEALTHY, None, 3),
            ("normal_1", LABEL_HEALTHY, None, 1),
            ("IR007_0", LABEL_INNER_RACE, 7, 0),
            ("IR014_2", LABEL_INNER_RACE, 14, 2),
            ("IR021_3", LABEL_INNER_RACE, 21, 3),
            ("OR007@6_0", LABEL_OUTER_RACE, 7, 0),
            ("OR014@6_1", LABEL_OUTER_RACE, 14, 1),
            ("OR021@12_2", LABEL_OUTER_RACE, 21, 2),
            ("B007_0", LABEL_BALL, 7, 0),
            ("B014_1", LABEL_BALL, 14, 1),
            ("B021_3", LABEL_BALL, 21, 3),
            ("97", LABEL_HEALTHY, None, 0),
            ("98", LABEL_HEALTHY, None, 1),
            ("105", LABEL_INNER_RACE, 7, 0),
            ("118", LABEL_BALL, 7, 0),
            ("130", LABEL_OUTER_RACE, 7, 0),
        ],
    )
    def test_standard_filenames(self, stem, expected_fault, expected_diam, expected_load):
        fault, diam, load = CWRULoader._parse_filename(stem)
        assert fault == expected_fault
        assert diam == expected_diam
        assert load == expected_load

    def test_non_standard_fallback(self):
        fault, diam, load = CWRULoader._parse_filename("custom_normal_run")
        assert fault == LABEL_HEALTHY

        fault, diam, load = CWRULoader._parse_filename("ir_seeded_motor")
        assert fault == LABEL_INNER_RACE

        fault, diam, load = CWRULoader._parse_filename("completely_unknown_xyz")
        assert fault == "unknown"


# ---------------------------------------------------------------------------
# Loader with temporary .mat files
# ---------------------------------------------------------------------------

class TestCWRULoaderOperations:
    @pytest.fixture
    def mock_cwru_dir(self, tmp_path):
        """Create a temporary directory with 3 mini .mat files."""
        # 1. Healthy file
        mat_normal = {
            "X097_DE_time": np.sin(np.linspace(0, 1, 1200, dtype=np.float32)).reshape(-1, 1),
            "X097_FE_time": (np.cos(np.linspace(0, 1, 1200, dtype=np.float32)) * 0.5).reshape(-1, 1),
            "X097_RPM": np.array([[1797.0]]),
        }
        scipy.io.savemat(str(tmp_path / "Normal_0.mat"), mat_normal)

        # 2. Inner race fault file
        mat_ir = {
            "X105_DE_time": (np.random.randn(1200).astype(np.float32)).reshape(-1, 1),
            "X105_RPM": np.array([[1797.0]]),
        }
        scipy.io.savemat(str(tmp_path / "IR007_0.mat"), mat_ir)

        # 3. Outer race fault file
        mat_or = {
            "X130_DE_time": (np.random.randn(1200).astype(np.float32) * 1.5).reshape(-1, 1),
            "X130_RPM": np.array([[1772.0]]),
        }
        scipy.io.savemat(str(tmp_path / "OR007@6_1.mat"), mat_or)

        return tmp_path

    def test_loader_discovery(self, mock_cwru_dir):
        loader = CWRULoader(mock_cwru_dir)
        files = loader.discover()
        assert len(files) == 3
        filenames = [f.name for f in files]
        assert "Normal_0.mat" in filenames
        assert "IR007_0.mat" in filenames
        assert "OR007@6_1.mat" in filenames

    def test_loader_load_all(self, mock_cwru_dir):
        loader = CWRULoader(mock_cwru_dir, load_fe=True)
        records = loader.load_all()
        assert len(records) == 3

        normal_rec = next(r for r in records if r.is_healthy)
        assert normal_rec.fault_type == LABEL_HEALTHY
        assert len(normal_rec.de_signal) == 1200
        assert normal_rec.fe_signal is not None
        assert len(normal_rec.fe_signal) == 1200
        assert normal_rec.rpm == pytest.approx(1797.0)

        ir_rec = next(r for r in records if r.fault_type == LABEL_INNER_RACE)
        assert ir_rec.fault_diameter_thou == 7
        assert ir_rec.load_hp == 0

        or_rec = next(r for r in records if r.fault_type == LABEL_OUTER_RACE)
        assert or_rec.load_hp == 1

    def test_loader_load_healthy_only(self, mock_cwru_dir):
        loader = CWRULoader(mock_cwru_dir)
        healthy_recs = loader.load_healthy()
        assert len(healthy_recs) == 1
        assert healthy_recs[0].fault_type == LABEL_HEALTHY

    def test_loader_load_by_fault(self, mock_cwru_dir):
        loader = CWRULoader(mock_cwru_dir)
        ir_recs = loader.load_by_fault(LABEL_INNER_RACE)
        assert len(ir_recs) == 1
        assert ir_recs[0].fault_type == LABEL_INNER_RACE

        ball_recs = loader.load_by_fault(LABEL_BALL)
        assert len(ball_recs) == 0

    def test_loader_summary_does_not_crash(self, mock_cwru_dir, capsys):
        loader = CWRULoader(mock_cwru_dir)
        loader.summary()
        captured = capsys.readouterr()
        assert "CWRU Dataset Summary" in captured.out
        assert "Normal_0.mat" in captured.out

    def test_loader_missing_dir_raises(self, tmp_path):
        missing_dir = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            CWRULoader(missing_dir)

    def test_loader_handles_corrupt_file_gracefully(self, mock_cwru_dir):
        corrupt_file = mock_cwru_dir / "Corrupt_0.mat"
        corrupt_file.write_text("not a real mat file")

        loader = CWRULoader(mock_cwru_dir)
        records = loader.load_all()
        # Should skip corrupt file and load the 3 valid ones
        assert len(records) == 3
