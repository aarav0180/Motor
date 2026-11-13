"""
tests/test_data_pipeline.py  —  Tests for the data pipeline.

These tests run without any CWRU .mat files by using synthetic data
injected via the internal helpers.  The CWRULoader itself is tested only
when real .mat files are present (skipped otherwise).
"""

import numpy as np
import pytest

from data.cleaner  import (
    SignalCleaner, CleanerConfig, clean_signal,
    remove_dc, detrend_signal, clip_outliers,
    normalise_unit_rms, normalise_minmax,
)
from data.windows  import (
    LabelledWindow, segment_signal, windows_to_arrays,
)
from data.splitter import DataSplitter, Split


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FS   = 12_000
WIN  = FS * 10   # 10 s window

rng  = np.random.default_rng(0)


def _make_window(label: str, seed: int = 0) -> LabelledWindow:
    rng2 = np.random.default_rng(seed)
    sig  = rng2.standard_normal(WIN).astype(np.float32)
    return LabelledWindow(signal=sig, label=label, source="synthetic")


@pytest.fixture
def healthy_windows():
    return [_make_window("healthy", i) for i in range(50)]


@pytest.fixture
def faulty_windows():
    return [_make_window("inner_race", i + 100) for i in range(20)]


@pytest.fixture
def all_windows(healthy_windows, faulty_windows):
    return healthy_windows + faulty_windows


# ===========================================================================
# Signal cleaner tests
# ===========================================================================

class TestCleanerSteps:
    def test_remove_dc_zeroes_mean(self):
        s = np.ones(1000, dtype=np.float32) * 5.0
        s += rng.standard_normal(1000).astype(np.float32) * 0.1
        out = remove_dc(s)
        assert abs(np.mean(out)) < 1e-5

    def test_clip_outliers_removes_spikes(self):
        # Use Gaussian noise base so std > 0, then inject spikes.
        rng2 = np.random.default_rng(42)
        s = rng2.standard_normal(1000).astype(np.float32)
        s[50]  =  100.0
        s[500] = -100.0
        mu, std = float(np.mean(s)), float(np.std(s))
        out = clip_outliers(s, k=3.0)
        # Verify extreme spikes are clipped to exactly mu ± 3*std
        assert out.max() < s.max()
        assert out.min() > s.min()
        assert out.max() == pytest.approx(mu + 3.0 * std, rel=1e-3)
        assert out.min() == pytest.approx(mu - 3.0 * std, rel=1e-3)

    def test_normalise_unit_rms(self):
        s   = rng.standard_normal(1000).astype(np.float32) * 7.0
        out = normalise_unit_rms(s)
        assert abs(np.sqrt(np.mean(out ** 2)) - 1.0) < 1e-4

    def test_normalise_minmax_range(self):
        s   = rng.standard_normal(1000).astype(np.float32)
        out = normalise_minmax(s)
        assert out.min() >= -1.0 - 1e-6
        assert out.max() <=  1.0 + 1e-6

    def test_clean_signal_preserves_length(self):
        s = rng.standard_normal(WIN).astype(np.float32)
        out = clean_signal(s, sample_rate=FS)
        assert len(out) == len(s)

    def test_clean_signal_dtype(self):
        s   = rng.standard_normal(WIN).astype(np.float64)
        out = clean_signal(s, sample_rate=FS)
        assert out.dtype == np.float32


class TestSignalCleaner:
    def test_batch_clean(self):
        cleaner = SignalCleaner(sample_rate=FS)
        signals = [rng.standard_normal(WIN).astype(np.float32) for _ in range(5)]
        results = cleaner.clean_batch(signals)
        assert len(results) == 5
        for r in results:
            assert len(r) == WIN

    def test_counter_increments(self):
        cleaner = SignalCleaner(sample_rate=FS)
        for _ in range(3):
            cleaner.clean(rng.standard_normal(1000).astype(np.float32))
        assert cleaner.signals_cleaned == 3

    def test_diagnose_returns_before_after(self):
        cleaner = SignalCleaner(sample_rate=FS)
        s = rng.standard_normal(WIN).astype(np.float32) * 5
        d = cleaner.diagnose(s)
        assert "before" in d and "after" in d
        assert "rms" in d["before"]

    def test_adxl345_preset_bandpass_range(self):
        cfg = CleanerConfig.for_adxl345()
        assert cfg.bandpass_low_hz  ==  5.0
        assert cfg.bandpass_high_hz == 450.0

    def test_cwru_preset_bandpass_range(self):
        cfg = CleanerConfig.for_cwru()
        assert cfg.bandpass_high_hz == 5_500.0


# ===========================================================================
# Windowing tests
# ===========================================================================

class TestWindowing:
    def test_segment_non_overlapping(self):
        s    = np.ones(WIN * 3, dtype=np.float32)
        segs = segment_signal(s, WIN)
        assert len(segs) == 3
        assert all(len(seg) == WIN for seg in segs)

    def test_segment_overlapping(self):
        s    = np.ones(WIN * 2, dtype=np.float32)
        segs = segment_signal(s, WIN, hop_size=WIN // 2)
        assert len(segs) == 3   # 2 non-overlap + 1 extra from overlap

    def test_segment_drops_tail(self):
        s    = np.ones(WIN + 100, dtype=np.float32)
        segs = segment_signal(s, WIN)
        assert len(segs) == 1    # tail 100 samples dropped

    def test_windows_to_arrays_shape(self, healthy_windows):
        X, labels = windows_to_arrays(healthy_windows)
        assert X.shape == (50, WIN)
        assert len(labels) == 50

    def test_labelled_window_is_healthy(self):
        w = _make_window("healthy")
        assert w.is_healthy is True

    def test_labelled_window_not_healthy(self):
        w = _make_window("ball")
        assert w.is_healthy is False


# ===========================================================================
# Splitter tests
# ===========================================================================

class TestDataSplitter:
    def test_split_proportions(self, all_windows):
        sp = DataSplitter(mock_fraction=0.2, train_fraction=0.8).split(all_windows)
        total_healthy = 50
        n_mock  = int(total_healthy * 0.2)       # 10
        n_rest  = total_healthy - n_mock          # 40
        n_train = int(n_rest * 0.8)              # 32
        n_eval_h = n_rest - n_train              # 8
        assert len(sp.mock_sensor)   == n_mock
        assert len(sp.train_healthy) == n_train
        assert len(sp.eval_healthy)  == n_eval_h
        assert len(sp.eval_faulty)   == 20

    def test_no_overlap_between_groups(self, all_windows):
        sp  = DataSplitter().split(all_windows)
        ids = lambda grp: {id(w) for w in grp}
        all_ids = [
            ids(sp.train_healthy),
            ids(sp.mock_sensor),
            ids(sp.eval_healthy),
            ids(sp.eval_faulty),
        ]
        for i, a in enumerate(all_ids):
            for j, b in enumerate(all_ids):
                if i != j:
                    assert a.isdisjoint(b), f"Overlap between groups {i} and {j}"

    def test_eval_faulty_all_faulty(self, all_windows):
        sp = DataSplitter().split(all_windows)
        assert all(not w.is_healthy for w in sp.eval_faulty)

    def test_train_healthy_all_healthy(self, all_windows):
        sp = DataSplitter().split(all_windows)
        assert all(w.is_healthy for w in sp.train_healthy)

    def test_eval_labels_binary(self, all_windows):
        sp = DataSplitter().split(all_windows)
        assert set(sp.eval_labels).issubset({0, 1})

    def test_all_eval_combines_correctly(self, all_windows):
        sp = DataSplitter().split(all_windows)
        assert len(sp.all_eval) == len(sp.eval_healthy) + len(sp.eval_faulty)

    def test_raises_on_no_healthy(self):
        windows = [_make_window("ball") for _ in range(10)]
        with pytest.raises(ValueError, match="No healthy"):
            DataSplitter().split(windows)

    def test_reproducible_with_same_seed(self, all_windows):
        sp1 = DataSplitter(seed=7).split(all_windows)
        sp2 = DataSplitter(seed=7).split(all_windows)
        assert [id(w) for w in sp1.mock_sensor] == [id(w) for w in sp2.mock_sensor]

    def test_different_seeds_give_different_splits(self, all_windows):
        sp1 = DataSplitter(seed=1).split(all_windows)
        sp2 = DataSplitter(seed=2).split(all_windows)
        assert [id(w) for w in sp1.mock_sensor] != [id(w) for w in sp2.mock_sensor]
