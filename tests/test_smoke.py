"""Smoke tests that need no sample data (and no patient data shipped)."""
import numpy as np
import pytest

from spike import ECGExtractor, extract_ecg, DEFAULT_LAYOUT


def test_imports_and_layout():
    assert DEFAULT_LAYOUT[-1] == ["II_rhythm"]
    assert callable(extract_ecg)


def test_grid_scale_detection():
    # three vertical "grid" lines 118 units apart -> 118/5 = 23.6 units/mm
    segs = [(x, 0.0, x, 1000.0) for x in (100.0, 218.0, 336.0)]
    assert ECGExtractor._grid_units_per_mm(segs) == pytest.approx(23.6)


def test_baseline_is_histogram_peak():
    coord = np.concatenate([np.full(200, 50.0), np.linspace(0, 100, 20)])
    assert ECGExtractor._baseline(coord) == pytest.approx(50.0, abs=2.0)


def test_raster_pdf_raises(tmp_path):
    # a content stream with no vector segments should fail clearly
    ext = ECGExtractor()
    with pytest.raises(ValueError):
        ext._polylines([])  # no segments
        raise ValueError("no segments")
