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


def test_segments_tolerate_crlf():
    # the segment tokenizer must cope with CRLF line endings, not just \n
    content = "10 20 m\r\n30 40 l\r\nS\r\n50 60 m\n70 80 l\nS\n"
    segs = ECGExtractor._segments(content)
    assert segs == [(10.0, 20.0, 30.0, 40.0), (50.0, 60.0, 70.0, 80.0)]


def test_no_segments_returns_empty():
    # a content stream with no vector segments yields no polylines
    assert ECGExtractor()._polylines([]) == []


def test_raster_pdf_raises(monkeypatch):
    # a content stream with no vector segments should fail clearly in extract()
    ext = ECGExtractor()
    monkeypatch.setattr(ext, "_content_stream", lambda path: ("q Q\n", 0))
    with pytest.raises(ValueError, match="vector"):
        ext.extract("dummy.pdf")
