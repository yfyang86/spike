"""Synthetic-data tests for the raster digitizer (no patient images shipped)."""
import numpy as np
import pytest

scipy = pytest.importorskip("scipy")

from spike import RasterECGExtractor, extract_ecg_image  # noqa: E402


def test_grid_px_per_mm_from_heavy_lines():
    # faint fine grid every 8 px + heavy lines every 5th (40 px) -> 40/5 = 8 px/mm
    L = np.full((200, 400), 255.0)
    L[:, ::8] = 180.0       # fine 1 mm grid
    L[:, ::40] = 60.0       # heavy 5 mm grid (more prominent)
    ext = RasterECGExtractor()
    pmm = ext._grid_px_per_mm(L, (0, 0, 400, 200))
    assert pmm == pytest.approx(8.0, abs=0.5)


def test_tracker_recovers_peak_not_midpoint():
    # a vertical spike: column of dark pixels from baseline up to a known peak.
    # taking the run *mean* would report mid-height; the tracker must reach the tip.
    H, W = 200, 60
    mask = np.zeros((H, W), bool)
    base = 150
    mask[base, :] = True                 # flat baseline
    peak = 60                            # 90 px above baseline
    mask[peak:base, 30] = True           # one steep upstroke column
    ext = RasterECGExtractor(max_jump_frac=5.0)
    yc = ext._track_row(mask, base, win=120, xfull=np.arange(W), y0=0, y1=H)
    assert yc.min() == pytest.approx(peak, abs=3)   # reached the tip, not (peak+base)/2


def test_trim_calibration_drops_leading_pulse():
    ext = RasterECGExtractor(trim_calibration_mv=0.5)
    t = np.arange(20, dtype=float)
    v = np.zeros(20)
    v[2:6] = 1.0          # leading calibration pulse
    v[10] = 1.5           # a real R wave that must survive
    tt, vv = ext._trim_calibration(t, v)
    assert vv.max() == pytest.approx(1.5)
    assert len(vv) < len(v)               # the pulse region was dropped


def _synthetic_ecg_png(path, pmm=8, speed=25, gain=10, nrows=4):
    """A faint 1 mm grid + 4 baselines, each a connected polyline with periodic
    triangular R waves of known height (drawn as a real trace, not filled bars)."""
    from PIL import Image

    ups = pmm * speed
    upmv = pmm * gain
    secs = 4.0
    spacing = 200
    top = 40
    W = int(secs * ups) + 40
    H = top + nrows * spacing + 40
    L = np.full((H, W), 255, np.uint8)
    # faint fine 1 mm grid + a darker heavy line every 5 mm (as on real ECG
    # paper); both are lighter than the near-black trace so Otsu masks it out
    g = int(round(pmm))
    L[:, ::g] = 230
    L[::g, :] = 230
    L[:, :: 5 * g] = 180
    L[:: 5 * g, :] = 180
    baselines = [top + spacing // 2 + i * spacing for i in range(nrows)]
    x = np.arange(20, W - 20)
    r_mv = 0.8                        # known R-wave height in mV
    r_px = int(round(r_mv * upmv))
    hw = 8                            # half-width of the triangular R (px)
    period = int(0.8 * ups)
    for b in baselines:
        yt = np.full(len(x), float(b))
        for xc0 in range(int(0.3 * ups), len(x), period):   # skip a flat lead-in
            for dx in range(-hw, hw + 1):
                j = xc0 + dx
                if 0 <= j < len(x):
                    yt[j] = b - r_px * (1 - abs(dx) / hw)
        yt = yt.astype(int)
        for i, xx in enumerate(x):                          # draw a connected line
            L[yt[i] - 1:yt[i] + 1, xx] = 30
            if i > 0:
                a, c = sorted((yt[i - 1], yt[i]))
                L[a:c + 1, xx] = 30
    Image.fromarray(L).save(path)
    return r_mv, pmm


def test_end_to_end_synthetic_image(tmp_path):
    path = str(tmp_path / "synth.png")
    r_mv, pmm = _synthetic_ecg_png(path)
    ecg = extract_ecg_image(path)

    # structure: 4 rows, 13 named leads (4+4+4 + rhythm)
    assert set(ecg.rows) == {"row1", "row2", "row3", "row4"}
    assert "II_rhythm" in ecg.leads
    assert len(ecg.leads) == sum(len(r) for r in ecg.layout)

    # grid scale recovered within ~15 %
    assert ecg.units_per_mm == pytest.approx(pmm, rel=0.15)

    # R-wave amplitude recovered (peaks must not be averaged away)
    _, v = ecg.rows["row1"]
    assert v.max() == pytest.approx(r_mv, abs=0.25)


def test_missing_image_dep_is_optional_import():
    # importing the package must not require pillow/scipy at import time
    import importlib
    import spike
    importlib.reload(spike)
    assert hasattr(spike, "extract_ecg_image")
