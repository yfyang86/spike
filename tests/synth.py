"""Shared synthetic ECG-image generator for the raster/CLI tests.

Kept out of any patient data: builds a faint 1 mm grid + a darker heavy 5 mm
grid + ``nrows`` baselines, each a connected polyline with periodic triangular
R waves of known height (drawn as a real trace, not filled bars)."""
import numpy as np


def synthetic_ecg_png(path, pmm=8, speed=25, gain=10, nrows=4):
    """Write a synthetic ECG PNG to ``path``; return ``(r_mv, pmm)``."""
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
        yt += 3.0 * np.sin(np.linspace(0, 4 * np.pi, len(x)))   # gentle baseline wander
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
