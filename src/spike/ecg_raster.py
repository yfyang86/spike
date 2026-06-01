"""
ecg_raster — digitize calibrated waveform time series from *raster* ECG images
(scans, screenshots, photos), the pixel-based counterpart to :mod:`ecg_pdf`.

Where ``ecg_pdf`` recovers samples *losslessly* from vector path operators, a
raster image has thrown that away: the waveform is just dark pixels on a grid.
We recover an approximation by (1) finding the grid scale (px per mm), (2)
masking the trace by *local* darkness so the recovery is grid-colour agnostic,
(3) locating the row baselines, and (4) tracking each row's centreline across
x with a velocity-predicted follower that climbs to the true R/S peaks instead
of averaging them away.

This is inherently lossy and approximate — verify against the source image and
prefer the vector pipeline whenever a vector PDF is available.

Pipeline
--------
1. Load the image, grayscale it, and find the waveform bounding box.
2. Grid scale: spacing of the heavy 5 mm grid lines (the solid, reliable comb),
   divided by 5 to give px per mm.
3. Trace mask: the darkest ink, via an Otsu split of the max-over-channels
   image. The max-channel makes a coloured (red / orange / pink) grid light (it
   is high in at least one channel) so it drops out; Otsu then separates the
   black trace from any remaining dark/grey grid.
4. Baselines: the per-row y where the trace dwells (projection peaks).
5. Track each row's centreline; split into the per-column leads, trimming the
   leading calibration pulse and the inter-lead pen transitions.

Calibration mirrors the vector module:
    mV = (baseline - y) / (px_per_mm * gain_mm_per_mV)
    s  = (x - x0)      / (px_per_mm * speed_mm_per_s)

Dependencies: numpy, pillow, scipy. Optional: matplotlib (via ``ECGResult.plot``).

Usage
-----
    from spike import extract_ecg_image
    ecg = extract_ecg_image("ECG_scan.png")
    ecg.rows_to_csv("rows.csv")
    t, mv = ecg.leads["V5"]
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ecg_pdf import DEFAULT_LAYOUT, ECGResult

__all__ = ["RasterECGExtractor", "extract_ecg_image"]


class RasterECGExtractor:
    """Digitize a raster ECG image into an :class:`ECGResult`.

    Parameters
    ----------
    gain_mm_per_mv, speed_mm_per_s : float
        Printed calibration (10 mm/mV, 25 mm/s by default).
    layout : sequence of rows
        Lead grid; the last row is the single rhythm lead (see ``DEFAULT_LAYOUT``).
    px_per_mm : float, optional
        Override the auto grid-scale detection.
    fs : float
        Output sample rate (Hz).
    nrows : int
        Number of printed rows to detect (defaults to ``len(layout)``).
    trace_darkness : float, optional
        ``None`` (default) masks the trace by an Otsu split of the darkness
        image. Set a value (0-255) to switch to local-contrast mode: keep pixels
        more than this much darker than a median-filtered background (for
        unevenly lit photos).
    bg_size : int
        Median-filter window (px) for the local-contrast background.
    window_frac : float
        Tracking half-window as a fraction of the inter-row spacing.
    max_jump_frac : float
        Reject per-column jumps larger than this fraction of the window
        (suppresses latching onto a neighbouring row's tall R/S wave).
    suppress_grid : bool
        Remove full-span grid lines (and isolated dots) from the trace mask.
        Essential for dark/near-black grids that survive the Otsu split; it also
        keeps a stray heavy line from being mistaken for a row baseline.
    grid_vspan, grid_hspan : float
        A mask column/row that is "on" for more than this fraction of the box
        height/width is treated as a heavy vertical/horizontal grid line and
        cleared. Heavy lines span the full box; the wandering trace does not.
    min_blob_px : int
        Drop connected components smaller than this (the dotted fine grid).
    row_sep_frac : float
        Minimum row-baseline separation as a fraction of ``box_height / nrows``.
    trim_connector_mv, trim_calibration_mv : float
        Off-baseline thresholds used to drop inter-lead pen transitions and the
        leading calibration pulse.
    """

    def __init__(
        self,
        gain_mm_per_mv: float = 10.0,
        speed_mm_per_s: float = 25.0,
        layout: Sequence[Sequence[str]] = DEFAULT_LAYOUT,
        px_per_mm: Optional[float] = None,
        fs: float = 100.0,
        nrows: Optional[int] = None,
        trace_darkness: Optional[float] = None,
        bg_size: int = 9,
        window_frac: float = 0.45,
        max_jump_frac: float = 0.6,
        steep_run_px: int = 4,
        suppress_grid: bool = True,
        grid_vspan: float = 0.5,
        grid_hspan: float = 0.7,
        min_blob_px: int = 4,
        row_sep_frac: float = 0.7,
        trim_connector_mv: float = 0.5,
        trim_calibration_mv: float = 0.5,
    ):
        self.gain = gain_mm_per_mv
        self.speed = speed_mm_per_s
        self.layout = [list(r) for r in layout]
        self.px_per_mm = px_per_mm
        self.fs = fs
        self.nrows = nrows or len(self.layout)
        self.trace_darkness = trace_darkness
        self.bg_size = bg_size
        self.window_frac = window_frac
        self.max_jump_frac = max_jump_frac
        self.steep_run_px = steep_run_px
        self.suppress_grid = suppress_grid
        self.grid_vspan = grid_vspan
        self.grid_hspan = grid_hspan
        self.min_blob_px = min_blob_px
        self.row_sep_frac = row_sep_frac
        self.trim_connector_mv = trim_connector_mv
        self.trim_calibration_mv = trim_calibration_mv

    # -- public ----------------------------------------------------------- #
    def extract(self, image_path: str) -> ECGResult:
        ink, dark = self._load_channels(image_path)
        box = self._bounding_box(ink)
        x0, y0, x1, y1 = box

        pmm = self.px_per_mm or self._grid_px_per_mm(ink, box)
        upmv = pmm * self.gain
        ups = pmm * self.speed

        mask = self._trace_mask(dark, box)
        if self.suppress_grid:
            mask = self._suppress_grid_lines(mask, box)

        baselines, spacing = self._baselines(mask, box, self.nrows)
        win = int(round(spacing * self.window_frac))

        xfull = np.arange(x0, x1)
        strip_secs = (x1 - x0) / ups
        n_cols = max(len(r) for r in self.layout[:-1]) if len(self.layout) > 1 else 1
        col_secs = strip_secs / n_cols

        leads: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        rows: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        rhythm_lead_names: set = set()

        for ri, names in enumerate(self.layout):
            if ri >= len(baselines):
                break
            yc = self._track_row(mask, baselines[ri], win, xfull, y0, y1)
            base = self._baseline_value(yc)
            t_row = (xfull - x0) / ups
            v_row = (base - yc) / upmv
            rows[f"row{ri + 1}"] = self._resample(t_row, v_row, self.fs)

            is_rhythm = (len(names) == 1)
            if is_rhythm:
                name = names[0]
                t, v = self._trim_calibration(t_row, v_row)
                leads[name] = self._resample_fixed(t - (t[0] if len(t) else 0),
                                                   v, self.fs, strip_secs)
                rhythm_lead_names.add(name)
                continue

            # split the short row into its per-column leads by time window
            for ci, name in enumerate(names):
                lo, hi = ci * col_secs, (ci + 1) * col_secs
                sel = (t_row >= lo) & (t_row < hi)
                t, v = t_row[sel] - lo, v_row[sel]
                if ci == 0:
                    t, v = self._trim_calibration(t, v)
                else:
                    t, v = self._trim_connector(t, v)
                t = t - (t[0] if len(t) else 0.0)
                leads[name] = self._resample_fixed(t, v, self.fs, col_secs)

        meta = {
            "source": "raster",
            "acquisition": {
                "gain_mm_per_mV": self.gain,
                "paper_speed_mm_per_s": self.speed,
            },
        }
        return ECGResult(
            leads=leads, rows=rows, meta=meta, fs=self.fs,
            units_per_mm=pmm, units_per_mv=upmv, units_per_s=ups,
            layout=self.layout, n_columns=n_cols,
            column_seconds=col_secs, strip_seconds=strip_secs,
        )

    # -- image / grid ----------------------------------------------------- #
    @staticmethod
    def _load_channels(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(ink, dark)`` lightness maps.

        ``ink`` = per-pixel min over RGB: low wherever there is any ink (grid or
        trace, any colour), ~255 on the white background — used to find the grid
        region and pitch. ``dark`` = per-pixel max over RGB: low only where the
        pixel is dark in *every* channel (the black trace); a coloured grid is
        light here and drops out — used to mask the trace.
        """
        from PIL import Image
        rgb = np.asarray(Image.open(image_path).convert("RGB")).astype(float)
        return rgb.min(2), rgb.max(2)

    @staticmethod
    def _bounding_box(L: np.ndarray) -> Tuple[int, int, int, int]:
        """Box the dense grid/waveform region (excludes the sparse text header)."""
        nw = L < 232
        cols = np.where(nw.mean(0) > 0.5)[0]
        rows = np.where(nw.mean(1) > 0.4)[0]
        if len(cols) == 0 or len(rows) == 0:
            raise ValueError("Could not locate the ECG grid region in the image.")
        return int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())

    def _grid_px_per_mm(self, L: np.ndarray, box) -> float:
        """Pixels per mm from the *heavy* (5 mm) grid lines.

        Clinical ECG paper carries a heavy line every 5 mm; these are solid and
        the most reliable feature, whereas the fine 1 mm grid is often
        dotted/faint and can either vanish from or dominate a naive FFT (which
        silently sets the scale 5x wrong). We detect the prominent, regularly
        spaced vertical lines (the heavy grid) from the ink projection and divide
        their median spacing by 5.
        """
        from scipy.signal import find_peaks

        x0, y0, x1, y1 = box
        # darkness-weighted column projection: heavy (darker) lines score higher
        # than a faint or dotted fine grid, so they stand out as the major peaks
        col = np.clip(232.0 - L[y0:y1, x0:x1], 0, None).sum(0)
        pk, props = find_peaks(col, distance=2, prominence=1)
        if len(pk) < 3:
            raise ValueError("Could not detect grid lines; pass px_per_mm.")
        prom = props["prominences"]
        major = pk[prom >= 0.5 * prom.max()]
        spacing = (np.median(np.diff(major)) if len(major) > 2
                   else np.median(np.diff(pk)))
        pmm = float(spacing) / 5.0
        if not (2.0 <= pmm <= 60.0):
            raise ValueError(
                f"Auto-detected px/mm={pmm:.2f} is implausible; pass px_per_mm."
            )
        return pmm

    def _trace_mask(self, dark: np.ndarray, box) -> np.ndarray:
        """Mask the waveform.

        The trace is the darkest ink: darker than the grid in *every* channel, so
        an Otsu split of the max-over-channels image over the inked pixels
        isolates it for coloured *and* dark grids alike. Setting
        ``trace_darkness`` switches to a local-contrast mode (pixels much darker
        than a median-filtered background) for unevenly lit photos.
        """
        x0, y0, x1, y1 = box
        mask = np.zeros(dark.shape, bool)
        if self.trace_darkness is not None:
            from scipy import ndimage
            bg = ndimage.median_filter(dark, size=(self.bg_size, self.bg_size))
            mask[y0:y1, x0:x1] = (bg - dark)[y0:y1, x0:x1] > self.trace_darkness
            return mask
        region = dark[y0:y1, x0:x1]
        thr = self._otsu(region[region < 232])
        mask[y0:y1, x0:x1] = region < thr
        return mask

    @staticmethod
    def _otsu(d: np.ndarray) -> float:
        if d.size == 0:
            return 128.0
        hist, edges = np.histogram(d, bins=64)
        p = hist / hist.sum()
        centers = (edges[:-1] + edges[1:]) / 2
        w = np.cumsum(p)
        mu = np.cumsum(p * centers)
        muT = mu[-1]
        denom = w * (1 - w)
        denom[denom == 0] = 1e-12
        sigma_b = (muT * w - mu) ** 2 / denom
        return float(centers[int(np.argmax(sigma_b))])

    def _suppress_grid_lines(self, mask: np.ndarray, box) -> np.ndarray:
        """Clear full-span grid lines and isolated dots from the trace mask.

        Heavy grid lines span the full box height (vertical) or width
        (horizontal); the wandering trace never does, so a high span fraction
        identifies them. A connected-component size filter then drops the dotted
        fine grid. Punching a 1-px gap where the trace crosses a removed line is
        harmless — the tracker interpolates across it.
        """
        from scipy import ndimage

        x0, y0, x1, y1 = box
        sub = mask[y0:y1, x0:x1].copy()
        h, w = sub.shape
        sub[:, sub.sum(0) > self.grid_vspan * h] = False
        sub[sub.sum(1) > self.grid_hspan * w, :] = False
        if self.min_blob_px > 1:
            lab, n = ndimage.label(sub)
            if n:
                sizes = np.bincount(lab.ravel())
                keep = sizes >= self.min_blob_px
                keep[0] = False
                sub = keep[lab]
        out = mask.copy()
        out[y0:y1, x0:x1] = sub
        return out

    def _baselines(self, mask: np.ndarray, box, nrows: int) -> Tuple[np.ndarray, int]:
        """Row baselines = y-positions where the trace dwells (projection peaks)."""
        from scipy import ndimage, signal

        x0, y0, x1, y1 = box
        proj = ndimage.gaussian_filter1d(mask[:, x0:x1].sum(1).astype(float), 3)
        minsep = int(self.row_sep_frac * (y1 - y0) / nrows)
        pk, _ = signal.find_peaks(proj, distance=max(1, minsep))
        if len(pk) == 0:
            # fall back to even spacing
            pk = np.linspace(y0, y1, nrows + 2)[1:-1].astype(int)
        else:
            pk = np.sort(pk[np.argsort(proj[pk])[::-1][:nrows]])
        spacing = int(np.median(np.diff(pk))) if len(pk) > 1 else (y1 - y0) // nrows
        return pk, max(spacing, 4)

    # -- tracking --------------------------------------------------------- #
    @staticmethod
    def _runs(ys: np.ndarray) -> List[Tuple[float, int, int]]:
        if len(ys) == 0:
            return []
        ys = np.sort(ys)
        brk = np.where(np.diff(ys) > 2)[0]
        return [(float(s.mean()), int(s.min()), int(s.max()))
                for s in np.split(ys, brk + 1)]

    def _track_row(self, mask, base, win, xfull, y0, y1) -> np.ndarray:
        """Follow one row's centreline across x.

        A constant-velocity prediction picks the right run when a column holds
        several. On a *short* run (flat / smooth segment) we take the centre to
        avoid jitter; on a *tall* run (a steep QRS limb, where the near-vertical
        pen fills a whole column) we take the endpoint that extends the
        trajectory — the one farther from the previous sample — so the trace
        climbs to the true R/S peak instead of being averaged to mid-height.
        Implausible jumps (a tall neighbouring-row deflection leaking into the
        window) are clamped.
        """
        lo = max(int(base - win), y0)
        hi = min(int(base + win), y1)
        max_jump = self.max_jump_frac * win
        steep = self.steep_run_px
        yc = np.full(len(xfull), np.nan)
        prev = float(base)
        vel = 0.0
        for j, xx in enumerate(xfull):
            yy = np.where(mask[lo:hi, xx])[0]
            if len(yy) == 0:
                continue
            yy = yy + lo
            pred = prev + vel
            rs = self._runs(yy)

            def dist(r):
                return 0.0 if r[1] <= pred <= r[2] else min(abs(r[1] - pred),
                                                            abs(r[2] - pred))
            r = min(rs, key=dist)
            if (r[2] - r[1]) <= steep:
                cand = r[0]                       # short run: smooth centre
            else:                                 # steep limb: extend trajectory
                cand = r[1] if abs(r[1] - prev) >= abs(r[2] - prev) else r[2]
            if abs(cand - prev) > max_jump:
                cand = float(np.clip(cand, prev - max_jump, prev + max_jump))
            vel = 0.5 * vel + 0.5 * (cand - prev)
            yc[j] = float(cand)
            prev = float(cand)

        idx = np.arange(len(yc))
        good = ~np.isnan(yc)
        if not good.any():
            return np.full(len(yc), float(base))
        return np.interp(idx, idx[good], yc[good])

    @staticmethod
    def _baseline_value(yc: np.ndarray) -> float:
        hist, edges = np.histogram(yc, bins=80)
        k = int(np.argmax(hist))
        return float((edges[k] + edges[k + 1]) / 2)

    # -- trimming / resampling ------------------------------------------- #
    def _trim_connector(self, t, v):
        k = 0
        while k < len(v) - 1 and abs(v[k]) > self.trim_connector_mv:
            k += 1
        if k:
            return t[k:], v[k:]
        return t, v

    def _trim_calibration(self, t, v):
        """Skip the leading calibration pulse: an initial off-baseline excursion
        followed by a settled return to baseline."""
        k = 0
        n = len(v)
        # advance through the first off-baseline excursion
        while k < n - 1 and abs(v[k]) <= self.trim_calibration_mv:
            k += 1
        while k < n - 1 and abs(v[k]) > self.trim_calibration_mv:
            k += 1
        if k >= n - 1:
            return t, v
        return t[k:], v[k:]

    @staticmethod
    def _resample(t, v, fs):
        if len(t) == 0:
            return np.array([]), np.array([])
        grid = np.arange(0, t.max(), 1.0 / fs)
        return grid, np.interp(grid, t, v)

    @staticmethod
    def _resample_fixed(t, v, fs, dur):
        if len(t) == 0:
            return np.array([]), np.array([])
        n = int(round(dur * fs))
        grid = np.linspace(0, dur, n, endpoint=False)
        return grid, np.interp(grid, t, v)


def extract_ecg_image(image_path: str, **kwargs) -> ECGResult:
    """Convenience wrapper: ``RasterECGExtractor(**kwargs).extract(image_path)``."""
    return RasterECGExtractor(**kwargs).extract(image_path)
