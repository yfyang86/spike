"""
ecg_pdf — extract calibrated waveform time series (and metadata) from
*vector* ECG report PDFs.

These PDFs (e.g. those produced by the GPL Ghostscript pipeline used by some
hospital ECG carts) draw each waveform as a sequence of PostScript-style
`x y m / x y l / S` (moveto / lineto / stroke) operators rather than as a
raster image. That means the underlying samples are recoverable losslessly.

Validated against a Shanghai-Tenth-People's-Hospital 12-lead report:
standard Mason-Likar 4x3 layout + one rhythm strip, 25 mm/s, 10 mm/mV,
`/Rotate 90`. The waveform extractor auto-detects the grid scale, the lead
baselines, the time axis and the signal polarity, so it should also handle
sibling formats; anything format-specific is exposed as a constructor
parameter. Metadata is parsed best-effort from the embedded (ToUnicode-mapped)
text layer.

Pipeline
--------
1. Decompress the page content stream (pikepdf / zlib).
2. Regex out every `m / l / S` line segment.
3. Reconnect segments that share an endpoint into polylines. Grid lines are
   isolated 2-point strokes and fall out; real traces are long polylines.
4. Detect units-per-mm from the heavy grid-line spacing (5 mm apart) and
   convert: mV = (baseline - v_coord) / (units_per_mm * gain_mm_per_mV),
            s  = (t_coord  - t0)     / (units_per_mm * speed_mm_per_s).
5. Cluster traces into rows by baseline, order columns by time, name leads
   from the layout, and resample to a uniform grid.

Dependencies: pikepdf, numpy. Optional: pypdfium2 (metadata text),
matplotlib (plotting).

Usage
-----
    from ecg_pdf import extract_ecg
    ecg = extract_ecg("ECG01.pdf")
    ecg.rows_to_csv("rows.csv")        # the 4 printed rows, 100 Hz
    ecg.leads_to_csv("leads.csv")      # the 12 leads + rhythm
    ecg.meta_to_json("meta.json")
    t, mv = ecg.leads["V5"]            # numpy arrays

    # or from the command line:
    #   python ecg_pdf.py ECG01.pdf -o out/ --plot
"""
from __future__ import annotations

import json
import re
import zlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["ECGExtractor", "ECGResult", "extract_ecg", "parse_metadata"]

# Default 4x3 + rhythm layout. rows[i] holds the leads printed left->right in
# printed row i; the final row is the continuous rhythm strip (one lead name).
DEFAULT_LAYOUT: List[List[str]] = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
    ["II_rhythm"],
]

# Whitespace between tokens is matched loosely (\s+) so that CRLF line endings
# and minor spacing variations between carts still tokenize.
_SEG_RE = re.compile(
    r"(-?[0-9.]+)\s+(-?[0-9.]+)\s+m\s+"
    r"(-?[0-9.]+)\s+(-?[0-9.]+)\s+l\s+S"
)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class ECGResult:
    """Container for an extracted ECG.

    Attributes
    ----------
    leads : dict[str, (t, mv)]
        Per-lead arrays. Short leads are ``column_seconds`` long; the rhythm
        lead spans the full strip.
    rows : dict[str, (t, mv)]
        The printed rows as continuous series over the full strip duration.
        Keys are ``"row1".."rowN"``; the lead identity changes between columns.
    meta : dict
        Best-effort metadata (patient, measurements, acquisition, diagnosis).
    fs : float
        Sample rate (Hz) of the resampled arrays.
    units_per_mm, units_per_mv, units_per_s : float
        Calibration actually used.
    layout : list[list[str]]
        The lead layout that was assumed.
    n_columns : int
        Number of time columns in the short-lead rows.
    """

    leads: Dict[str, Tuple[np.ndarray, np.ndarray]]
    rows: Dict[str, Tuple[np.ndarray, np.ndarray]]
    meta: dict
    fs: float
    units_per_mm: float
    units_per_mv: float
    units_per_s: float
    layout: List[List[str]]
    n_columns: int
    column_seconds: float
    strip_seconds: float

    # ---- exporters ------------------------------------------------------- #
    def rows_to_csv(self, path: str) -> str:
        keys = list(self.rows)
        t = self.rows[keys[0]][0]
        cols = [self.rows[k][1] for k in keys]
        header = ["time_s"] + [f"{k}_mV" for k in keys]
        self._write_csv(path, header, t, cols)
        return path

    def leads_to_csv(self, path: str) -> str:
        # one shared time column at fs; short leads are NaN-padded after their end
        n = max(len(v[0]) for v in self.leads.values())
        t = np.arange(n) / self.fs
        order = [l for row in self.layout for l in row]
        cols = []
        for name in order:
            _, v = self.leads[name]
            padded = np.full(n, np.nan)
            padded[: len(v)] = v
            cols.append(padded)
        header = ["time_s"] + [f"{name}_mV" for name in order]
        self._write_csv(path, header, t, cols, na="")
        return path

    def meta_to_json(self, path: str) -> str:
        payload = dict(self.meta)
        payload["acquisition"] = {
            **payload.get("acquisition", {}),
            "sample_rate_Hz": self.fs,
            "units_per_mm": round(self.units_per_mm, 4),
            "units_per_mV": round(self.units_per_mv, 4),
            "units_per_s": round(self.units_per_s, 4),
        }
        payload["layout"] = self.layout
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path

    def plot(self, path: Optional[str] = None):
        """Quick standard-layout sanity plot. Requires matplotlib."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        short_rows = self.layout[:-1]
        rhythm_name = self.layout[-1][0]
        nrow = len(short_rows)
        fig, ax = plt.subplots(nrow + 1, self.n_columns,
                               figsize=(4 * self.n_columns, 2.2 * (nrow + 1)))
        for r, row in enumerate(short_rows):
            for c, name in enumerate(row):
                t, v = self.leads[name]
                ax[r, c].plot(t, v, "k", lw=0.8)
                ax[r, c].set_title(name, fontsize=9)
                ax[r, c].grid(color="salmon", lw=0.3, alpha=0.5)
        for c in range(self.n_columns):
            ax[nrow, c].axis("off")
        axb = fig.add_subplot(nrow + 1, 1, nrow + 1)
        t, v = self.leads[rhythm_name]
        axb.plot(t, v, "k", lw=0.7)
        axb.set_title(f"{rhythm_name} (rhythm)", fontsize=9)
        axb.grid(color="salmon", lw=0.3, alpha=0.5)
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=110)
            return path
        return fig

    @staticmethod
    def _write_csv(path, header, t, cols, na="nan"):
        import csv
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for i in range(len(t)):
                row = [f"{t[i]:.4f}"]
                for c in cols:
                    val = c[i]
                    row.append(na if (isinstance(val, float) and np.isnan(val))
                               else f"{val:.4f}")
                w.writerow(row)


# --------------------------------------------------------------------------- #
# Extractor
# --------------------------------------------------------------------------- #
class ECGExtractor:
    def __init__(
        self,
        gain_mm_per_mv: float = 10.0,
        speed_mm_per_s: float = 25.0,
        layout: Sequence[Sequence[str]] = DEFAULT_LAYOUT,
        units_per_mm: Optional[float] = None,   # None -> auto-detect from grid
        fs: float = 100.0,
        min_trace_points: int = 50,
        trim_connector_mv: float = 0.5,
        polarity: str = "auto",                 # "auto" | "pos" | "neg"
        connect_tol: float = 0.5,
    ):
        self.gain = gain_mm_per_mv
        self.speed = speed_mm_per_s
        self.layout = [list(r) for r in layout]
        self.units_per_mm = units_per_mm
        self.fs = fs
        self.min_trace_points = min_trace_points
        self.trim_connector_mv = trim_connector_mv
        self.polarity = polarity
        self.connect_tol = connect_tol

    # -- public ----------------------------------------------------------- #
    def extract(self, pdf_path: str) -> ECGResult:
        content, rotate = self._content_stream(pdf_path)
        segs = self._segments(content)
        if not segs:
            raise ValueError("No vector path segments found — this PDF may be "
                             "a raster scan, not a vector ECG.")
        polys = self._polylines(segs)
        traces = [p for p in polys if len(p) >= self.min_trace_points]
        if not traces:
            raise ValueError("No trace polylines found; try lowering "
                             "min_trace_points.")

        upm = self.units_per_mm or self._grid_units_per_mm(segs)
        upmv = upm * self.gain
        ups = upm * self.speed

        # Axis roles: for /Rotate 90|270 the long sweep is the y axis.
        time_is_y = rotate in (90, 270)
        ti, vi = (1, 0) if time_is_y else (0, 1)   # column indices into points

        rows_pts = self._cluster_rows(traces, vi, len(self.layout))
        t0 = min(p[:, ti].min() for p in traces)
        sign = self._polarity_sign(rows_pts, ti, vi, t0, ups, upmv)

        leads: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        rows: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        n_short_cols = max(len(r) for r in self.layout[:-1]) if len(self.layout) > 1 else 1
        strip_secs = self._strip_seconds(traces, ti, t0, ups)
        col_secs = strip_secs / n_short_cols
        rhythm_lead_names: set = set()

        for ri, names in enumerate(self.layout):
            plist = sorted(rows_pts[ri], key=lambda p: p[:, ti].min())
            base = self._baseline(np.vstack(plist)[:, vi])
            is_rhythm = (len(names) == 1)

            row_t, row_v = [], []
            for ci, p in enumerate(plist):
                t, v = self._to_series(p, ti, vi, t0, base, ups, upmv, sign)
                if not is_rhythm and ci > 0:
                    t, v = self._trim_connector(t, v)
                row_t.append(t)
                row_v.append(v)
                name = names[ci] if ci < len(names) else f"{names[0]}_{ci}"
                # native lead time re-zeroed to its column
                leads[name] = (t - (t[0] if len(t) else 0.0), v)
                if is_rhythm:
                    rhythm_lead_names.add(name)

            rt = np.concatenate(row_t)
            rv = np.concatenate(row_v)
            order = np.argsort(rt)
            rows[f"row{ri+1}"] = self._resample(rt[order], rv[order], self.fs)

        # resample leads onto uniform grids: rhythm leads span the full strip,
        # short leads span a single column window.
        for name, (t, v) in list(leads.items()):
            dur = strip_secs if name in rhythm_lead_names else col_secs
            leads[name] = self._resample_fixed(t, v, self.fs, dur)

        meta = parse_metadata(pdf_path) or {}
        return ECGResult(
            leads=leads, rows=rows, meta=meta, fs=self.fs,
            units_per_mm=upm, units_per_mv=upmv, units_per_s=ups,
            layout=self.layout, n_columns=n_short_cols,
            column_seconds=col_secs,
            strip_seconds=strip_secs,
        )

    # -- internals -------------------------------------------------------- #
    @staticmethod
    def _content_stream(pdf_path: str) -> Tuple[str, int]:
        try:
            import pikepdf
            with pikepdf.open(pdf_path) as pdf:
                page = pdf.pages[0]
                rotate = int(page.get("/Rotate", 0)) % 360
                data = page.Contents.read_bytes()
            return data.decode("latin1"), rotate
        except ImportError:
            # minimal fallback: inflate the first flate stream by hand
            with open(pdf_path, "rb") as fh:
                raw = fh.read()
            rotate = 0
            m = re.search(rb"/Rotate\s+(\d+)", raw)
            if m:
                rotate = int(m.group(1)) % 360
            best = ""
            for m in re.finditer(rb"stream\r?\n", raw):
                start = m.end()
                end = raw.find(b"endstream", start)
                try:
                    dec = zlib.decompress(raw[start:end]).decode("latin1")
                except Exception:
                    continue
                if dec.count(" m\n") > best.count(" m\n"):
                    best = dec
            return best, rotate

    @staticmethod
    def _segments(content: str) -> List[Tuple[float, float, float, float]]:
        return [(float(a), float(b), float(c), float(d))
                for a, b, c, d in _SEG_RE.findall(content)]

    def _polylines(self, segs) -> List[np.ndarray]:
        polys, cur = [], None
        tol = self.connect_tol
        for x1, y1, x2, y2 in segs:
            if (cur is None or abs(cur[-1][0] - x1) > tol
                    or abs(cur[-1][1] - y1) > tol):
                cur = [(x1, y1), (x2, y2)]
                polys.append(cur)
            else:
                cur.append((x2, y2))
        return [np.asarray(p, float) for p in polys]

    @staticmethod
    def _grid_units_per_mm(segs) -> float:
        """Heavy grid lines sit 5 mm apart. Find the modal spacing of the
        long axis-aligned grid strokes and divide by 5.

        Assumes the full-height vertical strokes are the *heavy* (5 mm) grid
        lines, as on the reference cart. If a cart also draws the light 1 mm
        lines as full-height strokes the detected spacing would be 1 mm and the
        scale 5x too small — pass ``units_per_mm`` explicitly in that case.
        """
        arr = np.asarray(segs, float)
        spans_x = np.abs(arr[:, 2] - arr[:, 0])
        spans_y = np.abs(arr[:, 3] - arr[:, 1])
        # vertical grid lines: dx ~ 0, dy large
        vert = arr[(spans_x < 1) & (spans_y > spans_y.max() * 0.5)]
        coords = np.unique(np.round(vert[:, 0], 1))
        if len(coords) >= 3:
            diffs = np.diff(np.sort(coords))
            diffs = diffs[diffs > 1]
            if len(diffs):
                # modal spacing is more robust than the median to a few missing
                # or doubled grid lines
                vals, counts = np.unique(np.round(diffs, 1), return_counts=True)
                spacing = float(vals[np.argmax(counts)])
                return spacing / 5.0
        raise ValueError("Could not auto-detect grid scale; pass units_per_mm.")

    @staticmethod
    def _cluster_rows(traces, vi, n_rows) -> Dict[int, List[np.ndarray]]:
        """Group traces into n_rows by their baseline (voltage-axis) position."""
        bvals = np.array([ECGExtractor._baseline(p[:, vi]) for p in traces])
        order = np.argsort(bvals)
        sb = bvals[order]
        # split at the (n_rows-1) largest gaps
        gaps = np.diff(sb)
        if n_rows > 1:
            cut = np.sort(np.argsort(gaps)[-(n_rows - 1):])
        else:
            cut = np.array([], int)
        labels = np.zeros(len(sb), int)
        for c in cut:
            labels[c + 1:] += 1
        rows = {i: [] for i in range(n_rows)}
        for idx, lab in zip(order, labels):
            rows[int(lab)].append(traces[idx])
        return rows

    @staticmethod
    def _baseline(coord: np.ndarray) -> float:
        """Isoelectric line = histogram peak of the voltage coordinate
        (robust to QRS deflections, which bias the mean/median)."""
        lo, hi = coord.min(), coord.max()
        if hi - lo < 1e-6:
            return float(coord.mean())
        hist, edges = np.histogram(coord, bins=80)
        k = int(np.argmax(hist))
        return float((edges[k] + edges[k + 1]) / 2)

    def _polarity_sign(self, rows_pts, ti, vi, t0, ups, upmv) -> float:
        if self.polarity == "pos":
            return 1.0
        if self.polarity == "neg":
            return -1.0
        # auto: the rhythm row's dominant deflection (R wave) should be positive
        plist = rows_pts[len(self.layout) - 1]
        p = np.vstack(plist)
        base = self._baseline(p[:, vi])
        dev = (base - p[:, vi]) / upmv          # provisional +mV = baseline-coord
        return 1.0 if abs(dev.max()) >= abs(dev.min()) else -1.0

    @staticmethod
    def _to_series(p, ti, vi, t0, base, ups, upmv, sign):
        order = np.argsort(p[:, ti])
        t = (p[order, ti] - t0) / ups
        v = sign * (base - p[order, vi]) / upmv
        return t, v

    def _trim_connector(self, t, v):
        """Drop the leading inter-lead pen-transition samples (large
        off-baseline values before the trace settles)."""
        k = 0
        while k < len(v) - 1 and abs(v[k]) > self.trim_connector_mv:
            k += 1
        if k:
            t = t[k:] - t[k] + t[0]
            v = v[k:]
        return t, v

    @staticmethod
    def _strip_seconds(traces, ti, t0, ups) -> float:
        tmax = max(p[:, ti].max() for p in traces)
        return (tmax - t0) / ups

    def _resample(self, t, v, fs):
        if len(t) == 0:
            return np.array([]), np.array([])
        grid = np.arange(0, t.max(), 1.0 / fs)
        return grid, np.interp(grid, t, v)

    def _resample_fixed(self, t, v, fs, dur):
        if len(t) == 0:
            return np.array([]), np.array([])
        n = int(round(dur * fs))
        grid = np.linspace(0, dur, n, endpoint=False)
        return grid, np.interp(grid, t, v)


# --------------------------------------------------------------------------- #
# Metadata (best-effort, from the embedded ToUnicode text layer)
# --------------------------------------------------------------------------- #
def parse_metadata(pdf_path: str) -> dict:
    """Pull patient / measurement / acquisition fields from the text layer.

    Returns {} if pypdfium2 is unavailable or no text is present. Fields that
    cannot be located are simply omitted, so callers should use ``.get``.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return {}
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        txt = pdf[0].get_textpage().get_text_range()
    finally:
        pdf.close()

    def find(pat, cast=str, group=1):
        m = re.search(pat, txt)
        if not m:
            return None
        try:
            return cast(m.group(group))
        except Exception:
            return m.group(group)

    meta: dict = {}
    age = find(r"(\d+)\s*岁", int)
    if age is not None:
        meta["age"] = age
    sex = find(r"\n([男女])\b")
    if sex:
        meta["sex"] = {"男": "M", "女": "F"}.get(sex, sex)
    rid = find(r"\n([A-Z]\d{4,})\b")
    if rid:
        meta["record_id"] = rid
    rec = find(r"记录日期：\s*([\d]{4}-[\d]{2}-[\d]{2}[\s\d:]*)")
    if rec:
        meta["recorded"] = rec.strip()
    rep = find(r"报告日期：\s*([\d.]{8,})")
    if rep:
        meta["reported"] = rep.strip()

    # the seven measurements appear as a numeric block right after the units
    # row "bpm ms ms ° ms ms mv"
    mblk = re.search(r"mv\r?\n((?:[\d.]+\r?\n){6}[\d.]+)", txt)
    if mblk:
        nums = [float(x) for x in re.findall(r"[\d.]+", mblk.group(1))][:7]
        keys = ["HR_bpm", "PR_ms", "QRS_ms", "QRS_axis_deg",
                "QT_ms", "QTc_ms", "R_V5_plus_S_V1_mV"]
        meta["measurements"] = dict(zip(keys, nums))

    acq = {}
    f = find(r"滤波：\s*(\d+)\s*Hz", int)
    if f is not None:
        acq["filter_Hz"] = f
    sp = find(r"(\d+)\s*mm/sec", int)
    if sp is not None:
        acq["paper_speed_mm_per_s"] = sp
    gn = find(r"(\d+)\s*mm/mv", int)
    if gn is not None:
        acq["gain_mm_per_mV"] = gn
    if acq:
        meta["acquisition"] = acq

    dx = re.findall(r"\n\s*(\d\.[^\n]+)", txt)
    dx = [d.strip() for d in dx if "段" in d or "律" in d or "ST" in d]
    if dx:
        meta["diagnosis"] = dx
    return meta


def extract_ecg(pdf_path: str, **kwargs) -> ECGResult:
    """Convenience wrapper: ``ECGExtractor(**kwargs).extract(pdf_path)``."""
    return ECGExtractor(**kwargs).extract(pdf_path)


# The command-line interface lives in :mod:`spike.cli`, which routes both vector
# PDFs and raster images (and rasterizes scanned/image-only PDFs).
