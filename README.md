# spike

**Extract calibrated waveform time series (and metadata) from *vector* ECG report PDFs.**

Many hospital ECG carts emit PDF reports where each waveform is drawn as a
sequence of PostScript-style path operators (`x y m` / `x y l` / `S` —
moveto / lineto / stroke) rather than as a flattened raster image. When that's
the case, the underlying samples are recoverable **losslessly** — no
pixel-tracing, no OCR of the grid. `spike` decompresses the page content
stream, reconnects the stroke segments into per-lead polylines, calibrates them
against the printed grid, and hands you clean `(time, millivolt)` arrays.

It was built and validated against a 12-lead report (standard Mason–Likar 4×3
layout plus one rhythm strip, 25 mm/s, 10 mm/mV, `/Rotate 90`). The waveform
path auto-detects the grid scale, lead baselines, time axis and signal
polarity, so it should travel to sibling formats; anything format-specific is
exposed as a parameter.

---

## Install

```bash
pip install .                 # core: numpy + pikepdf
pip install ".[all]"          # + pypdfium2 (metadata) + matplotlib (plots)
pip install ".[meta,plot]"    # pick extras individually
```

Requires Python ≥ 3.12.

| Feature | Needs |
| --- | --- |
| Waveform extraction (core) | `numpy`, `pikepdf` |
| Metadata parsing (`parse_metadata`, `ECGResult.meta`) | `pypdfium2` (extra `meta`) |
| `ECGResult.plot(...)` | `matplotlib` (extra `plot`) |

Metadata and plotting degrade gracefully: without `pypdfium2` you simply get
`meta = {}`; `.plot()` is only called on demand.

---

## Quickstart

### Python API

```python
from spike import extract_ecg

ecg = extract_ecg("ECG01.pdf")          # -> ECGResult

t, mv = ecg.leads["V5"]                 # numpy arrays: seconds, millivolts
t, mv = ecg.rows["row1"]                # a full printed row (continuous, 10 s)

ecg.rows_to_csv("rows.csv")             # 4 printed rows @ fs
ecg.leads_to_csv("leads.csv")           # 12 leads + rhythm strip
ecg.meta_to_json("meta.json")           # patient + measurements + calibration
ecg.plot("check.png")                   # standard-layout sanity figure

print(ecg.fs, ecg.units_per_mm, ecg.strip_seconds)
```

### Command line

Installing the package provides a `spike` console script:

```bash
spike ECG01.pdf -o out/ --plot
```

Writes `out/ECG01_rows.csv`, `out/ECG01_leads.csv`, `out/ECG01_meta.json`
(and `_plot.png` with `--plot`).

```
spike PDF [-o OUTDIR] [--gain 10] [--speed 25] [--fs 100]
          [--units-per-mm FLOAT] [--polarity auto|pos|neg] [--plot]
```

You can also run the module directly without installing:

```bash
python -m spike.ecg_pdf ECG01.pdf -o out/ --plot
```

---

## Output

* **`leads`** — the individual leads. Short leads carry their own column window
  (default 2.5 s); the rhythm lead spans the full strip. Keyed by lead name
  (`"I"`, `"aVR"`, …, `"V6"`, `"II_rhythm"`).
* **`rows`** — the printed rows as continuous series over the whole strip
  (`"row1".."rowN"`). The lead *identity* changes between columns within a row.
* **`meta`** — best-effort patient/measurement/diagnosis fields plus the
  calibration actually used.

Voltages are in **mV**, time in **seconds**, resampled to a uniform rate
(`fs`, default **100 Hz** → 250 samples per 2.5 s column).

CSV column shape:

```
# rows.csv
time_s, row1_mV, row2_mV, row3_mV, row4_mV

# leads.csv  (short leads end early -> trailing blanks)
time_s, I_mV, aVR_mV, V1_mV, V4_mV, II_mV, ..., V6_mV, II_rhythm_mV
```

---

## How it works

1. **Decompress** the first page's content stream (`pikepdf`; a pure-`zlib`
   fallback exists if `pikepdf` is missing) and read `/Rotate`.
2. **Tokenize** every `m / l / S` line segment with a regex.
3. **Reconnect** segments that share an endpoint into polylines. Grid lines are
   isolated 2-point strokes and fall out; real traces become long polylines
   (filtered by `min_trace_points`).
4. **Calibrate.** The heavy grid lines sit 5 mm apart, giving `units_per_mm`
   automatically (≈ 23.6 in the reference file). Then
   `mV = (baseline − v) / (units_per_mm · gain_mm_per_mV)` and
   `s  = (t − t₀) / (units_per_mm · speed_mm_per_s)`.
5. **Assemble.** Cluster traces into rows by baseline (a histogram-peak
   estimate of the isoelectric line, robust to QRS deflection), order the
   columns by time, name leads from the layout, trim the inter-lead pen
   transitions, and resample to `fs`.

Axis roles follow `/Rotate`: for `90`/`270` the long sweep is the *y* axis
(time), the short axis is voltage. Polarity is inferred from the rhythm lead's
dominant R wave unless you force it.

---

## Configuration

`extract_ecg(path, **kwargs)` forwards to `ECGExtractor(...)`:

| Param | Default | Purpose |
| --- | --- | --- |
| `gain_mm_per_mv` | `10.0` | Printed gain. |
| `speed_mm_per_s` | `25.0` | Printed paper speed. |
| `layout` | 4×3 + rhythm | Lead grid; last row is the single rhythm lead. |
| `units_per_mm` | `None` | Override the auto grid-scale detection. |
| `fs` | `100.0` | Output sample rate (Hz). |
| `min_trace_points` | `50` | Min polyline length to count as a trace. |
| `trim_connector_mv` | `0.5` | Drop leading off-baseline pen-transition samples. |
| `polarity` | `"auto"` | `"auto"` / `"pos"` / `"neg"`. |
| `connect_tol` | `0.5` | Endpoint-match tolerance when joining segments. |

To change the lead layout, pass your own list of rows (last row = the single
rhythm-strip lead):

```python
layout = [["I","aVR","V1","V4"], ["II","aVL","V2","V5"],
          ["III","aVF","V3","V6"], ["II_rhythm"]]
ecg = extract_ecg("file.pdf", layout=layout)
```

---

## Limitations / honest caveats

* **Vector PDFs only.** A scanned/raster ECG has no path segments to recover;
  `extract` raises a clear error in that case.
* Validated on one Ghostscript-produced Mason–Likar format. The two things most
  likely to need tweaking for a different cart are the `m l S` segment pattern
  and the `layout` assumption — both are adjustable.
* **Metadata parsing is best-effort** and tuned to the reference report's label
  text (Chinese clinical labels). Missing fields are omitted rather than
  guessed; the authoritative calibration is always recorded from the values
  actually used.
* The 12-lead split assumes the columns are consecutive 2.5 s windows of one
  recording — the standard convention, but not something the PDF asserts.
* `spike` is a data-extraction tool, **not** a diagnostic device. Verify against
  the original report before any clinical or research use.

---

## License

MIT.
