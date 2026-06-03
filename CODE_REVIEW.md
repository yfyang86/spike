# Code Review — `spike` (ecg_pdf)

Reviewed on branch `claude/gracious-hopper-qBEs8`. Scope: the full source tree
(`src/spike/ecg_pdf.py`, `__init__.py`, `tests/`, `pyproject.toml`, `README.md`,
`LICENSE`). The 4 smoke tests pass (`PYTHONPATH=src python -m pytest`).

Overall this is clean, well-documented code with honest caveats. The findings
below are ordered by severity; each item I claim as a bug was reproduced.

## High — correctness

### H1. `test_raster_pdf_raises` is a false-positive test
`tests/test_smoke.py:24`

```python
with pytest.raises(ValueError):
    ext._polylines([])      # returns []  — does NOT raise
    raise ValueError("no segments")
```

`_polylines([])` returns `[]` and never raises (verified). The test only passes
because of the *unconditional* `raise ValueError` on the next line, so it
asserts nothing about the code. The behaviour it claims to cover — a raster PDF
with no segments raising a clear error — actually lives in `extract()`
(`if not segs:` / `if not traces:` at ecg_pdf.py:223,228). Recommend driving
that real path, e.g. assert `extract()` (or `_segments("")` → `[]` → the guard)
raises, and drop the manual `raise`.

### H2. Grid auto-detection mis-calibrates when light gridlines are full-height
`ecg_pdf.py:332` (`_grid_units_per_mm`)

The selector `vert = arr[(spans_x < 1) & (spans_y > spans_y.max()*0.5)]` keeps
*every* tall vertical stroke, then takes the **median** of all neighbour gaps.
If the cart draws the light 1 mm gridlines as full-height strokes (not just the
heavy 5 mm ones), the modal gap is 1 mm and the result is 5× too small —
reproduced:

```
heavy only:   23.6      # correct
heavy+light:   4.72     # 5x wrong -> every voltage & time off by 5x, silently
```

It happens to work on the reference file, but the failure is silent and
corrupts all output. Safer: take the *dominant large* spacing (e.g. cluster the
diffs and pick the modal value among the larger cluster), or filter by line
width / pick the heaviest strokes, and sanity-check the result against the
`gain`/`speed` priors before using it.

## Medium

### M1. License is declared three different ways
- `LICENSE` file: **Apache License 2.0** (full text)
- `README.md:182`: "Apache 2"
- `pyproject.toml:11,16`: `license = { text = "MIT" }` + `License :: OSI Approved :: MIT License`

This is legally meaningful metadata; PyPI and downstream tooling will read the
`pyproject` value. Pick one license and make all three agree.

### M2. Segment regex is brittle to line endings / spacing
`ecg_pdf.py:66` (`_SEG_RE`)

The pattern hardcodes a single `\n` between operators (`... m\n ... l\nS`).
Content streams that use `\r\n`, extra whitespace, or interleaved operators
won't match, and `extract()` then raises "No vector path segments found" on an
otherwise-valid vector PDF. Note the fallback stream finder already uses
`\r?\n` (ecg_pdf.py:303) — the segment regex should be similarly tolerant
(`\s+` / `\r?\n`).

### M3. PDF handles are never closed (resource leak)
- `ecg_pdf.py:290` `pikepdf.open(pdf_path)` — not closed
- `ecg_pdf.py:297` `open(pdf_path, "rb").read()` in the fallback — not closed
- `ecg_pdf.py:441` `pdfium.PdfDocument(pdf_path)` — not closed

Use context managers (`with pikepdf.open(...) as pdf:`) / explicit `.close()`.
Matters for callers that process many files in a loop.

## Low — cleanup / polish

- **L1.** `_cluster_rows` (ecg_pdf.py:360-363): `groups` and `start` are dead
  variables, and `gi` in `for gi, c in enumerate(cut)` is unused — use
  `for c in cut`.
- **L2.** `extract()` calls `self._strip_seconds(traces, ti, t0, ups)` three+
  times (col_secs, the per-lead resample loop, and the `ECGResult` ctor).
  Compute once and reuse.
- **L3.** `_trim_connector` (ecg_pdf.py:407): `return t[:], v[k:] if k else v`
  leans on operator precedence and makes a needless `t[:]` copy. Spell it out
  for readability.
- **L4.** Rhythm resampling keys off `name == self.layout[-1][0]`
  (ecg_pdf.py:273). If the rhythm row ever splits into >1 polyline, the extras
  are named `II_rhythm_1`… and get resampled to `col_secs` instead of the full
  strip duration. Edge case, but worth a guard.
- **L5.** Placeholder URLs `https://example.com/spike` in `pyproject.toml:42-43`.
- **L6.** Tests require `PYTHONPATH=src` (or an editable install) to import
  `spike`. Consider adding pytest config so `pytest` works from a clean checkout:
  ```toml
  [tool.pytest.ini_options]
  pythonpath = ["src"]
  ```

## Resolution (applied on this branch)

- **H1** — fixed: `test_raster_pdf_raises` now monkeypatches `_content_stream`
  to return segment-free content and asserts `extract()` raises; added
  `test_no_segments_returns_empty`.
- **H2** — hardened: detection now uses the *modal* gap (robust to missing /
  doubled lines) and the heavy-line assumption is documented, with guidance to
  pass `units_per_mm` when light gridlines are full-height. A complete fix needs
  stroke-width information that the segment endpoints don't carry.
- **M1** — fixed: now Apache-2.0 in `pyproject.toml` (text + classifier) and
  README, matching the `LICENSE` file.
- **M2** — fixed: `_SEG_RE` now tolerates CRLF / loose whitespace; added
  `test_segments_tolerate_crlf`.
- **M3** — fixed: pikepdf, the fallback file handle, and pypdfium2 documents are
  now closed (context managers / `try/finally`).
- **L1–L6** — fixed: dead vars removed; `strip_seconds` computed once; clearer
  `_trim_connector`; rhythm leads resampled by membership (not exact name);
  real repo URLs; `[tool.pytest.ini_options] pythonpath = ["src"]` so `pytest`
  runs from a clean checkout.

## What's good
- Clear module docstring and README that honestly state assumptions and the
  "not a diagnostic device" caveat.
- Graceful optional-dependency degradation (pikepdf fallback, `meta={}` without
  pypdfium2, on-demand matplotlib import).
- Robust baseline estimate via histogram peak rather than mean/median.
- Calibration actually used is always recorded in the output metadata.

---

# Code Review 2 — raster digitizer + unified CLI

Scope: `src/spike/ecg_raster.py`, the new `src/spike/cli.py`, `__init__.py`,
`pyproject.toml`, `tests/`. Seven finder angles + verification; all findings
below were reproduced and **fixed** in the same change, with tests added.

### Fixed
- **`--auto` silently overrode an explicit `--type`** (`cli.main`). `spike
  --type vector --auto x.pdf` discarded the forced pipeline. Now a conflict is
  an explicit error (`test_cli_auto_conflicts_with_explicit_type`).
- **`_pdf_is_vector` mis-routed undecodable vector PDFs.** A bare `except`
  returned "scanned", so an encrypted / exotic-filter *vector* PDF was sent to
  the lossy raster path. Now it assumes vector on decode failure, so the
  lossless pipeline runs and raises a clear error if the file is truly unusable.
- **`_count_path_ops` over-matched.** It counted any whitespace-delimited
  `m/l/c/v/y`, so stray letters inside drawn text could mark a scanned page as
  vector. Now it requires a numeric operand before the operator
  (`test_count_path_ops_ignores_lone_letters_in_text`).
- **Dead / divergent vector CLI.** The entry point moved to `spike.cli:main`,
  leaving `ecg_pdf._main` as a second, drifting CLI; removed it and updated the
  README (it had advertised `python -m spike.ecg_pdf`).
- **Missing baselines silently dropped leads** (`_baselines`). Fewer projection
  peaks than printed rows truncated the lead set; now falls back to even
  spacing so every layout row gets a baseline.
- **`_load_channels` ndarray range.** A `[0,1]` float image (a common numpy
  convention) made every `< 232` threshold treat the frame as solid ink; now
  scaled to 0–255.
- **pypdfium2 handle leak** (`_render_pdf_page`): the page and bitmap are now
  closed alongside the document.
- **Smaller items:** `--auto` install hint named only `meta` (raster also needs
  `image`) → now `spike[image,meta]`; the "wrote …" line dropped the `-o` dir
  (printed basenames) → now full paths; removed a dead no-op branch and the
  unused `RASTER_EXTS` in `resolve_kind`; dropped CLI-internal `resolve_kind`
  from the package's public `__all__`.

### Known limitation (not changed)
- The rhythm lead is resampled to the full `strip_seconds` after the leading
  calibration pulse is trimmed, leaving ~0.2 s of flat padding at its tail. Low
  severity and pre-existing; noted for a future pass.
