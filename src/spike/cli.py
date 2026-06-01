"""spike — one command-line entry point for both ECG extraction pipelines.

``spike INPUT`` digitizes a 12-lead ECG to CSV/JSON, routing the input to the
right pipeline:

* a **vector PDF** (waveform stored as path operators) → the lossless
  :mod:`spike.ecg_pdf` pipeline;
* a **raster image** (PNG/JPG/… scan, screenshot, photo) → the approximate
  :mod:`spike.ecg_raster` pipeline;
* a **scanned / image-only PDF** (a page that is just a bitmap, e.g. produced by
  a scanner or Ghostscript) → rendered to a bitmap, then the raster pipeline.

The third case is the point of ``--auto``: such a PDF carries no vector path
operators, so the vector pipeline would find nothing. ``--auto`` (the default)
inspects the PDF, detects that it is image-only, and reroutes it to the raster
pipeline instead of failing.

    spike report.pdf                 # vector PDF -> exact recovery
    spike scan.png                   # raster scan -> approximate recovery
    spike --auto scanned.pdf         # image-only PDF -> rendered + rasterized
    spike --type vector report.pdf   # force a pipeline
"""
from __future__ import annotations

import argparse
import os
import re

# A real vector ECG draws the waveform with thousands of path operators; a
# scanned/image-only PDF has essentially none. This is the cutoff between them.
PDF_VECTOR_MIN_OPS = 200


def _looks_like_pdf(path: str) -> bool:
    """True if the file extension is .pdf or it starts with the %PDF- magic."""
    if os.path.splitext(path)[1].lower() == ".pdf":
        return True
    try:
        with open(path, "rb") as fh:
            return fh.read(5) == b"%PDF-"
    except OSError:
        return False


def _count_path_ops(content: str) -> int:
    """Count PDF path-construction operators (moveto/lineto/curveto) in a content
    stream. Near-zero for an image-only page, in the thousands for a vector ECG.

    Each path operator is preceded by its numeric operands (``x y m``,
    ``x y l``, ``... y3 c``), so we require a number immediately before the
    operator token. That excludes stray ``m``/``l``/``c`` letters that appear
    inside drawn text strings, which would otherwise inflate the count.
    """
    return len(re.findall(r"\d\s+[mlcvy](?=\s|$)", content))


def _pdf_is_vector(path: str) -> bool:
    """Decide whether a PDF stores its waveform as vectors (True) or is just a
    scanned bitmap (False), by counting path operators on the first page."""
    from .ecg_pdf import ECGExtractor
    try:
        content, _ = ECGExtractor._content_stream(path)
    except Exception:
        # Could not decode a content stream (encrypted, exotic filter, …). We
        # don't know — assume vector so the lossless pipeline runs and raises a
        # clear error if the PDF is truly unusable, rather than silently routing
        # a good vector PDF into the lossy raster path.
        return True
    return _count_path_ops(content) >= PDF_VECTOR_MIN_OPS


def resolve_kind(path: str, mode: str) -> str:
    """Map (input, mode) to a concrete pipeline: ``vector``, ``raster``, or
    ``raster-pdf`` (a PDF page that must be rendered before rasterizing).

    ``mode`` is ``auto`` | ``vector`` | ``raster``.
    """
    is_pdf = _looks_like_pdf(path)
    if mode == "vector":
        return "vector"
    if mode == "raster":
        return "raster-pdf" if is_pdf else "raster"
    # auto: a non-PDF is a raster image (the loader surfaces a clear error if it
    # can't open an unknown extension); a PDF is vector unless it is image-only.
    if not is_pdf:
        return "raster"
    return "vector" if _pdf_is_vector(path) else "raster-pdf"


def _render_pdf_page(path: str, dpi: float):
    """Render the first PDF page to an RGB ``ndarray`` for the raster pipeline."""
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # optional dependency
        raise SystemExit(
            "Reading a scanned/image-only PDF needs pypdfium2 to rasterize it and "
            "pillow+scipy for the raster pipeline.\n"
            "    pip install 'spike[image,meta]'"
        ) from exc
    import numpy as np

    pdf = pdfium.PdfDocument(path)
    page = bitmap = None
    try:
        page = pdf[0]
        bitmap = page.render(scale=dpi / 72.0)
        arr = np.asarray(bitmap.to_pil().convert("RGB"))
    finally:
        for obj in (bitmap, page, pdf):
            if obj is not None:
                obj.close()
    return arr


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="spike",
        description="Extract calibrated 12-lead ECG time series from a vector PDF "
                    "or a raster image (scan/screenshot/photo).",
    )
    ap.add_argument("input", help="ECG PDF or image file")
    ap.add_argument("-o", "--outdir", default=".", help="output directory")
    ap.add_argument(
        "--type", choices=["auto", "vector", "raster"], default="auto",
        help="which pipeline to use; 'auto' (default) detects a vector PDF vs a "
             "scanned/image-only PDF vs a raster image",
    )
    ap.add_argument(
        "--auto", action="store_true",
        help="auto-detect vector PDFs, scanned/Ghostscript PDFs and raster images "
             "(this is the default; pass --type to force a specific pipeline)",
    )
    ap.add_argument("--plot", action="store_true", help="also write a sanity plot (needs matplotlib)")

    # shared calibration
    cal = ap.add_argument_group("calibration")
    cal.add_argument("--gain", type=float, default=10.0, help="mm per mV (default 10)")
    cal.add_argument("--speed", type=float, default=25.0, help="mm per second (default 25)")
    cal.add_argument("--fs", type=float, default=100.0, help="output sample rate Hz (default 100)")

    vec = ap.add_argument_group("vector PDF options")
    vec.add_argument("--units-per-mm", type=float, default=None,
                     help="override grid-scale detection")
    vec.add_argument("--polarity", choices=["auto", "pos", "neg"], default="auto")

    ras = ap.add_argument_group("raster / scanned-PDF options")
    ras.add_argument("--px-per-mm", type=float, default=None,
                     help="override grid-scale detection")
    ras.add_argument("--trace-darkness", type=float, default=None,
                     help="local-contrast mask threshold for unevenly lit photos")
    ras.add_argument("--no-suppress-grid", action="store_true",
                     help="keep grid lines in the trace mask")
    ras.add_argument("--dpi", type=float, default=300.0,
                     help="render DPI for scanned/image-only PDFs (default 300)")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if not os.path.exists(args.input):
        raise SystemExit(f"input not found: {args.input}")
    if args.auto and args.type != "auto":
        raise SystemExit(f"--auto conflicts with --type {args.type}; pass only one")

    mode = "auto" if args.auto else args.type
    kind = resolve_kind(args.input, mode)

    if kind == "vector":
        from .ecg_pdf import extract_ecg
        ecg = extract_ecg(
            args.input, gain_mm_per_mv=args.gain, speed_mm_per_s=args.speed,
            fs=args.fs, units_per_mm=args.units_per_mm, polarity=args.polarity,
        )
    else:
        from .ecg_raster import extract_ecg_image
        source = _render_pdf_page(args.input, args.dpi) if kind == "raster-pdf" else args.input
        ecg = extract_ecg_image(
            source, gain_mm_per_mv=args.gain, speed_mm_per_s=args.speed,
            fs=args.fs, px_per_mm=args.px_per_mm, trace_darkness=args.trace_darkness,
            suppress_grid=not args.no_suppress_grid,
        )

    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.join(args.outdir, os.path.splitext(os.path.basename(args.input))[0])
    written = [ecg.rows_to_csv(base + "_rows.csv"),
               ecg.leads_to_csv(base + "_leads.csv"),
               ecg.meta_to_json(base + "_meta.json")]
    if args.plot:
        written.append(ecg.plot(base + "_plot.png"))

    label = {"vector": "vector PDF", "raster": "raster image",
             "raster-pdf": "scanned PDF (rasterized)"}[kind]
    print(f"[{label}]  units/mm={ecg.units_per_mm:.3f}  fs={ecg.fs}Hz  "
          f"strip={ecg.strip_seconds:.2f}s  leads={list(ecg.leads)}")
    print("wrote " + ", ".join(written))


if __name__ == "__main__":
    main()
