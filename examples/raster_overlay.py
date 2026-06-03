"""Visualize the raster digitizer on an ECG image.

Overlays the tracked centrelines on the source scan and plots the recovered
12-lead figure, so you can eyeball whether grid scale, baselines, and the trace
follower are right.

    pip install ".[image,plot]"
    python examples/raster_overlay.py path/to/ecg.png -o out/

Writes ``<name>_overlay.png`` (tracked lines on the image) and
``<name>_leads.png`` (standard layout) into the output directory.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from spike.ecg_raster import RasterECGExtractor


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image")
    ap.add_argument("-o", "--outdir", default=".")
    ap.add_argument("--px-per-mm", type=float, default=None)
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    ext = RasterECGExtractor(px_per_mm=args.px_per_mm)
    ink, dark = ext._load_channels(args.image)
    box = ext._bounding_box(ink)
    x0, y0, x1, y1 = box
    pmm = args.px_per_mm or ext._grid_px_per_mm(ink, box)
    mask = ext._trace_mask(dark, box)
    baselines, spacing = ext._baselines(mask, box, ext.nrows)
    win = int(round(spacing * ext.window_frac))
    xfull = np.arange(x0, x1)

    fig, ax = plt.subplots(figsize=(18, 9))
    ax.imshow(np.asarray(Image.open(args.image).convert("RGB")))
    for b in baselines:
        ax.plot(xfull, ext._track_row(mask, b, win, xfull, y0, y1), "b", lw=0.7)
    ax.set_title(f"tracked centrelines  (px/mm={pmm:.2f})")
    ax.axis("off")
    base = os.path.join(args.outdir,
                        os.path.splitext(os.path.basename(args.image))[0])
    fig.tight_layout()
    fig.savefig(base + "_overlay.png", dpi=110)

    ecg = ext.extract(args.image)
    ecg.plot(base + "_leads.png")
    print(f"px/mm={ecg.units_per_mm:.2f}  strip={ecg.strip_seconds:.2f}s  "
          f"leads={list(ecg.leads)}")
    print(f"wrote {base}_overlay.png, {base}_leads.png")


if __name__ == "__main__":
    main()
