"""spike — extract calibrated waveform time series from ECG PDFs and images."""
from .ecg_pdf import (
    ECGExtractor,
    ECGResult,
    extract_ecg,
    parse_metadata,
    DEFAULT_LAYOUT,
)
from .ecg_raster import (
    RasterECGExtractor,
    extract_ecg_image,
)
from .cli import main

__version__ = "0.1.0"
__all__ = [
    "ECGExtractor",
    "ECGResult",
    "extract_ecg",
    "parse_metadata",
    "DEFAULT_LAYOUT",
    "RasterECGExtractor",
    "extract_ecg_image",
    "main",
    "__version__",
]
