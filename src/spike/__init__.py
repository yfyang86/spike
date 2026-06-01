"""spike — extract calibrated waveform time series from vector ECG PDFs."""
from .ecg_pdf import (
    ECGExtractor,
    ECGResult,
    extract_ecg,
    parse_metadata,
    DEFAULT_LAYOUT,
)

__version__ = "0.1.0"
__all__ = [
    "ECGExtractor",
    "ECGResult",
    "extract_ecg",
    "parse_metadata",
    "DEFAULT_LAYOUT",
    "__version__",
]
