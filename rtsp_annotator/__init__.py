"""Low-latency YOLO annotation pipeline for RTSP streams."""

from .config import Settings
from .pipeline import run_pipeline

__all__ = ["Settings", "run_pipeline"]
__version__ = "0.1.0"
