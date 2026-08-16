"""Screen-capture backends, selected at runtime behind a common protocol."""

from backends.capture.base import CaptureBackend, CaptureRegion, MonitorInfo
from backends.capture.factory import create_capture_backend

__all__ = ["CaptureBackend", "CaptureRegion", "MonitorInfo", "create_capture_backend"]
