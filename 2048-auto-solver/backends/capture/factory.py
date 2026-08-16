"""Runtime selection of the best available :class:`CaptureBackend`."""

from __future__ import annotations

import logging
import sys

from backends.capture.base import CaptureBackend

logger = logging.getLogger(__name__)


def create_capture_backend(prefer_dxcam: bool = True) -> CaptureBackend:
    """Return the best capture backend for the current OS.

    On Windows, ``dxcam`` (Desktop Duplication API) is tried first when ``prefer_dxcam`` is
    set and the package is installed, since it captures with lower latency than a GDI-based
    grab. Any failure to construct it -- missing package, no compatible GPU, headless CI
    runner -- falls back to ``mss`` silently, since ``mss`` alone is sufficient everywhere.
    """
    if prefer_dxcam and sys.platform == "win32":
        try:
            from backends.capture.dxcam_backend import DxcamCaptureBackend

            if DxcamCaptureBackend.is_available():
                return DxcamCaptureBackend()
        except Exception:  # noqa: BLE001 - any dxcam failure falls back to mss
            logger.info("dxcam capture backend unavailable, falling back to mss", exc_info=True)

    from backends.capture.mss_backend import MssCaptureBackend

    return MssCaptureBackend()
