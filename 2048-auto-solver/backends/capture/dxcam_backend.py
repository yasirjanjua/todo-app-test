"""Optional, faster Windows-only capture backend built on ``dxcam`` (Desktop Duplication API).

Selected automatically by :func:`backends.capture.factory.create_capture_backend` when
running on Windows and ``dxcam`` is importable; falls back to :class:`MssCaptureBackend`
everywhere else. Never required -- ``mss`` alone satisfies every platform target.
"""

from __future__ import annotations

import logging
import sys
import threading

import numpy as np

from backends.capture.base import CaptureRegion, MonitorInfo

logger = logging.getLogger(__name__)


class DxcamCaptureBackend:
    """:class:`backends.capture.base.CaptureBackend` implementation using ``dxcam``."""

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("DxcamCaptureBackend is only available on Windows")
        import dxcam  # local import: optional dependency, Windows-only

        self._dxcam = dxcam
        self._lock = threading.Lock()
        self._camera = dxcam.create()
        if self._camera is None:
            raise RuntimeError("dxcam.create() returned None (no compatible GPU/adapter found)")

    @staticmethod
    def is_available() -> bool:
        if sys.platform != "win32":
            return False
        try:
            import dxcam  # noqa: F401
        except ImportError:
            return False
        return True

    def list_monitors(self) -> list[MonitorInfo]:
        # dxcam doesn't expose per-monitor enumeration as richly as mss; delegate to mss for
        # the monitor list (cheap, only called during calibration) and keep dxcam for the hot
        # capture path.
        from backends.capture.mss_backend import MssCaptureBackend

        return MssCaptureBackend().list_monitors()

    def get_scale_factor(self, monitor_index: int = 0) -> float:
        from backends.capture.mss_backend import MssCaptureBackend

        return MssCaptureBackend().get_scale_factor(monitor_index)

    def grab(self, region: CaptureRegion) -> np.ndarray:
        left, top = region.left, region.top
        right, bottom = left + region.width, top + region.height
        with self._lock:
            frame = self._camera.grab(region=(left, top, right, bottom))
        if frame is None:
            # dxcam returns None when no new frame is ready yet (it's a duplication-API
            # delta stream); a single retry covers the overwhelming majority of cases.
            with self._lock:
                frame = self._camera.grab(region=(left, top, right, bottom))
        if frame is None:
            raise RuntimeError("dxcam failed to produce a frame for the requested region")
        # dxcam returns RGB; convert to the backend contract's BGR order.
        return np.ascontiguousarray(frame[:, :, ::-1])

    def close(self) -> None:
        with self._lock:
            self._camera.stop()
