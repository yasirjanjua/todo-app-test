"""Universal screen-capture backend built on ``mss``.

``mss`` works unmodified on macOS, Windows, and X11 Linux, which is why it is the default and
only mandatory capture backend (see module docstring in ``backends/capture/base.py`` for why
capture must stay OS-agnostic). It also natively reports monitor geometry in physical pixels,
which is what makes the Retina/HiDPI reconciliation in ``get_scale_factor`` possible.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from backends.capture.base import CaptureRegion, MonitorInfo

logger = logging.getLogger(__name__)


class MssCaptureBackend:
    """:class:`backends.capture.base.CaptureBackend` implementation using ``mss``."""

    def __init__(self) -> None:
        import mss  # local import: keeps this optional dependency out of modules that don't capture

        self._mss_module = mss
        # mss.mss() instances are not thread-safe; the play loop and any UI-triggered
        # calibration capture may run on different threads, so guard with a lock rather than
        # assuming single-threaded use.
        self._lock = threading.Lock()
        self._sct = mss.mss()

    @staticmethod
    def is_available() -> bool:
        try:
            import mss  # noqa: F401
        except ImportError:
            return False
        return True

    def list_monitors(self) -> list[MonitorInfo]:
        with self._lock:
            # mss.monitors[0] is the "all monitors combined" virtual bounding box; real
            # monitors start at index 1.
            monitors = self._sct.monitors[1:]
        result: list[MonitorInfo] = []
        for i, mon in enumerate(monitors):
            result.append(
                MonitorInfo(
                    index=i,
                    left=mon["left"],
                    top=mon["top"],
                    width=mon["width"],
                    height=mon["height"],
                    is_primary=(mon["left"] == 0 and mon["top"] == 0),
                )
            )
        return result

    def get_scale_factor(self, monitor_index: int = 0) -> float:
        """Best-effort physical/logical pixel ratio (2.0 on a typical Retina display).

        ``mss`` always reports physical pixels, but window geometry from the OS's window
        enumeration API is often reported in logical points. Grid math must stay consistent
        (see ``backends/capture/base.py``), so this multiplies logical coordinates up to
        physical ones before they're turned into a :class:`CaptureRegion`.
        """
        import sys

        if sys.platform == "darwin":
            try:
                from AppKit import NSScreen

                screens = NSScreen.screens()
                if monitor_index < len(screens):
                    return float(screens[monitor_index].backingScaleFactor())
            except Exception:  # noqa: BLE001 - pyobjc missing or API unavailable
                logger.debug("Could not read NSScreen backingScaleFactor; assuming 1.0", exc_info=True)
        elif sys.platform == "win32":
            try:
                import ctypes

                ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
                dpi = ctypes.windll.user32.GetDpiForSystem()
                return dpi / 96.0
            except Exception:  # noqa: BLE001 - older Windows without these APIs
                logger.debug("Could not read Windows DPI; assuming 1.0", exc_info=True)
        return 1.0

    def grab(self, region: CaptureRegion) -> np.ndarray:
        monitor = {
            "left": region.left,
            "top": region.top,
            "width": region.width,
            "height": region.height,
        }
        with self._lock:
            shot = self._sct.grab(monitor)
        # mss returns BGRA; drop the alpha channel to match the BGR contract.
        frame = np.array(shot, dtype=np.uint8)[:, :, :3]
        return frame

    def close(self) -> None:
        with self._lock:
            self._sct.close()
