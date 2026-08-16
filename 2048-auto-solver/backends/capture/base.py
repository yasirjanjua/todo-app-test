"""OS-agnostic screen-capture contract.

Assumption: all capture happens in *physical* pixels (the actual sensor/framebuffer
resolution), never OS "points"/logical pixels. On Retina/HiDPI displays those differ by the
display scale factor, and mixing them is the single most common cause of a correct-looking
ROI silently sampling the wrong pixels (Part 5, robustness requirement 7). Callers that need
to reconcile a window's logical geometry (e.g. from a window-enumeration API) with a captured
frame's physical pixels must query :meth:`CaptureBackend.get_scale_factor` and multiply
logical coordinates by it before building a :class:`CaptureRegion`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class CaptureRegion:
    """A capture rectangle in physical-pixel screen coordinates.

    ``left``/``top`` may be negative: on Windows, monitors to the left of or above the
    primary monitor have negative origins (Part 5, robustness requirement 8), and a valid
    region can legitimately start there.
    """

    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"CaptureRegion must have positive size, got {self.width}x{self.height}")


@dataclass(frozen=True)
class MonitorInfo:
    """One physical display, in physical-pixel coordinates."""

    index: int
    left: int
    top: int
    width: int
    height: int
    is_primary: bool


@runtime_checkable
class CaptureBackend(Protocol):
    """Contract every capture implementation (mss, dxcam, ...) must satisfy.

    Implementations must be cheap to call repeatedly in a tight loop: :meth:`grab` is called
    once per stability-poll during the play loop, so it must not allocate a full-screen frame
    when a small ``region`` is requested -- the whole point of passing a region through to the
    OS capture call is that the OS only copies the requested pixels.
    """

    def list_monitors(self) -> list[MonitorInfo]:
        """Return every physical display, in physical-pixel coordinates."""
        ...

    def get_scale_factor(self, monitor_index: int = 0) -> float:
        """Return the physical-to-logical pixel ratio for a monitor (2.0 on typical Retina)."""
        ...

    def grab(self, region: CaptureRegion) -> np.ndarray:
        """Capture ``region`` and return an HxWx3 uint8 array in BGR channel order."""
        ...

    def close(self) -> None:
        """Release any OS handles. Safe to call multiple times."""
        ...
