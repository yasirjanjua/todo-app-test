"""Frame-stability detection: waiting out a move's slide/merge animation.

This is the single mechanism the engine spec calls out as eliminating the majority of
misread-board bugs. A move's on-screen animation (100-150 ms typically) is not something the
app can time precisely -- it varies by game, by machine, and by whether other tiles are also
mid-merge -- so instead of guessing a fixed delay, this polls frames and accepts the board
only once consecutive frames stop changing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StabilityConfig:
    """Tunable knobs for stability polling. ``settle_frames`` and ``timeout_ms`` are measured
    and stored per-profile during calibration (see ``app/profile.py``)."""

    settle_frames: int = 2
    poll_interval_ms: float = 16.0
    pixel_diff_threshold: float = 2.0  # mean absolute per-pixel difference, 0-255 scale
    timeout_ms: float = 1500.0


@dataclass(frozen=True)
class StabilityResult:
    frame: np.ndarray
    settled: bool
    elapsed_ms: float
    polls: int


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        # A shape mismatch (e.g. the window resized mid-poll) can never be "stable"; treat it
        # as maximally different so the caller's timeout/re-acquire logic takes over.
        return 255.0
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def wait_for_stable(grab: Callable[[], np.ndarray], config: StabilityConfig = StabilityConfig()) -> StabilityResult:
    """Poll ``grab`` until the board stops changing, or until ``config.timeout_ms`` elapses.

    On timeout, the last-captured frame is returned anyway (``settled=False``) with a logged
    warning, per the engine spec's "include a timeout that logs a warning and forces a read"
    requirement -- a stuck animation must never hang the play loop.
    """
    start = time.monotonic()
    previous = grab()
    consecutive_stable = 0
    polls = 1

    while True:
        elapsed_ms = (time.monotonic() - start) * 1000.0
        if elapsed_ms >= config.timeout_ms:
            logger.warning(
                "Frame stability timed out after %.0fms (%d polls); forcing a read.", elapsed_ms, polls
            )
            return StabilityResult(frame=previous, settled=False, elapsed_ms=elapsed_ms, polls=polls)

        time.sleep(config.poll_interval_ms / 1000.0)
        current = grab()
        polls += 1

        diff = _mean_abs_diff(previous, current)
        if diff <= config.pixel_diff_threshold:
            consecutive_stable += 1
        else:
            consecutive_stable = 0

        previous = current
        if consecutive_stable >= config.settle_frames:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            return StabilityResult(frame=current, settled=True, elapsed_ms=elapsed_ms, polls=polls)


def measure_settle_time_ms(grab: Callable[[], np.ndarray], config: StabilityConfig = StabilityConfig()) -> float:
    """Run one stability wait and return how long settling took, for calibration storage."""
    result = wait_for_stable(grab, config)
    return result.elapsed_ms
