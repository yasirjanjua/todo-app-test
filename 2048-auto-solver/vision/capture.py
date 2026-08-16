"""ROI-scoped capture and cell-splitting.

Assumption: the capture backend is handed the bounding box directly (see
``backends/capture/base.py``) so the OS only copies the board's ~400x400 pixels, never a full
frame that gets cropped afterward in Python. Per the engine spec, capture is not the
bottleneck (2-5 ms for a cropped grab); a full-screen grab-then-crop would be needlessly
slower and is deliberately avoided everywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backends.capture.base import CaptureBackend, CaptureRegion

BOARD_SIZE = 4


@dataclass(frozen=True)
class Roi:
    """A calibrated board region, in the same physical-pixel space as the capture backend."""

    left: int
    top: int
    width: int
    height: int

    def to_capture_region(self) -> CaptureRegion:
        return CaptureRegion(left=self.left, top=self.top, width=self.width, height=self.height)


def capture_board(backend: CaptureBackend, roi: Roi) -> np.ndarray:
    """Grab exactly the calibrated board region as a BGR frame."""
    return backend.grab(roi.to_capture_region())


def split_cells(frame: np.ndarray, inset_ratio: float = 0.15) -> list[list[np.ndarray]]:
    """Split a board frame into a 4x4 grid of cell crops, insetting each cell.

    The inset (15% per side by default, per the calibration spec) trims cell borders, inter-
    cell gaps, and rounded corners before any pixel reaches the classifier, since those are
    the pixels most likely to differ between an otherwise-identical tile sprite and its
    neighbor.
    """
    height, width = frame.shape[:2]
    cell_h = height / BOARD_SIZE
    cell_w = width / BOARD_SIZE

    rows: list[list[np.ndarray]] = []
    for row in range(BOARD_SIZE):
        cols: list[np.ndarray] = []
        for col in range(BOARD_SIZE):
            y0 = row * cell_h
            y1 = y0 + cell_h
            x0 = col * cell_w
            x1 = x0 + cell_w

            inset_y = cell_h * inset_ratio
            inset_x = cell_w * inset_ratio

            crop = frame[
                int(round(y0 + inset_y)) : int(round(y1 - inset_y)),
                int(round(x0 + inset_x)) : int(round(x1 - inset_x)),
            ]
            cols.append(crop)
        rows.append(cols)
    return rows


def flatten_cells(cells: list[list[np.ndarray]]) -> list[np.ndarray]:
    """Row-major flatten of :func:`split_cells` output, matching ``core.board`` cell indices."""
    return [crop for row in cells for crop in row]
