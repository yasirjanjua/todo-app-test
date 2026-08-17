"""Unit tests for vision/grid_detect.py's sanity bounds."""

from __future__ import annotations

import cv2
import numpy as np

from vision.grid_detect import detect_grid


def test_rejects_a_near_full_frame_container_instead_of_reporting_it_confidently() -> None:
    """Regression test for a real report: the container-fallback path proposed
    roi=(0, 51, 2136, 1847) on a captured browser window -- essentially the entire frame, not
    the actual 4x4 board -- and the app treated it as a usable detection. A captured window
    always has *something* around the board (page chrome, a title bar, at minimum the
    browser's own UI), so a "container" candidate that swallows nearly the whole frame is a
    sign detection latched onto the page background/wrapper, not the board, and must be
    rejected rather than confidently reported.
    """
    frame = np.full((1898, 2136, 3), (245, 240, 230), dtype=np.uint8)
    # A single big rounded-corner-ish rect covering nearly the entire frame, exactly the shape
    # of contour a page's outer content wrapper would produce.
    cv2.rectangle(frame, (0, 51), (2136, 1898), (200, 190, 170), -1)

    result = detect_grid(frame)

    assert result.method == "failed"
    assert result.confidence == 0.0


def test_accepts_a_plausibly_sized_board_container() -> None:
    """A container occupying a modest, plausible fraction of the frame (with real chrome
    around it, unlike the near-full-frame case above) should still be usable."""
    frame = np.full((1000, 1000, 3), (245, 240, 230), dtype=np.uint8)
    # A ~45%-of-frame-area square board container, with page background all around it --
    # representative of a real 2048 page layout (title/score above, footer/ads beside/below).
    cv2.rectangle(frame, (250, 250), (900, 900), (187, 173, 160), -1)

    result = detect_grid(frame)

    assert result.method in ("lattice", "container")
    assert result.confidence > 0.0
    frame_area = frame.shape[0] * frame.shape[1]
    roi_area = result.width * result.height
    assert roi_area / frame_area <= 0.6
