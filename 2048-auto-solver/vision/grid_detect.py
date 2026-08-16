"""Automatic 4x4 board detection.

Runs once during calibration (never in the play loop) to propose a region of interest for the
user to confirm, per the "detect first, confirm second" first-run principle. Two strategies
are tried, in order:

1. Find ~16 roughly-square, roughly-equal-sized contours arranged in a 4x4 lattice (the
   individual tile cells, including empty ones -- which usually render as a flat rounded
   square in the board's background color and still contour cleanly).
2. Fall back to the single largest square-ish contour, on the assumption it's the board's
   outer container.

Either way the result is a proposed :class:`GridDetectionResult` for the UI to render as an
overlay -- this module never applies the result to a live capture loop itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BOARD_SIZE = 4
EXPECTED_CELL_COUNT = BOARD_SIZE * BOARD_SIZE

# Fraction of the source frame's area a single cell / the whole board plausibly occupies.
_MIN_CELL_AREA_FRACTION = 0.002
_MAX_CELL_AREA_FRACTION = 0.10
_MIN_BOARD_AREA_FRACTION = 0.10
_MAX_BOARD_AREA_FRACTION = 0.98
_MAX_ASPECT_DEVIATION = 0.30  # |w/h - 1| must stay under this to count as "roughly square"


@dataclass(frozen=True)
class GridCell:
    row: int
    col: int
    center_x: float
    center_y: float
    size: float  # average of width/height, in source-frame pixels


@dataclass(frozen=True)
class GridDetectionResult:
    left: int
    top: int
    width: int
    height: int
    cells: tuple[GridCell, ...]
    confidence: float
    method: str  # "lattice" | "container" | "failed"


def _candidate_quads(gray: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Return (center_x, center_y, width, height) for plausible tile-sized square contours."""
    frame_area = gray.shape[0] * gray.shape[1]
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[float, float, float, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        area_fraction = area / frame_area if frame_area else 0.0
        if not (_MIN_CELL_AREA_FRACTION <= area_fraction <= _MAX_CELL_AREA_FRACTION):
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            continue
        aspect_deviation = abs((w / h) - 1.0)
        if aspect_deviation > _MAX_ASPECT_DEVIATION:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if not (4 <= len(approx) <= 8):  # rounded-corner squares approximate to a few extra points
            continue

        candidates.append((x + w / 2.0, y + h / 2.0, float(w), float(h)))
    return candidates


def _cluster_by_size(candidates: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    """Keep only candidates whose size is close to the modal cell size (tolerating a spread)."""
    if not candidates:
        return []
    sizes = sorted((w + h) / 2.0 for _, _, w, h in candidates)
    median_size = sizes[len(sizes) // 2]
    tolerance = max(median_size * 0.35, 4.0)
    return [c for c in candidates if abs(((c[2] + c[3]) / 2.0) - median_size) <= tolerance]


def _dedupe_candidates(
    candidates: list[tuple[float, float, float, float]]
) -> list[tuple[float, float, float, float]]:
    """Merge near-duplicate detections of the same physical cell (inner vs. outer edge contours)."""
    if not candidates:
        return []
    avg_size = sum((w + h) / 2.0 for _, _, w, h in candidates) / len(candidates)
    merge_radius = max(avg_size * 0.25, 3.0)

    remaining = list(candidates)
    merged: list[tuple[float, float, float, float]] = []
    while remaining:
        cx, cy, w, h = remaining.pop()
        group = [(cx, cy, w, h)]
        still_remaining = []
        for other in remaining:
            if abs(other[0] - cx) <= merge_radius and abs(other[1] - cy) <= merge_radius:
                group.append(other)
            else:
                still_remaining.append(other)
        remaining = still_remaining
        n = len(group)
        merged.append(
            (
                sum(g[0] for g in group) / n,
                sum(g[1] for g in group) / n,
                sum(g[2] for g in group) / n,
                sum(g[3] for g in group) / n,
            )
        )
    return merged


def _estimate_spacing(coords: list[float], min_spacing_floor: float) -> float:
    """Robustly estimate the lattice step size along one axis from observed cell centers.

    Uses the *smallest* gap between distinct observed coordinates as the estimate: as long as
    any two lattice-adjacent cells were both detected (overwhelmingly likely even with a few
    missing cells), that gap equals the true spacing, whereas gaps spanning a missing cell are
    simply larger multiples of it.
    """
    unique_sorted = sorted(coords)
    gaps = [b - a for a, b in zip(unique_sorted, unique_sorted[1:]) if (b - a) >= min_spacing_floor]
    if not gaps:
        return 0.0
    return min(gaps)


def _fit_lattice(candidates: list[tuple[float, float, float, float]]) -> list[GridCell] | None:
    """Try to explain ``candidates`` as a 4x4 lattice; tolerates missing cells but not extras."""
    deduped = _dedupe_candidates(candidates)
    if len(deduped) < 6:  # too few survivors to trust a lattice fit
        return None

    avg_size = sum((w + h) / 2.0 for _, _, w, h in deduped) / len(deduped)
    size_floor = avg_size * 0.5
    spacing_x = _estimate_spacing([c[0] for c in deduped], size_floor)
    spacing_y = _estimate_spacing([c[1] for c in deduped], size_floor)
    if spacing_x <= 0 or spacing_y <= 0:
        return None

    xs = [c[0] for c in deduped]
    ys = [c[1] for c in deduped]
    origin_x, origin_y = min(xs), min(ys)

    slots: dict[tuple[int, int], list[tuple[float, float, float, float]]] = {}
    for cx, cy, w, h in deduped:
        col = round((cx - origin_x) / spacing_x)
        row = round((cy - origin_y) / spacing_y)
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            slots.setdefault((row, col), []).append((cx, cy, w, h))

    if len(slots) < 6:  # need enough evidence to trust the lattice, but tolerate empty cells
        return None

    cells: list[GridCell] = []
    for (row, col), members in slots.items():
        avg_cx = sum(m[0] for m in members) / len(members)
        avg_cy = sum(m[1] for m in members) / len(members)
        avg_member_size = sum((m[2] + m[3]) / 2.0 for m in members) / len(members)
        cells.append(GridCell(row=row, col=col, center_x=avg_cx, center_y=avg_cy, size=avg_member_size))
    return cells


def _bounding_roi_from_cells(cells: list[GridCell], frame_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    avg_size = sum(c.size for c in cells) / len(cells)

    def _axis_bounds(coord_of, span_of) -> tuple[float, float]:
        min_coord = min(coord_of(c) for c in cells)
        max_coord = max(coord_of(c) for c in cells)
        spacing = span_of()
        observed_span = round((max_coord - min_coord) / spacing) + 1 if spacing else 1
        observed_span = max(1, min(BOARD_SIZE, observed_span))
        missing = BOARD_SIZE - observed_span
        pad_before = missing // 2
        pad_after = missing - pad_before
        lo = min_coord - spacing * (0.5 + pad_before)
        hi = max_coord + spacing * (0.5 + pad_after)
        return lo, hi

    # Re-derive spacing per axis the same way _fit_lattice did, from the final cell set.
    xs = sorted({c.center_x for c in cells})
    ys = sorted({c.center_y for c in cells})
    spacing_x = min((b - a for a, b in zip(xs, xs[1:])), default=avg_size) or avg_size
    spacing_y = min((b - a for a, b in zip(ys, ys[1:])), default=avg_size) or avg_size

    min_x, max_x = _axis_bounds(lambda c: c.center_x, lambda: spacing_x)
    min_y, max_y = _axis_bounds(lambda c: c.center_y, lambda: spacing_y)

    frame_h, frame_w = frame_shape
    left = max(0, int(round(min_x)))
    top = max(0, int(round(min_y)))
    right = min(frame_w, int(round(max_x)))
    bottom = min(frame_h, int(round(max_y)))
    return left, top, max(1, right - left), max(1, bottom - top)


def _detect_container(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """Fallback: find the single largest square-ish contour (the board's outer container)."""
    frame_area = gray.shape[0] * gray.shape[1]
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best: tuple[int, int, int, int] | None = None
    best_area = 0.0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        area_fraction = area / frame_area if frame_area else 0.0
        if not (_MIN_BOARD_AREA_FRACTION <= area_fraction <= _MAX_BOARD_AREA_FRACTION):
            continue
        if h == 0 or abs((w / h) - 1.0) > _MAX_ASPECT_DEVIATION:
            continue
        if area > best_area:
            best_area = area
            best = (x, y, w, h)
    return best


def _sanity_check(left: int, top: int, width: int, height: int, frame_shape: tuple[int, int]) -> bool:
    frame_h, frame_w = frame_shape
    frame_area = frame_h * frame_w
    if frame_area == 0:
        return False
    area_fraction = (width * height) / frame_area
    if not (_MIN_BOARD_AREA_FRACTION <= area_fraction <= _MAX_BOARD_AREA_FRACTION):
        return False
    if height == 0:
        return False
    return abs((width / height) - 1.0) <= _MAX_ASPECT_DEVIATION


def detect_grid(frame: np.ndarray) -> GridDetectionResult:
    """Propose a board ROI from a captured window frame (BGR). Never raises on failure."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_shape = gray.shape[:2]

    candidates = _cluster_by_size(_candidate_quads(gray))
    lattice = _fit_lattice(candidates)
    if lattice is not None:
        left, top, width, height = _bounding_roi_from_cells(lattice, frame_shape)
        if _sanity_check(left, top, width, height, frame_shape):
            confidence = min(1.0, len(lattice) / EXPECTED_CELL_COUNT)
            logger.info(
                "Grid detected via lattice fit: %d/%d cells, roi=(%d,%d,%d,%d)",
                len(lattice), EXPECTED_CELL_COUNT, left, top, width, height,
            )
            return GridDetectionResult(
                left=left, top=top, width=width, height=height,
                cells=tuple(lattice), confidence=confidence, method="lattice",
            )
        logger.debug("Lattice fit failed the ROI sanity check; falling back to container detection.")

    container = _detect_container(gray)
    if container is not None:
        left, top, width, height = container
        if _sanity_check(left, top, width, height, frame_shape):
            logger.info("Grid detected via container fallback: roi=(%d,%d,%d,%d)", left, top, width, height)
            return GridDetectionResult(
                left=left, top=top, width=width, height=height,
                cells=(), confidence=0.5, method="container",
            )

    logger.warning("Automatic grid detection failed; the user will need to adjust manually.")
    frame_h, frame_w = frame_shape
    return GridDetectionResult(
        left=0, top=0, width=frame_w, height=frame_h, cells=(), confidence=0.0, method="failed",
    )
