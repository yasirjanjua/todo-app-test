"""Per-cell tile recognition.

Every classification is tier-based, never value-based: a stored template maps to an ordinal
tier, and the caller (``core.board``) treats tier ``n`` as tile value ``2**n``. That single
design choice is what makes a fruit board and a numeric board the same problem -- see
``vision/tile_learning.py`` for how templates get populated in the first place.

Matching is perceptual-hash-first (``imagehash.phash``), which is robust to the anti-aliasing
and minor color/lighting drift between one capture and the next, with normalized
cross-correlation (``cv2.matchTemplate``) as a tiebreaker when two templates hash close
together. ORB/feature matching is deliberately not used: 2048 sprites are small, flat, and
low-detail, exactly the conditions ORB's corner-based descriptors handle worst.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import imagehash
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

TEMPLATE_SIZE = 64  # all crops are resized to TEMPLATE_SIZE x TEMPLATE_SIZE before comparison
PHASH_SIZE = 16  # imagehash hash_size; 16 -> 256-bit hash, enough resolution for flat sprites
MAX_HAMMING_DISTANCE = PHASH_SIZE * PHASH_SIZE  # 256, the hash's full bit length

# An empty cell renders as a near-uniform patch of the board's background color; below this
# per-pixel standard deviation, don't even attempt template matching.
_EMPTY_STD_THRESHOLD = 6.0

DEFAULT_CONFIDENCE_THRESHOLD = 0.6


@dataclass(frozen=True)
class TileTemplate:
    """One learned tile tier: its perceptual hash plus a small thumbnail for tiebreaking."""

    tier: int
    phash: imagehash.ImageHash
    thumbnail: np.ndarray  # BGR, TEMPLATE_SIZE x TEMPLATE_SIZE


@dataclass(frozen=True)
class RecognitionResult:
    tier: int | None  # None means "below confidence threshold, do not guess"
    confidence: float  # 0.0-1.0
    is_empty: bool


def normalize_crop(crop: np.ndarray) -> np.ndarray:
    """Resize a raw cell crop to the fixed template size used for all comparisons."""
    if crop.size == 0:
        return np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE, 3), dtype=np.uint8)
    return cv2.resize(crop, (TEMPLATE_SIZE, TEMPLATE_SIZE), interpolation=cv2.INTER_AREA)


def compute_phash(crop_bgr: np.ndarray) -> imagehash.ImageHash:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    return imagehash.phash(pil_image, hash_size=PHASH_SIZE)


def make_template(tier: int, crop: np.ndarray) -> TileTemplate:
    """Build a :class:`TileTemplate` from a raw cell crop (called by the tile learner)."""
    normalized = normalize_crop(crop)
    return TileTemplate(tier=tier, phash=compute_phash(normalized), thumbnail=normalized)


def is_empty_cell(crop: np.ndarray) -> bool:
    """A cell is "empty" when it's a near-uniform patch (background color, no sprite)."""
    if crop.size == 0:
        return True
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(np.std(gray)) < _EMPTY_STD_THRESHOLD


def _cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    result = cv2.matchTemplate(a_gray, b_gray, cv2.TM_CCOEFF_NORMED)
    return float(result[0, 0])


class TileRecognizer:
    """Classifies cell crops against a set of learned :class:`TileTemplate` instances."""

    def __init__(
        self,
        templates: dict[int, TileTemplate] | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.templates: dict[int, TileTemplate] = dict(templates or {})
        self.confidence_threshold = confidence_threshold

    def add_template(self, template: TileTemplate) -> None:
        self.templates[template.tier] = template

    def classify(self, crop: np.ndarray) -> RecognitionResult:
        """Classify a single cell crop. Never guesses below ``confidence_threshold``."""
        if is_empty_cell(crop):
            return RecognitionResult(tier=0, confidence=1.0, is_empty=True)

        if not self.templates:
            return RecognitionResult(tier=None, confidence=0.0, is_empty=False)

        normalized = normalize_crop(crop)
        crop_hash = compute_phash(normalized)

        # Rank every template by perceptual-hash Hamming distance (primary signal).
        ranked = sorted(
            self.templates.values(),
            key=lambda t: crop_hash - t.phash,
        )
        best = ranked[0]
        best_distance = crop_hash - best.phash
        phash_confidence = 1.0 - (best_distance / MAX_HAMMING_DISTANCE)

        # If the top two candidates are close in hash distance, break the tie with normalized
        # cross-correlation, which is more sensitive to fine pixel differences than phash.
        if len(ranked) > 1:
            runner_up = ranked[1]
            runner_up_distance = crop_hash - runner_up.phash
            if abs(best_distance - runner_up_distance) <= 4:
                best_corr = _cross_correlation(normalized, best.thumbnail)
                runner_up_corr = _cross_correlation(normalized, runner_up.thumbnail)
                if runner_up_corr > best_corr:
                    best = runner_up
                    phash_confidence = 1.0 - (runner_up_distance / MAX_HAMMING_DISTANCE)

        if phash_confidence < self.confidence_threshold:
            logger.debug(
                "Low-confidence tile classification (%.2f < %.2f); routing to caller for a decision.",
                phash_confidence, self.confidence_threshold,
            )
            return RecognitionResult(tier=None, confidence=phash_confidence, is_empty=False)

        return RecognitionResult(tier=best.tier, confidence=phash_confidence, is_empty=False)

    def classify_board(self, cell_crops: list[np.ndarray]) -> list[RecognitionResult]:
        """Classify all 16 cells (row-major, matching ``core.board`` indexing)."""
        return [self.classify(crop) for crop in cell_crops]
