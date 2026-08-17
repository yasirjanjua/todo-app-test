"""The tile-learning state machine: how the app teaches itself a board's sprite set.

Implements the procedure from the first-run spec, without requiring the user to label
anything:

1. A fresh board (right after "New Game") contains only the lowest tile -- whatever sprite is
   present is Tier 1, captured automatically.
2. The first *unrecognized* sprite seen afterward is Tier 2, but it is held for one explicit
   user confirmation, because 2048 spawns the lowest and second-lowest tile randomly
   (~90/10), so an early Tier-2 sighting can't be trusted on its own.
3. Every tier above 2 can only appear via a merge, so tiers are strictly ascending; each new
   unrecognized sprite auto-promotes to the next tier silently once Tier 2 is confirmed.
4. If the user starts mid-game, :meth:`TileLearner.seed_from_frequency` ranks unknown sprites
   by observation frequency (lower tiers appear more often) and asks for one confirmation
   instead of a full walkthrough.

Numeric and solid-color boards need no separate code path: a rendered digit or a flat color
is just another sprite to this state machine.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from core.board import MAX_TIER
from vision.recognition import TileRecognizer, is_empty_cell, make_template

logger = logging.getLogger(__name__)


class LearningPhase(Enum):
    AWAITING_FRESH_BOARD = auto()  # waiting on the user to start a new game
    AWAITING_TIER1_SPRITE = auto()  # fresh board started; capturing the sole sprite as tier 1
    AWAITING_TIER2_CONFIRMATION = auto()  # first unrecognized sprite seen; needs user confirmation
    AUTO_PROMOTING = auto()  # steady state: new unrecognized sprites silently become the next tier
    MID_GAME_FALLBACK = auto()  # user started mid-game; ranking unknowns by frequency instead


@dataclass
class LearningEvent:
    """Something the UI should react to (show a toast, ask for confirmation, etc.)."""

    kind: str  # "learned_tile" | "await_confirmation" | "anomaly" | "no_change"
    tier: int | None = None
    crop: np.ndarray | None = None


@dataclass
class TileLearner:
    """Drives template acquisition; wraps a :class:`TileRecognizer` it populates over time."""

    recognizer: TileRecognizer = field(default_factory=TileRecognizer)
    phase: LearningPhase = LearningPhase.AWAITING_FRESH_BOARD
    next_tier: int = 1
    # A single board read revealing more new tiers than this is not physically plausible from
    # normal play (a move can advance the board's max tier by at most one merge chain -- see
    # the module docstring), so exceeding it in one observe() call means recognition itself
    # has gone unreliable (most likely the calibrated region has drifted onto changing content
    # instead of the actual board) rather than that the game genuinely produced that many
    # brand-new tiles at once. See _observe_auto_promote's "anomaly" LearningEvent.
    max_new_tiles_per_observation: int = 2
    # Mirrors max_new_tiles_per_observation's reasoning for the mid-game bootstrapping path:
    # observing this many distinct sprites while ranking a board that's supposedly still
    # early enough to need bootstrapping at all is implausible (every real report that hit
    # this had a game score in the tens, i.e. a max tile around 8 -- tier 3 -- on screen) and
    # is a much stronger sign the calibrated region isn't actually the board than that the
    # game genuinely has that many distinct tiles. See finish_mid_game_ranking.
    max_mid_game_tiers: int = 8
    _pending_tier2_crop: np.ndarray | None = field(default=None, repr=False)
    _mid_game_observations: Counter[bytes] = field(default_factory=Counter, repr=False)
    _mid_game_crop_by_hash: dict[bytes, np.ndarray] = field(default_factory=dict, repr=False)

    def start_new_game(self) -> None:
        """Call when the user clicks the "start a new game" learning button."""
        self.recognizer = TileRecognizer(confidence_threshold=self.recognizer.confidence_threshold)
        self.phase = LearningPhase.AWAITING_TIER1_SPRITE
        self.next_tier = 1
        self._pending_tier2_crop = None
        logger.info("Tile learning: waiting for the fresh board's sole tile to appear.")

    def start_mid_game_fallback(self) -> None:
        """Call when the user starts observation on a board that's already in progress."""
        self.phase = LearningPhase.MID_GAME_FALLBACK
        self._mid_game_observations.clear()
        self._mid_game_crop_by_hash.clear()
        logger.info("Tile learning: falling back to frequency ranking for a mid-game start.")

    def resume_for_play(self) -> None:
        """(Re)enter the steady auto-promoting state from the recognizer's existing templates.

        Used when resuming a previously-calibrated profile, where there is no live
        ``TileLearner`` left over from the original calibration session -- only the templates
        it produced. Tiers above 2 can only ever arise from a merge (see the module
        docstring), so that invariant holds regardless of when or in which session the known
        templates were learned, which is what makes it safe to auto-promote from here without
        the tier-2 confirmation calibration itself required.
        """
        self.next_tier = max(self.recognizer.templates.keys(), default=0) + 1
        self.phase = LearningPhase.AUTO_PROMOTING
        logger.info("Tile learning: resuming in auto-promote mode from tier %d.", self.next_tier)

    def observe(self, cell_crops: list[np.ndarray]) -> list[LearningEvent]:
        """Feed one frame's worth of 16 cell crops through the learner; returns UI events."""
        if self.phase is LearningPhase.MID_GAME_FALLBACK:
            return self._observe_mid_game(cell_crops)

        non_empty = [crop for crop in cell_crops if not is_empty_cell(crop)]

        if self.phase is LearningPhase.AWAITING_TIER1_SPRITE:
            return self._observe_tier1(non_empty)

        if self.phase is LearningPhase.AWAITING_TIER2_CONFIRMATION:
            # Hold steady; we're waiting on an explicit confirm_tier2() call, not new frames.
            return []

        if self.phase is LearningPhase.AUTO_PROMOTING:
            return self._observe_auto_promote(non_empty)

        return []

    def _observe_tier1(self, non_empty_crops: list[np.ndarray]) -> list[LearningEvent]:
        if not non_empty_crops:
            return []
        # A fresh board has exactly one populated sprite (occasionally two, if both starting
        # tiles happened to render identically); either way, the first crop we see is tier 1.
        template = make_template(1, non_empty_crops[0])
        self.recognizer.add_template(template)
        self.phase = LearningPhase.AUTO_PROMOTING
        self.next_tier = 2
        logger.info("Tile learning: captured tier 1 from the fresh board.")
        return [LearningEvent(kind="learned_tile", tier=1, crop=non_empty_crops[0])]

    def _observe_auto_promote(self, non_empty_crops: list[np.ndarray]) -> list[LearningEvent]:
        events: list[LearningEvent] = []
        learned_this_call = 0
        anomaly_flagged = False
        for crop in non_empty_crops:
            result = self.recognizer.classify(crop)
            if result.tier is not None:
                continue  # already-known tier, nothing to learn

            if self.next_tier == 2:
                # The 90/10 spawn means an early unknown sprite might just be tier 1's sibling
                # rather than a merge product; hold it for one explicit confirmation.
                self._pending_tier2_crop = crop
                self.phase = LearningPhase.AWAITING_TIER2_CONFIRMATION
                events.append(LearningEvent(kind="await_confirmation", tier=2, crop=crop))
                return events

            if self.next_tier > MAX_TIER:
                # The bitboard's 4-bit cells physically cannot hold a tier above MAX_TIER (15,
                # i.e. on-screen value 32768) -- core.board.set_cell raises ValueError past
                # that. Reaching this in real play would already be an extraordinary game;
                # reaching it after only a handful of moves (as opposed to a long session)
                # means recognition is almost certainly drifting -- e.g. the ROI has landed on
                # dynamic content (an ad, a changing thumbnail) rather than the actual board,
                # and every poll looks like a brand-new "sprite". Stop minting templates and
                # leave this crop unrecognized so the caller's normal pause-and-diagnose path
                # takes over instead of silently corrupting the board state further.
                logger.warning(
                    "Refusing to learn a tier above %d; recognition looks unstable (wrong ROI?).",
                    MAX_TIER,
                )
                continue

            if learned_this_call >= self.max_new_tiles_per_observation:
                if not anomaly_flagged:
                    logger.error(
                        "More than %d new tiles in a single observation; this isn't physically "
                        "plausible from normal play and almost certainly means the calibrated "
                        "region is no longer looking at the actual board.",
                        self.max_new_tiles_per_observation,
                    )
                    events.append(LearningEvent(kind="anomaly", tier=None, crop=crop))
                    anomaly_flagged = True
                continue

            # Tier 3+ can only arise from a merge, so ascending order is guaranteed; promote
            # silently.
            template = make_template(self.next_tier, crop)
            self.recognizer.add_template(template)
            logger.info("Tile learning: auto-promoted a new sprite to tier %d.", self.next_tier)
            events.append(LearningEvent(kind="learned_tile", tier=self.next_tier, crop=crop))
            self.next_tier += 1
            learned_this_call += 1
        return events

    def confirm_tier2(self, accept: bool) -> LearningEvent | None:
        """Resolve the pending tier-2 confirmation (see :class:`LearningPhase`)."""
        if self.phase is not LearningPhase.AWAITING_TIER2_CONFIRMATION or self._pending_tier2_crop is None:
            logger.warning("confirm_tier2() called with no pending confirmation; ignoring.")
            return None

        crop = self._pending_tier2_crop
        self._pending_tier2_crop = None
        self.phase = LearningPhase.AUTO_PROMOTING

        if not accept:
            # User says this wasn't actually a new tile (e.g. a misread of an existing one);
            # stay at tier 2 and wait for the next unrecognized sprite.
            logger.info("Tile learning: tier-2 candidate rejected by the user.")
            return None

        template = make_template(2, crop)
        self.recognizer.add_template(template)
        self.next_tier = 3
        logger.info("Tile learning: tier 2 confirmed by the user.")
        return LearningEvent(kind="learned_tile", tier=2, crop=crop)

    def _observe_mid_game(self, cell_crops: list[np.ndarray]) -> list[LearningEvent]:
        from vision.recognition import compute_phash

        for crop in cell_crops:
            if is_empty_cell(crop):
                continue
            key = bytes(str(compute_phash(crop)), "ascii")
            self._mid_game_observations[key] += 1
            self._mid_game_crop_by_hash.setdefault(key, crop)
        return []

    def finish_mid_game_ranking(self) -> list[LearningEvent]:
        """Rank observed unique sprites by frequency (most common = lowest tier) and learn them.

        Returns one ``learned_tile`` event per ranked sprite plus a trailing
        ``await_confirmation`` event, since the spec calls for a single confirmation of the
        overall ranking rather than a per-tile walkthrough. Returns a single ``anomaly`` event
        instead, ranking nothing, if the observed sprite count is implausible (see
        ``max_mid_game_tiers``) -- a real report showed this ranking 14 "tiles" from a board
        that never showed anything above an 8 on screen, all silently accepted with no warning
        at all.
        """
        ranked = self._mid_game_observations.most_common()

        if len(ranked) > self.max_mid_game_tiers:
            logger.error(
                "Observed %d distinct sprites while bootstrapping from a mid-game board -- "
                "more than a real board plausibly shows in this short a window. This almost "
                "certainly means the calibrated region isn't actually the board. Refusing to "
                "rank them.",
                len(ranked),
            )
            self._mid_game_observations.clear()
            self._mid_game_crop_by_hash.clear()
            return [LearningEvent(kind="anomaly", tier=None, crop=None)]

        events: list[LearningEvent] = []
        for tier, (key, _count) in enumerate(ranked, start=1):
            crop = self._mid_game_crop_by_hash[key]
            self.recognizer.add_template(make_template(tier, crop))
            events.append(LearningEvent(kind="learned_tile", tier=tier, crop=crop))
        self.next_tier = len(ranked) + 1
        self.phase = LearningPhase.AUTO_PROMOTING
        if events:
            events.append(LearningEvent(kind="await_confirmation", tier=None, crop=None))
        return events
