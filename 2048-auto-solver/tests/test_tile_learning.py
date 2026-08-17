"""Unit tests for vision/tile_learning.py's TileLearner."""

from __future__ import annotations

import numpy as np

from core.board import MAX_TIER
from vision.tile_learning import LearningPhase, TileLearner


def _noise_crop(seed: int, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(30, 220, (size, size, 3), dtype=np.uint8)


def test_mid_game_ranking_learns_a_plausible_number_of_tiers() -> None:
    learner = TileLearner()
    learner.start_mid_game_fallback()
    for i in range(4):
        learner._observe_mid_game([_noise_crop(i)])

    events = learner.finish_mid_game_ranking()

    learned = [e for e in events if e.kind == "learned_tile"]
    assert len(learned) == 4
    assert events[-1].kind == "await_confirmation"
    assert set(learner.recognizer.templates) == {1, 2, 3, 4}


def test_mid_game_ranking_rejects_an_implausible_number_of_distinct_sprites() -> None:
    """Regression test for a real report: a drifted ROI produced 14 distinct "tiles" while
    bootstrapping from a board that only ever showed a single-digit tile on screen, and the
    (buggy) old code ranked and learned all 14 with no warning at all.
    """
    learner = TileLearner()
    learner.start_mid_game_fallback()
    for i in range(14):
        learner._observe_mid_game([_noise_crop(i)])

    events = learner.finish_mid_game_ranking()

    assert len(events) == 1
    assert events[0].kind == "anomaly"
    assert learner.recognizer.templates == {}, "must not learn any of them -- all-or-nothing"


def test_mid_game_ranking_at_exactly_the_cap_still_succeeds() -> None:
    learner = TileLearner(max_mid_game_tiers=8)
    learner.start_mid_game_fallback()
    for i in range(8):
        learner._observe_mid_game([_noise_crop(i)])

    events = learner.finish_mid_game_ranking()

    assert all(e.kind != "anomaly" for e in events)
    assert len(learner.recognizer.templates) == 8


def test_auto_promote_never_learns_a_tier_above_the_bitboard_limit() -> None:
    """Regression test for a real crash: next_tier climbing past MAX_TIER (15) used to reach
    core.board.set_cell() and raise ValueError. The learner must refuse to mint templates
    beyond the representable range, full stop.
    """
    # A generous per-observation cap isolates this test to the MAX_TIER ceiling specifically,
    # separate from max_new_tiles_per_observation (covered by the test below).
    learner = TileLearner(max_new_tiles_per_observation=100)
    learner.next_tier = 3  # tiers 1 and 2 already known, as after a normal calibration
    learner.phase = LearningPhase.AUTO_PROMOTING

    crops = [_noise_crop(i) for i in range(20)]
    events = learner.observe(crops)

    learned_tiers = [e.tier for e in events if e.kind == "learned_tile"]
    assert all(tier <= MAX_TIER for tier in learned_tiers)
    assert learner.next_tier == MAX_TIER + 1


def test_auto_promote_flags_anomaly_after_max_new_tiles_per_observation() -> None:
    learner = TileLearner(max_new_tiles_per_observation=2)
    learner.next_tier = 3
    learner.phase = LearningPhase.AUTO_PROMOTING

    crops = [_noise_crop(i) for i in range(5)]
    events = learner.observe(crops)

    assert any(e.kind == "anomaly" for e in events)
    learned = [e for e in events if e.kind == "learned_tile"]
    assert len(learned) == 2
