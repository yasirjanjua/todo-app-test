"""Unit tests for app/profile.py's persistence and load-time sanitization."""

from __future__ import annotations

import numpy as np

from app.profile import GameProfile, ProfileStore, RoiOffset, profile_from_json, profile_to_json
from core.board import MAX_TIER
from vision.recognition import make_template


def _make_profile(tile_templates: dict[int, object]) -> GameProfile:
    return GameProfile(
        window_title="Test Game",
        roi=RoiOffset(10, 20, 300, 300),
        window_size_at_calibration=(800, 600),
        tile_templates=tile_templates,
        settle_frames=2,
        settle_timeout_ms=1500.0,
        confidence_threshold=0.6,
    )


def test_profile_round_trips(tmp_path) -> None:
    store = ProfileStore(tmp_path)
    templates = {1: make_template(1, np.full((64, 64, 3), (200, 190, 180), dtype=np.uint8))}
    profile = _make_profile(templates)

    store.save(profile)
    loaded = store.load("Test Game")

    assert loaded is not None
    assert loaded.window_title == "Test Game"
    assert loaded.roi == profile.roi
    assert set(loaded.tile_templates.keys()) == {1}
    assert loaded.tile_templates[1].phash == templates[1].phash


def test_load_drops_out_of_range_tile_templates() -> None:
    """Regression test: a profile saved by a build that predates the tile-learning anomaly
    guard (see vision/tile_learning.py) can contain templates minted during a runaway-learning
    episode, with tiers past what a 4-bit bitboard cell can represent (real report: tiers up
    to 17 in a saved profile). Loading such a profile must not resurrect those -- otherwise
    TileLearner.resume_for_play() would immediately set next_tier past MAX_TIER again.
    """
    good_template = make_template(2, np.full((64, 64, 3), (210, 200, 190), dtype=np.uint8))
    bad_template = make_template(17, np.full((64, 64, 3), (10, 10, 10), dtype=np.uint8))
    profile = _make_profile({2: good_template, 17: bad_template})

    data = profile_to_json(profile)
    # profile_to_json is well-behaved and wouldn't itself produce this, but the JSON on disk
    # from an old, buggy build would look exactly like this -- construct it directly to
    # simulate loading that stale file.
    assert any(t["tier"] == 17 for t in data["tile_templates"])

    reloaded = profile_from_json(data)

    assert set(reloaded.tile_templates.keys()) == {2}
    assert all(tier <= MAX_TIER for tier in reloaded.tile_templates)


def test_load_drops_zero_and_negative_tiers_too() -> None:
    good_template = make_template(1, np.full((64, 64, 3), (200, 190, 180), dtype=np.uint8))
    profile = _make_profile({1: good_template})
    data = profile_to_json(profile)
    data["tile_templates"].append({"tier": 0, "phash": str(good_template.phash), "thumbnail_png_b64": data["tile_templates"][0]["thumbnail_png_b64"]})

    reloaded = profile_from_json(data)

    assert set(reloaded.tile_templates.keys()) == {1}
