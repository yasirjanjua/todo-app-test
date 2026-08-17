"""Integration smoke test for the closed-loop play controller against a fake game.

Exercises the full perceive -> decide -> act -> verify cycle (app/play_loop.py) without any
real screen or OS input: a tiny in-memory "game" renders synthetic tile sprites to a frame and
applies core.board moves in response to injected keys, standing in for backends.capture and
backends.input.
"""

from __future__ import annotations

import random
import time

import cv2
import numpy as np

from app.config import AppConfig
from app.play_loop import PlayController, PlayState
from backends.capture.base import CaptureRegion
from backends.input.base import Key
from core.board import Move, empty_cells, from_grid, set_cell
from vision.capture import Roi
from vision.recognition import TileRecognizer, make_template
from vision.tile_learning import TileLearner

_KEY_TO_MOVE = {Key.UP: Move.UP, Key.DOWN: Move.DOWN, Key.LEFT: Move.LEFT, Key.RIGHT: Move.RIGHT}
_TIER_COLORS = {
    1: (200, 193, 180),
    2: (238, 225, 201),
    3: (243, 178, 122),
    4: (246, 150, 100),
    5: (247, 124, 95),
    6: (237, 204, 97),
}
_EMPTY_COLOR = (187, 173, 160)
_CELL_PX = 80


def _render(board: int) -> np.ndarray:
    from core.board import to_grid

    grid = to_grid(board)
    frame = np.zeros((_CELL_PX * 4, _CELL_PX * 4, 3), dtype=np.uint8)
    for r in range(4):
        for c in range(4):
            tier = grid[r][c]
            color = _EMPTY_COLOR if tier == 0 else _TIER_COLORS.get(tier, (0, 0, 0))
            y0, x0 = r * _CELL_PX, c * _CELL_PX
            cv2.rectangle(frame, (x0, y0), (x0 + _CELL_PX, y0 + _CELL_PX), color, -1)
            if tier:
                cv2.putText(
                    frame, str(tier), (x0 + 15, y0 + 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (50, 50, 50), 3, cv2.LINE_AA
                )
    return frame


def _spawn_random_tile(board: int, rng: random.Random) -> int:
    empties = empty_cells(board)
    if not empties:
        return board
    index = rng.choice(empties)
    tier = 1 if rng.random() < 0.9 else 2
    return set_cell(board, index, tier)


class FakeGame:
    def __init__(self, seed: int = 0) -> None:
        rng = random.Random(seed)
        board = 0
        board = _spawn_random_tile(board, rng)
        board = _spawn_random_tile(board, rng)
        self.board = board
        self._rng = rng

    def apply_key(self, key: Key) -> None:
        from core.board import move as apply_move

        result = apply_move(self.board, _KEY_TO_MOVE[key])
        if result.moved:
            self.board = _spawn_random_tile(result.board, self._rng)

    def frame(self) -> np.ndarray:
        return _render(self.board)


class FakeCaptureBackend:
    def __init__(self, game: FakeGame) -> None:
        self._game = game

    def list_monitors(self):
        return []

    def get_scale_factor(self, monitor_index: int = 0) -> float:
        return 1.0

    def grab(self, region: CaptureRegion) -> np.ndarray:
        return self._game.frame()

    def close(self) -> None:
        pass


class FakeInputBackend:
    def __init__(self, game: FakeGame) -> None:
        self._game = game
        self.taps: list[Key] = []

    def tap_key(self, key: Key, hold_ms: int = 40) -> None:
        self.taps.append(key)
        self._game.apply_key(key)

    def release_all(self) -> None:
        pass


def _build_tile_learner(known_tiers: set[int] | None = None) -> TileLearner:
    from core.board import set_cell
    from vision.capture import flatten_cells, split_cells

    # Build templates from the *inset* crop (matching how the play loop crops cells at
    # recognition time, per vision/capture.py's 15% inset), not the raw uncropped sprite --
    # comparing an inset crop against a non-inset template would otherwise systematically
    # under-report confidence.
    recognizer = TileRecognizer(confidence_threshold=0.6)
    for tier in _TIER_COLORS:
        if known_tiers is not None and tier not in known_tiers:
            continue
        board = set_cell(0, 0, tier)
        frame = _render(board)
        cell_crop = flatten_cells(split_cells(frame, inset_ratio=0.15))[0]
        recognizer.add_template(make_template(tier, cell_crop))
    learner = TileLearner(recognizer=recognizer)
    # Mirrors what ui/main_window.py does when resuming a saved profile: pre-populated
    # templates but no live calibration session, so jump straight into steady-state
    # auto-promotion instead of waiting on a "start a new game" click that will never come.
    learner.resume_for_play()
    return learner


def test_play_loop_plays_moves_and_stops_cleanly() -> None:
    game = FakeGame(seed=42)
    capture = FakeCaptureBackend(game)
    input_backend = FakeInputBackend(game)
    tile_learner = _build_tile_learner()
    roi = Roi(left=0, top=0, width=_CELL_PX * 4, height=_CELL_PX * 4)
    config = AppConfig(settle_timeout_ms=200.0, settle_frames=1)

    events = []
    controller = PlayController(
        capture_backend=capture,
        input_backend=input_backend,
        tile_learner=tile_learner,
        roi=roi,
        app_config=config,
        on_event=events.append,
    )

    controller.start()
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline and controller.stats.moves_made < 5:
        time.sleep(0.05)
    controller.stop()

    assert controller.state == PlayState.STOPPED
    assert controller.stats.moves_made >= 1
    assert len(input_backend.taps) >= 1
    # No exceptions should have been logged as "error" events.
    assert all(e.kind != "error" for e in events)


def test_play_loop_dry_run_never_calls_input() -> None:
    game = FakeGame(seed=7)
    capture = FakeCaptureBackend(game)
    input_backend = FakeInputBackend(game)
    tile_learner = _build_tile_learner()
    roi = Roi(left=0, top=0, width=_CELL_PX * 4, height=_CELL_PX * 4)
    config = AppConfig(dry_run=True, settle_timeout_ms=200.0, settle_frames=1)

    events = []
    controller = PlayController(
        capture_backend=capture,
        input_backend=input_backend,
        tile_learner=tile_learner,
        roi=roi,
        app_config=config,
        on_event=events.append,
    )
    controller.start()
    time.sleep(0.5)
    controller.stop()

    assert input_backend.taps == []
    assert any(e.kind == "decision" for e in events)


def test_play_loop_learns_new_tier_live_instead_of_pausing_forever() -> None:
    """Regression test: a resumed profile's recognizer only knows the tiers it saw during its
    original calibration; a merge in the *current* session routinely produces a tier that
    calibration never saw. Before the live-learning fix, this paused the controller
    indefinitely in PAUSED_UNKNOWN_TILE with no way to recover -- exactly what a real user hit.
    """
    game = FakeGame(seed=1)
    # Force a merge on the very next LEFT move: two tier-2 tiles adjacent in row 0, producing
    # a tier-3 tile the learner (which only knows tiers 1 and 2) has never seen.
    game.board = from_grid([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    capture = FakeCaptureBackend(game)
    input_backend = FakeInputBackend(game)
    tile_learner = _build_tile_learner(known_tiers={1, 2})
    roi = Roi(left=0, top=0, width=_CELL_PX * 4, height=_CELL_PX * 4)
    config = AppConfig(settle_timeout_ms=200.0, settle_frames=1)

    events = []
    controller = PlayController(
        capture_backend=capture,
        input_backend=input_backend,
        tile_learner=tile_learner,
        roi=roi,
        app_config=config,
        on_event=events.append,
    )

    controller.start()
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline and controller.stats.moves_made < 1:
        time.sleep(0.05)
    controller.stop()

    learned_events = [e for e in events if e.kind == "learned_tile"]
    assert learned_events, "expected the new tier to be learned live, not just detected as unknown"
    assert controller.state == PlayState.STOPPED
    assert all(e.kind != "unknown_tile" for e in events), "should never have needed to pause for the user"
    assert 3 in tile_learner.recognizer.templates, "the newly-learned tier should now be a known template"


class _NoisyCaptureBackend:
    """Always returns a board full of distinct, never-repeating noise -- standing in for a
    misaligned/drifted ROI that has landed on dynamic content (an ad, a rotating thumbnail)
    instead of the actual game board. Every cell looks like a brand-new, never-before-seen
    "sprite" on every single read."""

    def __init__(self) -> None:
        self._counter = 0

    def grab(self, region: CaptureRegion) -> np.ndarray:
        rng = np.random.default_rng(self._counter)
        self._counter += 1
        frame = rng.integers(40, 220, (_CELL_PX * 4, _CELL_PX * 4, 3), dtype=np.uint8)
        # Give each cell a bit of internal texture so is_empty_cell() doesn't treat it as
        # background -- matching a real ad/thumbnail image, not a flat color.
        for r in range(4):
            for c in range(4):
                y0, x0 = r * _CELL_PX, c * _CELL_PX
                frame[y0 + 10 : y0 + 20, x0 + 10 : x0 + 60] = rng.integers(0, 255, 3, dtype=np.uint8)
        return frame

    def close(self) -> None:
        pass


def test_play_loop_stops_learning_after_implausibly_many_new_tiles() -> None:
    """Regression test for a real crash report: a resumed profile whose ROI had drifted onto
    dynamic content (visually, an ad banner) caused the recognizer to see a flood of distinct
    "new" sprites. Before this fix, the learner minted a template for every single one with no
    upper bound, eventually minting a tier above core.board.MAX_TIER (15) -- which
    core.board.set_cell() rejects with ValueError, crashing the play loop entirely (the user
    saw "The play loop hit an unexpected error and stopped" plus a HUD showing tiers up to
    32768 after only two real moves). The fix caps both the absolute tier ceiling and the
    number of new tiles a single observation may learn, and treats exceeding the latter as an
    unresolvable anomaly rather than something to keep guessing at.
    """
    capture = _NoisyCaptureBackend()
    input_backend = FakeInputBackend(FakeGame(seed=3))  # never actually used (loop pauses first)
    tile_learner = _build_tile_learner(known_tiers={1, 2})
    roi = Roi(left=0, top=0, width=_CELL_PX * 4, height=_CELL_PX * 4)
    config = AppConfig(settle_timeout_ms=200.0, settle_frames=1)

    events = []
    controller = PlayController(
        capture_backend=capture,
        input_backend=input_backend,
        tile_learner=tile_learner,
        roi=roi,
        app_config=config,
        on_event=events.append,
    )

    controller.start()
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline and controller.state != PlayState.PAUSED_RECOGNITION_ANOMALY:
        time.sleep(0.05)
    controller.stop()

    assert controller.state == PlayState.STOPPED  # stop() always wins, but it got there via...
    assert controller.had_recognition_anomaly is True
    # No exception should ever have escaped the loop -- the whole point of the cap.
    assert all(e.kind != "error" for e in events)
    # The tier ceiling and per-observation cap must both have held: never above MAX_TIER, and
    # only a couple of tiles actually minted despite dozens of "new" sprites flooding in.
    from core.board import MAX_TIER

    learned_tiers = sorted(tile_learner.recognizer.templates.keys())
    assert all(t <= MAX_TIER for t in learned_tiers)
    assert len(learned_tiers) <= 2 + len({1, 2}), f"learned too many tiles before the anomaly guard tripped: {learned_tiers}"
