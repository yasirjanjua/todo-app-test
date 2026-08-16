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
from core.board import Move, empty_cells, set_cell
from vision.capture import Roi
from vision.recognition import TileRecognizer, make_template

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


def _build_recognizer() -> TileRecognizer:
    from core.board import set_cell
    from vision.capture import flatten_cells, split_cells

    # Build templates from the *inset* crop (matching how the play loop crops cells at
    # recognition time, per vision/capture.py's 15% inset), not the raw uncropped sprite --
    # comparing an inset crop against a non-inset template would otherwise systematically
    # under-report confidence.
    recognizer = TileRecognizer(confidence_threshold=0.6)
    for tier in _TIER_COLORS:
        board = set_cell(0, 0, tier)
        frame = _render(board)
        cell_crop = flatten_cells(split_cells(frame, inset_ratio=0.15))[0]
        recognizer.add_template(make_template(tier, cell_crop))
    return recognizer


def test_play_loop_plays_moves_and_stops_cleanly() -> None:
    game = FakeGame(seed=42)
    capture = FakeCaptureBackend(game)
    input_backend = FakeInputBackend(game)
    recognizer = _build_recognizer()
    roi = Roi(left=0, top=0, width=_CELL_PX * 4, height=_CELL_PX * 4)
    config = AppConfig(settle_timeout_ms=200.0, settle_frames=1)

    events = []
    controller = PlayController(
        capture_backend=capture,
        input_backend=input_backend,
        recognizer=recognizer,
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
    recognizer = _build_recognizer()
    roi = Roi(left=0, top=0, width=_CELL_PX * 4, height=_CELL_PX * 4)
    config = AppConfig(dry_run=True, settle_timeout_ms=200.0, settle_frames=1)

    events = []
    controller = PlayController(
        capture_backend=capture,
        input_backend=input_backend,
        recognizer=recognizer,
        roi=roi,
        app_config=config,
        on_event=events.append,
    )
    controller.start()
    time.sleep(0.5)
    controller.stop()

    assert input_backend.taps == []
    assert any(e.kind == "decision" for e in events)
