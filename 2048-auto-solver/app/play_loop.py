"""The closed-loop play controller: perceive, decide, act, verify -- repeat.

This is where Part 2 (Engine) and Part 5 (Robustness) of the spec actually get enforced at
runtime:

* Every move is verified by re-reading the board; an unchanged board means the keystroke was
  illegal or dropped, and the controller falls through to the next-ranked move instead of
  resending blindly (Part 2.4, Part 5.3).
* The window's position is re-checked periodically; if it moved, resized, or the profile's
  ROI no longer looks like a board, the loop pauses with a clear reason instead of feeding the
  solver garbage (Part 5.1).
* Pause is instant (checked every loop iteration and inside the animation wait), and
  ``stop()`` always calls ``release_all()`` so no key is left held down (Part 5.5).
* Dry-run mode runs the full perceive/decide pipeline and logs decisions without calling
  ``input_backend.tap_key`` at all (Part 3).
* An unrecognized tile first tries to *learn itself*: tiers above 2 can only ever arise from
  a merge (see ``vision/tile_learning.py``), so a never-seen sprite mid-play is unambiguously
  the next tier and is silently promoted the same way calibration does, without stopping play.
  Only a genuinely ambiguous case (or a learner that still can't resolve it) falls through to
  an actual pause (Part 1.4, Part 2.2).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

import numpy as np

from app.config import AppConfig
from backends.capture.base import CaptureBackend
from backends.input.base import InputBackend, Key
from core.board import Move, from_grid, is_game_over, to_grid
from core.board import move as apply_move
from core.solver import SolverConfig, get_best_move
from vision.capture import Roi, capture_board, flatten_cells, split_cells
from vision.recognition import RecognitionResult
from vision.stability import StabilityConfig, wait_for_stable
from vision.tile_learning import TileLearner

logger = logging.getLogger(__name__)

_MOVE_TO_KEY = {
    Move.UP: Key.UP,
    Move.DOWN: Key.DOWN,
    Move.LEFT: Key.LEFT,
    Move.RIGHT: Key.RIGHT,
}

# 2048's own win condition; play continues past it unless the board also has no legal moves,
# matching how every mainstream implementation behaves ("keep going" after the 2048 tile).
WIN_TIER = 11


class PlayState(Enum):
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    PAUSED_UNKNOWN_TILE = auto()
    PAUSED_WINDOW_MOVED = auto()
    STOPPED = auto()
    GAME_OVER = auto()


@dataclass
class PlayStats:
    moves_made: int = 0
    invalid_move_attempts: int = 0
    total_score: int = 0
    max_tile_tier: int = 0
    started_monotonic: float = field(default_factory=time.monotonic)
    last_decision_ms: float = 0.0
    last_search_depth: int = 0

    @property
    def moves_per_second(self) -> float:
        elapsed = time.monotonic() - self.started_monotonic
        return self.moves_made / elapsed if elapsed > 0 else 0.0


@dataclass
class PlayEvent:
    """One notification for the UI's HUD; see ``ui/play_hud.py`` for consumption."""

    kind: str
    state: PlayState
    stats: PlayStats
    grid: list[list[int]] | None = None
    chosen_move: Move | None = None
    message: str = ""
    low_confidence_cells: tuple[tuple[int, int], ...] = ()
    diagnostic_crop: np.ndarray | None = None


class PlayController:
    """Owns the play loop's background thread and its state transitions."""

    def __init__(
        self,
        capture_backend: CaptureBackend,
        input_backend: InputBackend,
        tile_learner: TileLearner,
        roi: Roi,
        app_config: AppConfig,
        on_event: Callable[[PlayEvent], None] | None = None,
        window_bounds_probe: Callable[[], tuple[int, int, int, int] | None] | None = None,
    ) -> None:
        self._capture = capture_backend
        self._input = input_backend
        # Shared by reference with whoever constructed this controller (typically
        # ui/main_window.py's wizard.data.tile_learner): tiles learned mid-play mutate this
        # same object, so the caller can persist them back to the profile after stop().
        self._learner = tile_learner
        self._roi = roi
        self._config = app_config
        self._on_event = on_event or (lambda event: None)
        # Optional callback returning the calibrated window's *current* (left, top, width,
        # height); used to detect the window having moved or resized (Part 5.1). Left unset in
        # contexts (like tests) that don't have a live window to probe.
        self._window_bounds_probe = window_bounds_probe

        self._solver_config = SolverConfig(
            base_depth=app_config.base_search_depth,
            max_depth=app_config.max_search_depth,
            weights=app_config.heuristic_weights,
        )
        self._stability_config = StabilityConfig(
            settle_frames=app_config.settle_frames, timeout_ms=app_config.settle_timeout_ms
        )

        self._state = PlayState.IDLE
        self._stats = PlayStats()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def state(self) -> PlayState:
        return self._state

    @property
    def stats(self) -> PlayStats:
        return self._stats

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.warning("start() called while the play loop is already running; ignoring.")
            return
        self._stop_event.clear()
        self._pause_event.clear()
        self._stats = PlayStats()
        self._state = PlayState.RUNNING
        self._thread = threading.Thread(target=self._run_loop, name="play-loop", daemon=True)
        self._thread.start()

    def pause(self) -> None:
        """Instant: sets a flag the loop checks before every capture and every keystroke."""
        self._pause_event.set()
        if self._state is PlayState.RUNNING:
            self._state = PlayState.PAUSED
        self._emit("paused")

    def resume(self) -> None:
        if self._state in (PlayState.PAUSED, PlayState.PAUSED_UNKNOWN_TILE, PlayState.PAUSED_WINDOW_MOVED):
            self._pause_event.clear()
            self._state = PlayState.RUNNING
            self._emit("resumed")

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.clear()  # wake a paused loop so it can observe the stop event
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        # Guaranteed cleanup regardless of where in the loop we were: no key is ever left held.
        self._input.release_all()
        self._state = PlayState.STOPPED
        self._emit("stopped")

    def _emit(self, kind: str, **kwargs) -> None:
        self._on_event(PlayEvent(kind=kind, state=self._state, stats=self._stats, **kwargs))

    def _capture_frame(self) -> np.ndarray:
        return capture_board(self._capture, self._roi)

    def capture_debug_frame(self) -> np.ndarray:
        """Grab the current ROI frame on demand, for the "save snapshot" hotkey/debug tooling."""
        return self._capture_frame()

    def _classify_frame(self, frame: np.ndarray) -> tuple[int, list[RecognitionResult]]:
        cells = flatten_cells(split_cells(frame, inset_ratio=self._config.cell_inset_ratio))
        results = self._learner.recognizer.classify_board(cells)
        grid = [[0] * 4 for _ in range(4)]
        for i, result in enumerate(results):
            grid[i // 4][i % 4] = result.tier or 0
        return from_grid(grid), results

    def _try_learn_unknown_tiles(self, frame: np.ndarray) -> bool:
        """Attempt to silently learn any newly-appeared tile tier(s) from ``frame``.

        Returns True if at least one tile was actually learned (meaning the caller should
        re-classify the frame with the now-updated recognizer). Returns False when the
        learner can't resolve it on its own -- e.g. the rare case where the profile's
        calibration was interrupted before the tier-2 confirmation step -- leaving the normal
        pause-and-wait-for-the-user path as the fallback.
        """
        cells = flatten_cells(split_cells(frame, inset_ratio=self._config.cell_inset_ratio))
        events = self._learner.observe(cells)
        learned_any = False
        for event in events:
            if event.kind == "learned_tile":
                learned_any = True
                logger.info("Learned tile tier %d during play.", event.tier)
                self._emit("learned_tile", message=f"Learned a new tile (tier {event.tier}).")
        return learned_any

    def _read_board(self) -> tuple[int, list[RecognitionResult], np.ndarray]:
        frame = self._capture_frame()
        board, results = self._classify_frame(frame)
        return board, results, frame

    def _check_window_moved(self) -> bool:
        if self._window_bounds_probe is None:
            return False
        current = self._window_bounds_probe()
        if current is None:
            logger.warning("Calibrated window is no longer visible; pausing.")
            return True
        left, top, width, height = current
        tolerance = 4  # small OS-level jitter is normal; a real move/resize exceeds this easily
        moved = (
            abs(left - self._roi.left) > tolerance
            or abs(top - self._roi.top) > tolerance
            or abs(width - self._roi.width) > tolerance * 4
            or abs(height - self._roi.height) > tolerance * 4
        )
        return moved

    def _wait_while_paused(self) -> bool:
        """Block while paused. Returns True if a stop was requested while waiting."""
        while self._pause_event.is_set():
            if self._stop_event.wait(timeout=0.1):
                return True
        return False

    def _run_loop(self) -> None:
        window_check_counter = 0
        try:
            while not self._stop_event.is_set():
                if self._wait_while_paused():
                    break

                window_check_counter += 1
                if window_check_counter % 20 == 0 and self._check_window_moved():
                    self._state = PlayState.PAUSED_WINDOW_MOVED
                    self._pause_event.set()
                    self._emit("window_moved", message="The game window moved or resized. Pausing.")
                    continue

                board, results, frame = self._read_board()
                unknown_cells = tuple(
                    (i // 4, i % 4) for i, r in enumerate(results) if r.tier is None
                )
                if unknown_cells:
                    if self._try_learn_unknown_tiles(frame):
                        board, results = self._classify_frame(frame)
                        unknown_cells = tuple(
                            (i // 4, i % 4) for i, r in enumerate(results) if r.tier is None
                        )

                if unknown_cells:
                    # The learner couldn't resolve this on its own -- genuinely ambiguous
                    # (e.g. calibration was stopped before the tier-2 confirmation step) or a
                    # true misread. This is the one case that still needs a human.
                    self._state = PlayState.PAUSED_UNKNOWN_TILE
                    self._pause_event.set()
                    self._emit(
                        "unknown_tile",
                        grid=to_grid(board),
                        message=(
                            "Saw a tile I can't place even after trying to learn it. "
                            "Recalibrate to teach it, then resume."
                        ),
                        low_confidence_cells=unknown_cells,
                    )
                    continue

                decision = get_best_move(board, self._solver_config)
                self._stats.last_decision_ms = 0.0  # measured below
                self._stats.last_search_depth = decision.depth

                if decision.best_move is None:
                    self._finish_game(board)
                    break

                success = self._attempt_move(board, decision.ranked_moves)
                if not success:
                    if is_game_over(board):
                        self._finish_game(board)
                        break
                    # No candidate move changed the board but the board isn't technically
                    # "game over" by our own rules either (e.g. a transient overlay froze
                    # input); log and retry next iteration rather than spinning hot.
                    logger.warning("No move changed the board; will re-read and retry.")
                    time.sleep(0.1)

                self._stats.max_tile_tier = max(self._stats.max_tile_tier, _max_tier(board))
                if self._stats.max_tile_tier >= WIN_TIER:
                    self._emit("win", grid=to_grid(board), message="Reached the 2048 tile!")

        except Exception:  # noqa: BLE001 - the play thread must never die silently
            logger.error("Play loop crashed unexpectedly.", exc_info=True)
            self._emit("error", message="The play loop hit an unexpected error and stopped.")
        finally:
            self._input.release_all()

    def _attempt_move(self, board_before: int, ranked_moves: tuple[tuple[Move, float], ...]) -> bool:
        decision_start = time.monotonic()
        for move, _value in ranked_moves:
            if self._stop_event.is_set() or self._pause_event.is_set():
                return False

            if not self._config.dry_run:
                self._input.tap_key(_MOVE_TO_KEY[move], hold_ms=self._config.key_hold_ms)
                stable = wait_for_stable(self._capture_frame, self._stability_config)
                new_board, _results = self._classify_frame(stable.frame)
            else:
                new_board = board_before  # dry-run never actually changes the board

            self._stats.last_decision_ms = (time.monotonic() - decision_start) * 1000.0

            if self._config.dry_run:
                logger.info("[dry-run] would play %s", move)
                self._emit("decision", grid=to_grid(board_before), chosen_move=move, message="dry-run")
                return True

            if new_board != board_before:
                self._stats.moves_made += 1
                # The score of the move actually played on screen isn't independently OCR'd
                # (see Part 1.4: OCR is an optional confidence booster, never load-bearing);
                # the simulated score for the same move on the pre-move board is used as the
                # displayed running total instead.
                self._stats.total_score += apply_move(board_before, move).score
                self._emit("move", grid=to_grid(new_board), chosen_move=move)
                return True

            logger.info("Move %s produced no board change; trying next-best move.", move)
            self._stats.invalid_move_attempts += 1
            board_before = new_board  # re-read is authoritative; keep verifying against it

        return False

    def _finish_game(self, final_board: int) -> None:
        self._state = PlayState.GAME_OVER
        grid = to_grid(final_board)
        max_tier = _max_tier(final_board)
        self._emit(
            "game_over",
            grid=grid,
            message=f"Game over. Final score: {self._stats.total_score}, max tile tier: {max_tier}.",
        )


def _max_tier(board: int) -> int:
    grid = to_grid(board)
    return max(cell for row in grid for cell in row)
