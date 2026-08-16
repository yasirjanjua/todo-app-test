"""Tile learning: "start a new game" is the only instruction the user ever gets.

Polls the calibrated ROI at a modest rate and feeds frames through ``vision.tile_learning``.
Tier 1 and tier 3+ appear as silent toasts; tier 2 pauses for one explicit confirmation (the
90/10 spawn ambiguity described in the spec). A "started mid-game" escape hatch runs the
frequency-ranking fallback instead.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QListWidget, QPushButton, QVBoxLayout, QWidget

from backends.capture.base import CaptureBackend
from vision.capture import Roi, capture_board, flatten_cells, split_cells
from vision.tile_learning import LearningEvent, TileLearner

logger = logging.getLogger(__name__)

_POLL_INTERVAL_MS = 200


class TileLearningPage(QWidget):
    """Drives a :class:`TileLearner` from live (or injected, for tests) capture polling."""

    learning_finished = Signal()

    def __init__(
        self,
        capture_backend: CaptureBackend,
        roi: Roi,
        learner: TileLearner,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._capture = capture_backend
        self._roi = roi
        self._learner = learner
        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._poll_once)
        self._mid_game_mode = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        heading = QLabel(
            "<h2>Teach me the tiles</h2><p>Start a new game in the app, then click the button "
            "below. I'll learn the tile pictures as they appear -- no typing required.</p>",
            self,
        )
        heading.setWordWrap(True)
        layout.addWidget(heading)

        self._start_button = QPushButton("I started a new game", self)
        self._start_button.clicked.connect(self._on_start_new_game)
        layout.addWidget(self._start_button)

        self._mid_game_button = QPushButton("Actually, I'm already mid-game", self)
        self._mid_game_button.clicked.connect(self._on_start_mid_game)
        layout.addWidget(self._mid_game_button)

        self._toast_list = QListWidget(self)
        layout.addWidget(self._toast_list)

        confirm_row = QHBoxLayout()
        self._confirm_label = QLabel("", self)
        confirm_row.addWidget(self._confirm_label)
        self._confirm_yes_button = QPushButton("Yes, that's a new tile", self)
        self._confirm_yes_button.setVisible(False)
        self._confirm_yes_button.clicked.connect(lambda: self._resolve_tier2(True))
        confirm_row.addWidget(self._confirm_yes_button)
        self._confirm_no_button = QPushButton("No, that's a misread", self)
        self._confirm_no_button.setVisible(False)
        self._confirm_no_button.clicked.connect(lambda: self._resolve_tier2(False))
        confirm_row.addWidget(self._confirm_no_button)
        layout.addLayout(confirm_row)

        self._finish_button = QPushButton("Done learning tiles", self)
        self._finish_button.setEnabled(False)
        self._finish_button.clicked.connect(self._on_finish)
        layout.addWidget(self._finish_button)

    def _on_start_new_game(self) -> None:
        self._learner.start_new_game()
        self._mid_game_mode = False
        self._toast_list.clear()
        self._timer.start()

    def _on_start_mid_game(self) -> None:
        self._learner.start_mid_game_fallback()
        self._mid_game_mode = True
        self._toast_list.clear()
        self._timer.start()
        # Give the fallback a few seconds of frames to observe before ranking.
        QTimer.singleShot(4000, self._finish_mid_game_observation)

    def _poll_once(self) -> None:
        try:
            frame = capture_board(self._capture, self._roi)
        except Exception:  # noqa: BLE001 - a transient capture failure shouldn't kill the poller
            logger.warning("Tile-learning capture failed; will retry next tick.", exc_info=True)
            return
        cells = flatten_cells(split_cells(frame))
        events = self._learner.observe(cells)
        for event in events:
            self._handle_event(event)

    def _handle_event(self, event: LearningEvent) -> None:
        if event.kind == "learned_tile":
            self._toast_list.addItem(f"Learned a new tile (tier {event.tier}).")
            self._finish_button.setEnabled(True)
        elif event.kind == "await_confirmation":
            self._timer.stop()
            self._confirm_label.setText("Is this a brand-new tile?")
            self._confirm_yes_button.setVisible(True)
            self._confirm_no_button.setVisible(True)

    def _resolve_tier2(self, accept: bool) -> None:
        event = self._learner.confirm_tier2(accept)
        if event is not None:
            self._handle_event(event)
        self._confirm_label.setText("")
        self._confirm_yes_button.setVisible(False)
        self._confirm_no_button.setVisible(False)
        self._timer.start()

    def _finish_mid_game_observation(self) -> None:
        self._timer.stop()
        events = self._learner.finish_mid_game_ranking()
        for event in events:
            self._handle_event(event)
        self._confirm_label.setText("Does that ranking look right (lowest tile first)?")
        self._confirm_yes_button.setVisible(True)
        self._confirm_yes_button.setText("Yes, that's right")
        self._confirm_no_button.setVisible(False)
        self._confirm_yes_button.clicked.disconnect()
        self._confirm_yes_button.clicked.connect(self._acknowledge_mid_game_ranking)

    def _acknowledge_mid_game_ranking(self) -> None:
        self._confirm_label.setText("")
        self._confirm_yes_button.setVisible(False)
        self._finish_button.setEnabled(True)

    def _on_finish(self) -> None:
        self._timer.stop()
        self.learning_finished.emit()
