"""Tile learning: "start a new game" is the only instruction the user ever gets.

Polls the calibrated ROI at a modest rate and feeds frames through ``vision.tile_learning``.
Tier 1 and tier 3+ appear as silent toasts; tier 2 pauses for one explicit confirmation (the
90/10 spawn ambiguity described in the spec). A "started mid-game" escape hatch runs the
frequency-ranking fallback instead.

A live thumbnail of exactly what's being captured is shown throughout (see
``_preview_label``): real-world feedback showed that a wrong or drifted grid region produces a
confusing scroll of "Learned a new tile" toasts with no way to tell why, when the actual
problem -- the ROI capturing the wrong part of the screen entirely -- would have been obvious
at a glance from the picture itself.

Every "Learned a new tile" entry now also shows a thumbnail of the actual crop that was
learned, not just a tier number -- direct user feedback was that a text-only toast gave no way
to tell if what got learned was right, leaving "blindly accept" as the only option. The
mid-game ranking confirmation (the one place a *whole batch* of tiles gets accepted at once
from an automatic frequency guess, rather than one at a time) also gets a real "No, let me redo
it" path that discards the batch and restarts observation, instead of only ever being able to
accept it.

A live recognized-board grid (``ui.grid_view.TileGridWidget``, shared with the play HUD) is now
the primary display, updated every poll: 16 cells in their actual board positions, each showing
what's currently recognized there (or "new?" if it doesn't match anything yet). Direct user
feedback was that the toast list alone -- text traces with no spatial context -- was
unintuitive; seeing the whole board at once, the way the play HUD already shows it during play,
makes a misaligned grid or a misread tile obvious immediately instead of only inferable from a
scrolling log.
"""

from __future__ import annotations

import logging

import cv2
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from backends.capture.base import CaptureBackend
from ui.grid_view import TileGridWidget
from vision.capture import Roi, capture_board, flatten_cells, split_cells
from vision.tile_learning import LearningEvent, TileLearner

logger = logging.getLogger(__name__)

_POLL_INTERVAL_MS = 200
_PREVIEW_MAX_DIM = 220
_THUMBNAIL_DIM = 48


def _bgr_to_pixmap(frame) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


def _crop_to_icon(crop) -> QIcon:
    pixmap = _bgr_to_pixmap(crop).scaled(
        _THUMBNAIL_DIM,
        _THUMBNAIL_DIM,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    return QIcon(pixmap)


class TileLearningPage(QWidget):
    """Drives a :class:`TileLearner` from live (or injected, for tests) capture polling."""

    learning_finished = Signal()
    recalibrate_requested = Signal()

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
        # Tiers learned so far in the current mid-game ranking batch, so a rejection can undo
        # exactly those templates (and only those) rather than guessing what to discard.
        self._pending_mid_game_tiers: list[int] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        heading = QLabel(
            "<h2>Teach me the tiles</h2><p>Start a new game in the app, then click the button "
            "below. I'll learn the tile pictures as they appear -- no typing required. Each "
            "learned tile below shows the actual picture I saw, so you can check it's right.</p>",
            self,
        )
        heading.setWordWrap(True)
        layout.addWidget(heading)

        preview_row = QHBoxLayout()

        preview_column = QVBoxLayout()
        preview_column.addWidget(QLabel("What I'm currently looking at:", self))
        self._preview_label = QLabel(self)
        self._preview_label.setFixedSize(_PREVIEW_MAX_DIM, _PREVIEW_MAX_DIM)
        self._preview_label.setStyleSheet("border: 1px solid #555;")
        self._preview_label.setScaledContents(False)
        preview_column.addWidget(self._preview_label)
        preview_row.addLayout(preview_column)

        grid_column = QVBoxLayout()
        grid_column.addWidget(QLabel("What I currently recognize, cell by cell:", self))
        self._recognized_grid = TileGridWidget(cell_size=44, parent=self)
        grid_column.addWidget(self._recognized_grid)
        preview_row.addLayout(grid_column)

        preview_row.addStretch(1)
        layout.addLayout(preview_row)

        self._start_button = QPushButton("I started a new game", self)
        self._start_button.clicked.connect(self._on_start_new_game)
        layout.addWidget(self._start_button)

        self._mid_game_button = QPushButton("Actually, I'm already mid-game", self)
        self._mid_game_button.clicked.connect(self._on_start_mid_game)
        layout.addWidget(self._mid_game_button)

        self._toast_list = QListWidget(self)
        layout.addWidget(self._toast_list)

        self._anomaly_label = QLabel("", self)
        self._anomaly_label.setWordWrap(True)
        self._anomaly_label.setStyleSheet("color: #a15c00; font-weight: bold;")
        self._anomaly_label.setVisible(False)
        layout.addWidget(self._anomaly_label)

        self._fix_grid_button = QPushButton("Fix the grid region", self)
        self._fix_grid_button.setVisible(False)
        self._fix_grid_button.clicked.connect(self.recalibrate_requested.emit)
        layout.addWidget(self._fix_grid_button)

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
        self._pending_mid_game_tiers = []
        self._toast_list.clear()
        self._reset_anomaly_state()
        self._timer.start()

    def _on_start_mid_game(self) -> None:
        self._learner.start_mid_game_fallback()
        self._mid_game_mode = True
        self._pending_mid_game_tiers = []
        self._toast_list.clear()
        self._reset_anomaly_state()
        self._timer.start()
        # Give the fallback a few seconds of frames to observe before ranking.
        QTimer.singleShot(4000, self._finish_mid_game_observation)

    def _reset_anomaly_state(self) -> None:
        self._anomaly_label.setVisible(False)
        self._fix_grid_button.setVisible(False)

    def _poll_once(self) -> None:
        try:
            frame = capture_board(self._capture, self._roi)
        except Exception:  # noqa: BLE001 - a transient capture failure shouldn't kill the poller
            logger.warning("Tile-learning capture failed; will retry next tick.", exc_info=True)
            return
        self._update_preview(frame)
        cells = flatten_cells(split_cells(frame))
        self._update_recognized_grid(cells)
        events = self._learner.observe(cells)
        for event in events:
            self._handle_event(event)

    def _update_preview(self, frame) -> None:
        pixmap = _bgr_to_pixmap(frame)
        self._preview_label.setPixmap(
            pixmap.scaled(self._preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        )

    def _update_recognized_grid(self, cells: list) -> None:
        # A plain classify -- read-only against whatever the recognizer knows *so far* -- so
        # this never itself teaches anything; it only reflects the learner's current state,
        # the same way the play HUD reflects PlayController's.
        results = self._learner.recognizer.classify_board(cells)
        grid = [[0] * 4 for _ in range(4)]
        unknown: set[tuple[int, int]] = set()
        for i, result in enumerate(results):
            row, col = i // 4, i % 4
            if result.is_empty:
                continue
            if result.tier is None:
                unknown.add((row, col))
            else:
                grid[row][col] = result.tier
        self._recognized_grid.update_grid(grid, unknown_cells=frozenset(unknown))

    def _handle_event(self, event: LearningEvent) -> None:
        if event.kind == "learned_tile":
            item = QListWidgetItem(f"Learned a new tile (tier {event.tier}).")
            if event.crop is not None:
                item.setIcon(_crop_to_icon(event.crop))
            self._toast_list.addItem(item)
            self._finish_button.setEnabled(True)
            if self._mid_game_mode and event.tier is not None:
                self._pending_mid_game_tiers.append(event.tier)
        elif event.kind == "await_confirmation":
            self._timer.stop()
            self._confirm_label.setText("Is this a brand-new tile?")
            self._confirm_yes_button.setVisible(True)
            self._confirm_no_button.setVisible(True)
        elif event.kind == "anomaly":
            self._timer.stop()
            self._finish_button.setEnabled(False)
            self._anomaly_label.setText(
                "I'm seeing far more different tiles than a real board should have right now. "
                "This almost always means the grid box isn't actually lined up with the board "
                "-- click below to go back and fix it."
            )
            self._anomaly_label.setVisible(True)
            self._fix_grid_button.setVisible(True)

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
        if any(event.kind == "anomaly" for event in events):
            self._pending_mid_game_tiers = []
            return
        if not self._pending_mid_game_tiers:
            return  # nothing observed to rank; nothing to confirm either
        self._confirm_label.setText(
            "Here's what I learned, lowest tile first (pictured above, in order). Does that "
            "look right?"
        )
        self._confirm_yes_button.setVisible(True)
        self._confirm_yes_button.setText("Yes, that's right")
        self._confirm_yes_button.clicked.disconnect()
        self._confirm_yes_button.clicked.connect(self._acknowledge_mid_game_ranking)
        self._confirm_no_button.setVisible(True)
        self._confirm_no_button.setText("No, let me redo it")
        self._confirm_no_button.clicked.disconnect()
        self._confirm_no_button.clicked.connect(self._reject_mid_game_ranking)

    def _restore_tier2_confirmation_handlers(self) -> None:
        """The confirm buttons are shared between the tier-2 single-tile confirmation and the
        mid-game ranking's whole-batch confirmation; whichever one last reconfigured them must
        put the tier-2 wiring back so a later real tier-2 confirmation still works."""
        self._confirm_yes_button.setText("Yes, that's a new tile")
        self._confirm_yes_button.clicked.disconnect()
        self._confirm_yes_button.clicked.connect(lambda: self._resolve_tier2(True))
        self._confirm_no_button.setText("No, that's a misread")
        self._confirm_no_button.clicked.disconnect()
        self._confirm_no_button.clicked.connect(lambda: self._resolve_tier2(False))

    def _acknowledge_mid_game_ranking(self) -> None:
        self._confirm_label.setText("")
        self._confirm_yes_button.setVisible(False)
        self._confirm_no_button.setVisible(False)
        self._restore_tier2_confirmation_handlers()
        self._pending_mid_game_tiers = []
        self._finish_button.setEnabled(True)

    def _reject_mid_game_ranking(self) -> None:
        """The user says the ranking is wrong -- undo exactly the templates this batch minted
        (not a full reset) and restart mid-game observation from scratch, instead of leaving
        "accept it anyway" as the only option."""
        for tier in self._pending_mid_game_tiers:
            self._learner.recognizer.templates.pop(tier, None)
        self._pending_mid_game_tiers = []
        self._confirm_label.setText("")
        self._confirm_yes_button.setVisible(False)
        self._confirm_no_button.setVisible(False)
        self._restore_tier2_confirmation_handlers()
        self._toast_list.clear()
        self._finish_button.setEnabled(False)
        self._learner.start_mid_game_fallback()
        self._timer.start()
        QTimer.singleShot(4000, self._finish_mid_game_observation)

    def _on_finish(self) -> None:
        self._timer.stop()
        self.learning_finished.emit()
