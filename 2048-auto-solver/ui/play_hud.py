"""The always-on-top play HUD: recognized grid, chosen move, timing, and Pause/Stop.

A frameless, translucent, always-on-top panel so it stays visible over the game window
without stealing its focus -- exactly the kind of window PySide6 was chosen for (see the
project README). Low-confidence cells (only ever seen transiently, since the play loop pauses
outright below the confidence threshold rather than rendering a guess) are highlighted so a
user watching the HUD can see why a pause happened.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.play_loop import PlayEvent, PlayState

_TIER_BACKGROUND = QColor(50, 50, 60)
_LOW_CONFIDENCE_BACKGROUND = QColor(180, 60, 60)
_EMPTY_BACKGROUND = QColor(35, 35, 42)


class PlayHud(QWidget):
    """Compact, frameless, always-on-top HUD. Emits no signals itself -- callers connect the
    Pause and Stop buttons' ``clicked`` signals directly (see ``ui/main_window.py``)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            parent,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setWindowOpacity(0.96)
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(6)

        self._grid_labels: list[list[QLabel]] = []
        grid_frame = QFrame(self)
        grid_layout = QGridLayout(grid_frame)
        grid_layout.setSpacing(3)
        for row in range(4):
            label_row: list[QLabel] = []
            for col in range(4):
                cell = QLabel("", grid_frame)
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cell.setFixedSize(40, 40)
                cell.setFont(QFont("Sans Serif", 11, QFont.Weight.Bold))
                cell.setAutoFillBackground(True)
                _set_label_background(cell, _EMPTY_BACKGROUND)
                grid_layout.addWidget(cell, row, col)
                label_row.append(cell)
            self._grid_labels.append(label_row)
        outer.addWidget(grid_frame)

        self._move_label = QLabel("Move: -", self)
        self._depth_label = QLabel("Depth: - | Decision: - ms", self)
        self._stats_label = QLabel("0 moves | 0.0 moves/sec", self)
        for label in (self._move_label, self._depth_label, self._stats_label):
            label.setStyleSheet("color: white;")
            outer.addWidget(label)

        self._message_label = QLabel("", self)
        self._message_label.setWordWrap(True)
        self._message_label.setStyleSheet("color: #ffcc66;")
        outer.addWidget(self._message_label)

        buttons = QHBoxLayout()
        self.pause_button = QPushButton("Pause", self)
        self.stop_button = QPushButton("Stop", self)
        for button in (self.pause_button, self.stop_button):
            button.setMinimumHeight(36)
        buttons.addWidget(self.pause_button)
        buttons.addWidget(self.stop_button)
        outer.addLayout(buttons)

        self.setStyleSheet("background-color: rgba(20,20,26,235); border-radius: 8px;")

    @Slot(object)
    def on_play_event(self, event: PlayEvent) -> None:
        """Update every element of the HUD from one :class:`app.play_loop.PlayEvent`."""
        if event.grid is not None:
            self._update_grid(event.grid, event.low_confidence_cells)

        if event.chosen_move is not None:
            self._move_label.setText(f"Move: {event.chosen_move.value.upper()}")

        self._depth_label.setText(
            f"Depth: {event.stats.last_search_depth} | Decision: {event.stats.last_decision_ms:.0f} ms"
        )
        self._stats_label.setText(
            f"{event.stats.moves_made} moves | {event.stats.moves_per_second:.1f} moves/sec"
        )
        self._message_label.setText(event.message)

        self.pause_button.setText("Resume" if event.state is PlayState.PAUSED else "Pause")

    def _update_grid(self, grid: list[list[int]], low_confidence_cells: tuple[tuple[int, int], ...]) -> None:
        low_confidence = set(low_confidence_cells)
        for row in range(4):
            for col in range(4):
                tier = grid[row][col]
                label = self._grid_labels[row][col]
                if (row, col) in low_confidence:
                    _set_label_background(label, _LOW_CONFIDENCE_BACKGROUND)
                    label.setText("?")
                elif tier == 0:
                    _set_label_background(label, _EMPTY_BACKGROUND)
                    label.setText("")
                else:
                    _set_label_background(label, _TIER_BACKGROUND)
                    label.setText(str(1 << tier))


def _set_label_background(label: QLabel, color: QColor) -> None:
    palette = label.palette()
    palette.setColor(label.backgroundRole(), color)
    label.setPalette(palette)
    label.setStyleSheet(f"color: white; background-color: {color.name()}; border-radius: 4px;")
