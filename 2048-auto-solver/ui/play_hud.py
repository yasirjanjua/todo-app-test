"""The always-on-top play HUD: recognized grid, chosen move, timing, and Pause/Stop.

A frameless, translucent, always-on-top panel so it stays visible over the game window
without stealing its focus -- exactly the kind of window PySide6 was chosen for (see the
project README). Low-confidence cells (only ever seen transiently, since the play loop pauses
outright below the confidence threshold rather than rendering a guess) are highlighted so a
user watching the HUD can see why a pause happened.

It appears as a separate floating window (rather than living inside the main wizard window)
specifically so it can stay visible on top of the *game's* window, which is typically somewhere
else on screen entirely -- that only works as a standalone always-on-top panel. "Without
stealing its focus" used to be aspirational rather than actually enforced: showing this window
had nothing telling the OS it must never receive keyboard focus, and a real report (all four
candidate moves failing to change the board for several seconds, over and over, immediately
after Start) traced back to exactly that -- show() briefly handing OS keyboard focus to this
window (or our own app generally) instead of the game, so the injected keystrokes were landing
on nothing useful. WindowDoesNotAcceptFocus is the fix: it tells the OS this window must never
become key/focused, independent of z-order or visibility.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from app.play_loop import PlayEvent, PlayState
from ui.grid_view import TileGridWidget


class PlayHud(QWidget):
    """Compact, frameless, always-on-top HUD. Emits no signals itself -- callers connect the
    Pause and Stop buttons' ``clicked`` signals directly (see ``ui/main_window.py``)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            parent,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setWindowOpacity(0.96)
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(6)

        self._grid_widget = TileGridWidget(cell_size=40, parent=self)
        outer.addWidget(self._grid_widget)

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
        # Mirrors the "S" global hotkey via a mouse click, since global hotkeys are off by
        # default on macOS (see AppConfig.enable_macos_global_hotkeys) -- this keeps the
        # snapshot feature reachable without depending on pynput's listener at all.
        self.save_snapshot_button = QPushButton("Save Snapshot", self)
        for button in (self.pause_button, self.stop_button, self.save_snapshot_button):
            button.setMinimumHeight(36)
        buttons.addWidget(self.pause_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.save_snapshot_button)
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
        self._grid_widget.update_grid(grid, low_confidence_cells=frozenset(low_confidence_cells))
