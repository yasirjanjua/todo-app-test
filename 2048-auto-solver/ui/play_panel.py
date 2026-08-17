"""The in-window play panel: recognized grid, chosen move, timing, and Pause/Stop.

This used to be ``PlayHud``, a separate always-on-top floating window positioned independently
of the main app window. Direct user feedback (with a screenshot) showed exactly what that
design risked: the floating window landing directly on top of several board tiles. Since screen
capture reads the literal compositor output, that overlap wasn't just visually distracting --
the recognizer was reading the floating window's own buttons and stats as if they were tiles,
which is what a good number of "recognition anomaly" / "board went out of sync" reports across
many sessions traced back to.

This panel now lives inside MainWindow's own central stack instead (see
``ui/wizard_arrange_screen.py`` for the first-run step that gets the user's *game* window and
this app's window laid out side by side, so this window is never on top of the game either). It
can never overlap the board because it isn't a separate window at all.
"""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from app.play_loop import PlayEvent, PlayState
from ui.grid_view import TileGridWidget


class PlayPanel(QWidget):
    """Embedded (not a top-level window) play-status panel. Emits no signals itself --
    callers connect the Pause and Stop buttons' ``clicked`` signals directly (see
    ``ui/main_window.py``)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        self._grid_widget = TileGridWidget(cell_size=48, parent=self)
        outer.addWidget(self._grid_widget)

        self._move_label = QLabel("Move: -", self)
        self._depth_label = QLabel("Depth: - | Decision: - ms", self)
        self._stats_label = QLabel("0 moves | 0.0 moves/sec", self)
        for label in (self._move_label, self._depth_label, self._stats_label):
            outer.addWidget(label)

        self._message_label = QLabel("", self)
        self._message_label.setWordWrap(True)
        self._message_label.setStyleSheet("color: #a15c00; font-weight: bold;")
        outer.addWidget(self._message_label)

        buttons = QHBoxLayout()
        self.pause_button = QPushButton("Pause", self)
        self.stop_button = QPushButton("Stop", self)
        self.save_snapshot_button = QPushButton("Save Snapshot", self)
        for button in (self.pause_button, self.stop_button, self.save_snapshot_button):
            button.setMinimumHeight(36)
        buttons.addWidget(self.pause_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.save_snapshot_button)
        outer.addLayout(buttons)

    def on_play_event(self, event: PlayEvent) -> None:
        """Update every element of the panel from one :class:`app.play_loop.PlayEvent`."""
        if event.grid is not None:
            self._grid_widget.update_grid(event.grid, low_confidence_cells=frozenset(event.low_confidence_cells))

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
