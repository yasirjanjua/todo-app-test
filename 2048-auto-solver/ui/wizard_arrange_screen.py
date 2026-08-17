"""First-run screen-arrangement step: the game gets ~70% of the screen, this app gets the rest.

Direct user feedback, with a screenshot, showed the previous floating always-on-top HUD landing
directly on top of several board tiles -- and since screen capture reads the actual compositor
output, that overlap wasn't just visually distracting: the recognizer was reading the HUD's own
buttons and stats as if they were tiles, which is what a good number of "recognition anomaly" /
"board went out of sync" reports across many sessions traced back to. Merging the play status
into the main window (see ``ui/play_panel.py``) removes the one floating window that could land
on the board, but that only holds if this window itself never sits on top of the game either --
this step is what gets the user's screen laid out side by side, once, up front, rather than
leaving it to chance (or worse, to a user noticing it's broken after the fact).

Shown once per installation -- see ``app.profile.ProfileStore.has_completed_screen_setup`` --
not on every launch, matching the "second run must be zero-setup" rule the rest of the wizard
already follows.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class ArrangeScreenPage(QWidget):
    arranged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        heading = QLabel(
            "<h2>Set up your screen</h2>"
            "<p>This app needs to sit <b>next to</b> the game, never on top of it. If any of "
            "its own windows ever overlap the game board, it can end up reading its own text "
            "and buttons as if they were tiles.</p>"
            "<p>Resize your browser (or game app) to take up roughly the <b>left 70%</b> of "
            "your screen, then use a button below to move this window into the remaining 30% "
            "-- side by side, with no overlap. You only need to do this once.</p>",
            self,
        )
        heading.setWordWrap(True)
        layout.addWidget(heading)

        buttons = QHBoxLayout()
        self._dock_right_button = QPushButton("Move this window to the right 30%", self)
        self._dock_right_button.clicked.connect(lambda: self._dock(right=True))
        buttons.addWidget(self._dock_right_button)
        self._dock_left_button = QPushButton("Move this window to the left 30%", self)
        self._dock_left_button.clicked.connect(lambda: self._dock(right=False))
        buttons.addWidget(self._dock_left_button)
        layout.addLayout(buttons)

        continue_button = QPushButton("Done -- my screen is set up like this", self)
        continue_button.setMinimumHeight(48)
        continue_button.clicked.connect(self.arranged.emit)
        layout.addWidget(continue_button)

    def _dock(self, right: bool) -> None:
        window = self.window()
        screen = window.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        width = max(320, round(available.width() * 0.3))
        left = available.right() - width + 1 if right else available.left()
        window.setGeometry(left, available.top(), width, available.height())
