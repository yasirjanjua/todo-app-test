"""Shared 4x4 tile-grid mini-display.

Used by both the play HUD and the tile-learning wizard page so "what does the app currently
think the board looks like" always renders the same way: an actual spatial grid of recognized
tiles, not a scrolling text trace. Direct user feedback on the tile-learning page was that a
list of "Learned a new tile (tier N)" toasts gave no sense of where on the board anything was,
or what the board looks like as a whole right now -- this widget is the fix, showing all 16
cells at once, each labeled with what's currently recognized there (or flagged unknown).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QWidget

_TIER_BACKGROUND = QColor(50, 50, 60)
_LOW_CONFIDENCE_BACKGROUND = QColor(180, 60, 60)
_UNKNOWN_BACKGROUND = QColor(150, 100, 20)
_EMPTY_BACKGROUND = QColor(35, 35, 42)


class TileGridWidget(QFrame):
    """A 4x4 grid of labeled cells: tile values (2/4/8/...), empty cells, and
    unrecognized/low-confidence cells, all visible at a glance in their actual board position.
    """

    def __init__(self, cell_size: int = 40, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QGridLayout(self)
        layout.setSpacing(3)
        self._labels: list[list[QLabel]] = []
        for row in range(4):
            label_row: list[QLabel] = []
            for col in range(4):
                cell = QLabel("", self)
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cell.setFixedSize(cell_size, cell_size)
                cell.setFont(QFont("Sans Serif", max(8, cell_size // 4), QFont.Weight.Bold))
                cell.setAutoFillBackground(True)
                _set_label_background(cell, _EMPTY_BACKGROUND)
                layout.addWidget(cell, row, col)
                label_row.append(cell)
            self._labels.append(label_row)

    def update_grid(
        self,
        grid: list[list[int]],
        unknown_cells: frozenset[tuple[int, int]] = frozenset(),
        low_confidence_cells: frozenset[tuple[int, int]] = frozenset(),
    ) -> None:
        """``grid[row][col]`` is a tier (0 = empty); tiers in ``unknown_cells`` render as
        "new?" regardless of the (meaningless, 0) value in ``grid`` for that cell."""
        for row in range(4):
            for col in range(4):
                label = self._labels[row][col]
                if (row, col) in low_confidence_cells:
                    _set_label_background(label, _LOW_CONFIDENCE_BACKGROUND)
                    label.setText("?")
                elif (row, col) in unknown_cells:
                    _set_label_background(label, _UNKNOWN_BACKGROUND)
                    label.setText("new?")
                else:
                    tier = grid[row][col]
                    if tier == 0:
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
