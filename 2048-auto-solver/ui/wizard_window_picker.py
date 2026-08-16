""""Find my game": a visual picker with live thumbnails, not a numbered text list.

Users recognize their game by appearance, not by window title -- this is the single biggest
first-run UX decision in the spec, so this page renders thumbnails, full stop. The best title
match (see ``backends/window_enum.py``) is pre-selected.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from backends.window_enum import WindowInfo, list_candidate_windows

logger = logging.getLogger(__name__)

_THUMB_SIZE = QSize(220, 160)


def _bgr_to_pixmap(frame: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


class WindowPickerPage(QWidget):
    """Enumerates candidate windows and lets the user pick theirs."""

    window_selected = Signal(object)  # emits a WindowInfo

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._windows: list[WindowInfo] = []
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        heading = QLabel("<h2>Find my game</h2><p>Click the window that has your 2048 game open.</p>", self)
        heading.setWordWrap(True)
        layout.addWidget(heading)

        self._list = QListWidget(self)
        self._list.setViewMode(QListWidget.ViewMode.IconMode)
        self._list.setIconSize(_THUMB_SIZE)
        self._list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._list.setSpacing(12)
        self._list.setMovement(QListWidget.Movement.Static)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._list)

        buttons_row = QVBoxLayout()
        self._refresh_button = QPushButton("Refresh", self)
        self._refresh_button.clicked.connect(self.refresh)
        buttons_row.addWidget(self._refresh_button)

        self._continue_button = QPushButton("This is my game", self)
        self._continue_button.setEnabled(False)
        self._continue_button.clicked.connect(self._on_continue)
        buttons_row.addWidget(self._continue_button)
        layout.addLayout(buttons_row)

    def refresh(self) -> None:
        self._list.clear()
        try:
            self._windows = list_candidate_windows()
        except Exception:  # noqa: BLE001 - enumeration failing must not crash the wizard
            logger.error("Failed to enumerate windows.", exc_info=True)
            self._windows = []

        for window in self._windows:
            item = QListWidgetItem(window.title)
            if window.thumbnail is not None:
                item.setIcon(QIcon(_bgr_to_pixmap(window.thumbnail)))
            self._list.addItem(item)

        if self._windows:
            # Pre-select the best title match (list is already sorted best-first).
            self._list.setCurrentRow(0)

    def _on_selection_changed(self) -> None:
        self._continue_button.setEnabled(self._list.currentRow() >= 0)

    def _on_continue(self) -> None:
        row = self._list.currentRow()
        if 0 <= row < len(self._windows):
            self.window_selected.emit(self._windows[row])
