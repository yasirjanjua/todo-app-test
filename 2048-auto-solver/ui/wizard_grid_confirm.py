"""Grid confirmation: a highlighted overlay on a screenshot, not a raw ``cv2.selectROI`` call.

Auto-detection (``vision/grid_detect.py``) runs first; this page just renders its proposal as
a rectangle over the captured window and asks "Looks right?" / "Let me adjust". The adjust
path is a custom drag-handle widget -- ``cv2.selectROI`` is a developer tool and was
explicitly ruled out by the spec because it looks like one.
"""

from __future__ import annotations

import cv2
import numpy as np

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPaintEvent, QPen, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

_HANDLE_RADIUS = 8
_HANDLE_HIT_RADIUS = 14


def _bgr_to_qimage(frame: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    return QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()


class _GridOverlayWidget(QWidget):
    """Renders the screenshot with a draggable rectangle on top.

    The rectangle is axis-aligned (matching :class:`backends.capture.base.CaptureRegion`);
    each of its four corner handles can be dragged independently, which adjusts the two edges
    that meet at that corner.
    """

    rect_changed = Signal()

    def __init__(self, frame: np.ndarray, initial_rect: QRect, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image = _bgr_to_qimage(frame)
        self._rect = QRect(initial_rect)
        self._dragging_corner: int | None = None  # 0=TL, 1=TR, 2=BL, 3=BR
        self._draggable = False
        self.setMinimumSize(self._image.width(), self._image.height())
        self.setMouseTracking(True)

    def set_draggable(self, draggable: bool) -> None:
        self._draggable = draggable
        self.update()

    def rect(self) -> QRect:
        return QRect(self._rect)

    def set_rect(self, rect: QRect) -> None:
        self._rect = QRect(rect)
        self.update()

    def _corners(self) -> list[QPoint]:
        r = self._rect
        return [r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight()]

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override signature
        painter = QPainter(self)
        painter.drawImage(0, 0, self._image)

        overlay_pen = QPen(QColor(80, 220, 120), 3)
        painter.setPen(overlay_pen)
        painter.setBrush(QColor(80, 220, 120, 40))
        painter.drawRect(self._rect)

        if self._draggable:
            handle_pen = QPen(QColor(255, 255, 255), 2)
            painter.setPen(handle_pen)
            painter.setBrush(QColor(80, 220, 120))
            for corner in self._corners():
                painter.drawEllipse(corner, _HANDLE_RADIUS, _HANDLE_RADIUS)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not self._draggable:
            return
        pos = event.position().toPoint()
        for index, corner in enumerate(self._corners()):
            if (corner - pos).manhattanLength() <= _HANDLE_HIT_RADIUS:
                self._dragging_corner = index
                return

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not self._draggable or self._dragging_corner is None:
            return
        pos = event.position().toPoint()
        r = self._rect
        if self._dragging_corner == 0:  # top-left
            r.setTopLeft(pos)
        elif self._dragging_corner == 1:  # top-right
            r.setTopRight(pos)
        elif self._dragging_corner == 2:  # bottom-left
            r.setBottomLeft(pos)
        elif self._dragging_corner == 3:  # bottom-right
            r.setBottomRight(pos)
        self._rect = r.normalized()
        self.update()
        self.rect_changed.emit()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._dragging_corner = None


class GridConfirmPage(QWidget):
    """The "Looks right / Let me adjust" screen."""

    grid_confirmed = Signal(int, int, int, int)  # left, top, width, height

    def __init__(self, frame: np.ndarray, detected_left: int, detected_top: int, detected_width: int, detected_height: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._overlay = _GridOverlayWidget(
            frame, QRect(detected_left, detected_top, detected_width, detected_height), self
        )
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        heading = QLabel("<h2>Does this look right?</h2>", self)
        layout.addWidget(heading)
        layout.addWidget(self._overlay)

        buttons = QHBoxLayout()
        self._looks_right_button = QPushButton("Looks right", self)
        self._looks_right_button.clicked.connect(self._on_looks_right)
        buttons.addWidget(self._looks_right_button)

        self._adjust_button = QPushButton("Let me adjust", self)
        self._adjust_button.clicked.connect(self._on_adjust)
        buttons.addWidget(self._adjust_button)

        self._done_adjusting_button = QPushButton("Done adjusting", self)
        self._done_adjusting_button.setVisible(False)
        self._done_adjusting_button.clicked.connect(self._on_looks_right)
        buttons.addWidget(self._done_adjusting_button)

        layout.addLayout(buttons)

    def _on_adjust(self) -> None:
        self._overlay.set_draggable(True)
        self._adjust_button.setVisible(False)
        self._looks_right_button.setVisible(False)
        self._done_adjusting_button.setVisible(True)

    def _on_looks_right(self) -> None:
        rect = self._overlay.rect()
        self.grid_confirmed.emit(rect.left(), rect.top(), rect.width(), rect.height())
