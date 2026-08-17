"""Grid confirmation: a highlighted overlay on a screenshot, not a raw ``cv2.selectROI`` call.

Auto-detection (``vision/grid_detect.py``) runs first; this page just renders its proposal as
a rectangle over the captured window and asks "Looks right?" / "Let me adjust". The adjust
path is a custom drag-handle widget -- ``cv2.selectROI`` is a developer tool and was
explicitly ruled out by the spec because it looks like one.

Two things this page has to get right that a naive "just show the screenshot" implementation
doesn't:

* **The captured window can be far bigger than the dialog.** A full browser window is often
  1000+ physical pixels a side; shown at 1:1 it overflows any reasonable window. This widget
  always scales the displayed image down to fit (never up -- that would blur it), and when
  auto-detection found a plausible grid, it also crops the *displayed* screenshot down to a
  padded region around that grid, so the user is looking at "the board and a bit of context"
  rather than an entire browser chrome. The crop is display-only: the rectangle the user
  confirms is still translated back into full-frame coordinates before being reported.
* **The rectangle must be movable as a whole, not just resizable by its corners.** Dragging
  inside the rectangle's body (away from a corner handle) translates the whole box instead of
  resizing it, which is what most users try first.
"""

from __future__ import annotations

import cv2
import numpy as np

from PySide6.QtCore import QPoint, QPointF, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QImage, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

_HANDLE_RADIUS = 8
_HANDLE_HIT_RADIUS = 14
_MIN_RECT_SIZE = 24  # image-space pixels; a rectangle can't be resized smaller than this

# How much room (relative to the detected grid's own size) to show around it when cropping
# the displayed screenshot down. Generous enough to show surrounding board chrome for context.
_CROP_PADDING_FACTOR = 0.75
_CROP_MIN_PADDING_PX = 60


def _bgr_to_qimage(frame: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    return QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def _default_max_display_size() -> QSize:
    """A display budget that always fits the current screen, even on a small laptop."""
    screen = QApplication.primaryScreen()
    available = screen.availableGeometry() if screen is not None else QRect(0, 0, 1280, 800)
    width = _clamp(int(available.width() * 0.7), 400, 900)
    height = _clamp(int(available.height() * 0.55), 300, 650)
    return QSize(width, height)


class _GridOverlayWidget(QWidget):
    """Renders a (possibly display-cropped) screenshot, scaled to fit, with a draggable
    rectangle on top.

    The rectangle is axis-aligned (matching :class:`backends.capture.base.CaptureRegion`) and
    stored internally in *image space* (the natural pixel coordinates of ``image``, before any
    fit-to-widget scaling) -- :meth:`rect` always returns image-space coordinates regardless of
    how small the widget is actually rendered on screen. Each of its four corner handles can be
    dragged independently to resize; dragging anywhere else inside the rectangle moves the
    whole thing.
    """

    rect_changed = Signal()

    def __init__(self, image: QImage, initial_rect: QRect, max_display_size: QSize, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image = image
        self._rect = QRect(initial_rect)
        self._dragging_corner: int | None = None  # 0=TL, 1=TR, 2=BL, 3=BR
        self._dragging_move = False
        self._move_start_image_pos = QPoint()
        self._move_start_rect = QRect()
        self._draggable = False

        img_w, img_h = max(1, image.width()), max(1, image.height())
        self._scale = min(1.0, max_display_size.width() / img_w, max_display_size.height() / img_h)
        self.setFixedSize(max(1, round(img_w * self._scale)), max(1, round(img_h * self._scale)))
        self.setMouseTracking(True)

    def set_draggable(self, draggable: bool) -> None:
        self._draggable = draggable
        if not draggable:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def rect(self) -> QRect:
        """The confirmed/edited rectangle, in image-space (natural pixel) coordinates."""
        return QRect(self._rect)

    def _image_bounds(self) -> QRect:
        return QRect(0, 0, self._image.width(), self._image.height())

    def _to_image_point(self, widget_pos: QPointF) -> QPoint:
        return QPoint(round(widget_pos.x() / self._scale), round(widget_pos.y() / self._scale))

    def _to_widget_point(self, image_point: QPoint) -> QPointF:
        return QPointF(image_point.x() * self._scale, image_point.y() * self._scale)

    def _widget_corners(self) -> list[QPointF]:
        r = self._rect
        return [
            self._to_widget_point(r.topLeft()),
            self._to_widget_point(r.topRight()),
            self._to_widget_point(r.bottomLeft()),
            self._to_widget_point(r.bottomRight()),
        ]

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override signature
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(self._widget_rect(), self._image)

        top_left = self._to_widget_point(self._rect.topLeft())
        bottom_right = self._to_widget_point(self._rect.bottomRight())
        widget_rect = QRect(top_left.toPoint(), bottom_right.toPoint())

        overlay_pen = QPen(QColor(80, 220, 120), 3)
        painter.setPen(overlay_pen)
        painter.setBrush(QColor(80, 220, 120, 40))
        painter.drawRect(widget_rect)

        # The 4x4 cell-boundary lines this box implies, so a misalignment (a boundary cutting
        # through a tile instead of the gap between tiles) is visible right here, before
        # calibration ever starts -- rather than only inferable later from a flood of
        # "unrecognized tile" noise once cell crops are actually being sliced along it.
        if widget_rect.width() > 0 and widget_rect.height() > 0:
            grid_pen = QPen(QColor(255, 255, 255, 130), 1, Qt.PenStyle.DashLine)
            painter.setPen(grid_pen)
            for i in range(1, 4):
                x = widget_rect.left() + widget_rect.width() * i / 4
                painter.drawLine(QPointF(x, widget_rect.top()), QPointF(x, widget_rect.bottom()))
                y = widget_rect.top() + widget_rect.height() * i / 4
                painter.drawLine(QPointF(widget_rect.left(), y), QPointF(widget_rect.right(), y))

        if self._draggable:
            handle_pen = QPen(QColor(255, 255, 255), 2)
            painter.setPen(handle_pen)
            painter.setBrush(QColor(80, 220, 120))
            for corner in self._widget_corners():
                painter.drawEllipse(corner, _HANDLE_RADIUS, _HANDLE_RADIUS)

    def _widget_rect(self) -> QRect:
        size = self.size()
        return QRect(0, 0, size.width(), size.height())

    def _corner_at(self, widget_pos: QPointF) -> int | None:
        for index, corner in enumerate(self._widget_corners()):
            if (corner - widget_pos).manhattanLength() <= _HANDLE_HIT_RADIUS:
                return index
        return None

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not self._draggable:
            return
        widget_pos = event.position()
        corner = self._corner_at(widget_pos)
        if corner is not None:
            self._dragging_corner = corner
            return

        image_pos = self._to_image_point(widget_pos)
        if self._rect.contains(image_pos):
            self._dragging_move = True
            self._move_start_image_pos = image_pos
            self._move_start_rect = QRect(self._rect)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not self._draggable:
            return
        widget_pos = event.position()

        if self._dragging_corner is not None:
            image_pos = self._clamp_to_image(self._to_image_point(widget_pos))
            self._resize_corner(self._dragging_corner, image_pos)
            self.update()
            self.rect_changed.emit()
            return

        if self._dragging_move:
            image_pos = self._to_image_point(widget_pos)
            delta = image_pos - self._move_start_image_pos
            self._rect = self._clamp_rect_to_image(self._move_start_rect.translated(delta))
            self.update()
            self.rect_changed.emit()
            return

        self._update_hover_cursor(widget_pos)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._dragging_corner = None
        self._dragging_move = False

    def _clamp_to_image(self, point: QPoint) -> QPoint:
        bounds = self._image_bounds()
        return QPoint(_clamp(point.x(), bounds.left(), bounds.right()), _clamp(point.y(), bounds.top(), bounds.bottom()))

    def _clamp_rect_to_image(self, rect: QRect) -> QRect:
        bounds = self._image_bounds()
        width = min(rect.width(), bounds.width())
        height = min(rect.height(), bounds.height())
        x = _clamp(rect.x(), 0, max(0, bounds.width() - width))
        y = _clamp(rect.y(), 0, max(0, bounds.height() - height))
        return QRect(x, y, width, height)

    def _resize_corner(self, corner: int, image_pos: QPoint) -> None:
        r = QRect(self._rect)
        if corner == 0:
            r.setTopLeft(image_pos)
        elif corner == 1:
            r.setTopRight(image_pos)
        elif corner == 2:
            r.setBottomLeft(image_pos)
        elif corner == 3:
            r.setBottomRight(image_pos)
        r = r.normalized()
        if r.width() < _MIN_RECT_SIZE or r.height() < _MIN_RECT_SIZE:
            return  # ignore a resize that would collapse the box past a usable minimum
        self._rect = r

    def _update_hover_cursor(self, widget_pos: QPointF) -> None:
        if self._corner_at(widget_pos) is not None:
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif self._rect.contains(self._to_image_point(widget_pos)):
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)


class GridConfirmPage(QWidget):
    """The "Looks right / Let me adjust" screen.

    Accepts the *full* captured window frame plus the auto-detected rectangle (both in that
    frame's own pixel coordinates); internally crops the displayed screenshot down to a padded
    region around the detection (falling back to the whole frame when detection failed) and
    scales it to fit the screen, then translates the user's confirmed rectangle back into
    full-frame coordinates before emitting :attr:`grid_confirmed`.
    """

    grid_confirmed = Signal(int, int, int, int)  # left, top, width, height -- full-frame coordinates

    def __init__(
        self,
        frame: np.ndarray,
        detected_left: int,
        detected_top: int,
        detected_width: int,
        detected_height: int,
        detection_method: str = "lattice",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        # "lattice" (individual tile cells matched into a grid) is the only method with real
        # per-cell evidence behind it; "container" (a single largest square-ish contour) and
        # "failed" (no usable detection at all, box defaults to the whole frame) are
        # best-effort guesses that need a human's eyes far more often. A real report showed
        # exactly this: a "container" result confidently proposed nearly the entire browser
        # window as the board, and nothing in the UI stopped the user proceeding with it, so
        # every "tile" learned afterward was noise from the page around the actual board.
        # Rather than rely on the user noticing on their own, low-confidence detections open
        # directly in drag-to-adjust mode with an explicit warning instead of behind an extra
        # "Let me adjust" click.
        self._needs_manual_check = detection_method != "lattice"
        frame_h, frame_w = frame.shape[:2]

        pad = max(int(max(detected_width, detected_height) * _CROP_PADDING_FACTOR), _CROP_MIN_PADDING_PX)
        crop_left = _clamp(detected_left - pad, 0, frame_w)
        crop_top = _clamp(detected_top - pad, 0, frame_h)
        crop_right = _clamp(detected_left + detected_width + pad, 0, frame_w)
        crop_bottom = _clamp(detected_top + detected_height + pad, 0, frame_h)
        if crop_right <= crop_left or crop_bottom <= crop_top:
            # Degenerate detection (e.g. zero-size); fall back to showing the whole frame.
            crop_left, crop_top, crop_right, crop_bottom = 0, 0, frame_w, frame_h

        self._crop_offset = (crop_left, crop_top)
        display_frame = frame[crop_top:crop_bottom, crop_left:crop_right]
        display_image = _bgr_to_qimage(display_frame)

        initial_rect = QRect(
            detected_left - crop_left, detected_top - crop_top, detected_width, detected_height
        )

        self._overlay = _GridOverlayWidget(display_image, initial_rect, _default_max_display_size(), self)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        heading = QLabel("<h2>Does this look right?</h2>", self)
        layout.addWidget(heading)

        if self._needs_manual_check:
            warning = QLabel(
                "I couldn't confidently find the board on my own this time. Drag the green "
                "box's corners (or its middle, to move the whole thing) until it lines up "
                "with just the 4x4 grid, then click Looks right.",
                self,
            )
            warning.setWordWrap(True)
            warning.setStyleSheet("color: #a15c00; font-weight: bold;")
            layout.addWidget(warning)

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

        if self._needs_manual_check:
            # Skip straight to drag-to-adjust rather than requiring an extra click a user
            # might not realize they need -- see this class's docstring comment for why.
            self._on_adjust()

    def _on_adjust(self) -> None:
        self._overlay.set_draggable(True)
        self._adjust_button.setVisible(False)
        self._looks_right_button.setVisible(False)
        self._done_adjusting_button.setVisible(True)

    def _on_looks_right(self) -> None:
        rect = self._overlay.rect()
        offset_left, offset_top = self._crop_offset
        self.grid_confirmed.emit(rect.left() + offset_left, rect.top() + offset_top, rect.width(), rect.height())
