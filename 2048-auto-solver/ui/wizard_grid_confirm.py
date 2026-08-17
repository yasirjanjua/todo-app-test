"""Grid confirmation: a static preview of the captured screenshot, plus a live on-screen box.

Auto-detection (``vision/grid_detect.py``) runs first; this page renders its proposal as a
static (non-interactive) preview and asks "Looks right?". Adjustment itself doesn't happen here
at all -- it happens on ``ui/screen_overlay.py``'s ``ScreenRegionOverlay``, a transparent window
drawn directly on top of the real game, so the user drags a box over what they can actually see
instead of a small, scaled-down copy of it in a dialog. Direct user feedback named this
specifically: dragging on a static screenshot was fiddly and imprecise, especially for a board
that's a small fraction of a large browser window -- exactly the situation that kept producing
bad calibrations. ``cv2.selectROI`` was ruled out from the start for looking like a developer
tool; a live selection over the real screen is the non-technical equivalent of one.

Two things this page still has to get right about the static preview it shows:

* **The captured window can be far bigger than the dialog.** A full browser window is often
  1000+ physical pixels a side; shown at 1:1 it overflows any reasonable window. This widget
  always scales the displayed image down to fit (never up -- that would blur it), and when
  auto-detection found a plausible grid, it also crops the *displayed* screenshot down to a
  padded region around that grid, so the user is looking at "the board and a bit of context"
  rather than an entire browser chrome.
* **A low-confidence detection must not be confirmable without actually being checked.** See
  ``_needs_manual_check`` -- "lattice" (individual tile cells matched into a grid) is the only
  detection method with real per-cell evidence behind it; "container" (a single largest
  square-ish contour) and "failed" (no usable detection at all) are best-effort guesses that
  need a human's eyes far more often. A real report showed exactly this: a "container" result
  confidently proposed nearly the entire browser window as the board, and nothing in the UI
  stopped the user proceeding with it, so every "tile" learned afterward was noise from the
  page around the actual board.
"""

from __future__ import annotations

import cv2
import numpy as np

from PySide6.QtCore import QPointF, QRect, QSize, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

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


class _GridPreviewWidget(QWidget):
    """Renders a (possibly display-cropped) screenshot, scaled to fit, with a static rectangle
    showing the currently-confirmed board region. Purely informational -- see
    ``ui/screen_overlay.py``'s ``ScreenRegionOverlay`` for the actual interactive editing, which
    happens directly on the live screen instead of on a static, scaled-down copy of it.
    """

    def __init__(self, image: QImage, initial_rect: QRect, max_display_size: QSize, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image = image
        self._rect = QRect(initial_rect)

        img_w, img_h = max(1, image.width()), max(1, image.height())
        self._scale = min(1.0, max_display_size.width() / img_w, max_display_size.height() / img_h)
        self.setFixedSize(max(1, round(img_w * self._scale)), max(1, round(img_h * self._scale)))

    def rect(self) -> QRect:
        """The current rectangle, in image-space (natural pixel) coordinates."""
        return QRect(self._rect)

    def _to_widget_point(self, image_point) -> QPointF:
        return QPointF(image_point.x() * self._scale, image_point.y() * self._scale)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override signature
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(QRect(0, 0, self.width(), self.height()), self._image)

        top_left = self._to_widget_point(self._rect.topLeft())
        bottom_right = self._to_widget_point(self._rect.bottomRight())
        widget_rect = QRect(top_left.toPoint(), bottom_right.toPoint())

        pen = QPen(QColor(80, 220, 120), 3)
        painter.setPen(pen)
        painter.setBrush(QColor(80, 220, 120, 40))
        painter.drawRect(widget_rect)


class GridConfirmPage(QWidget):
    """The "Does this look right?" screen.

    Accepts the *full* captured window frame plus the auto-detected rectangle (both in that
    frame's own pixel coordinates); internally crops the displayed screenshot down to a padded
    region around the detection (falling back to the whole frame when detection failed) and
    scales it to fit the screen. Confirming without adjustment (only offered when the detection
    method is trustworthy -- see the module docstring) translates the auto-detected rectangle
    into full-frame coordinates and emits :attr:`grid_confirmed` directly; adjustment is
    delegated entirely to the live on-screen overlay (see :attr:`adjust_on_screen_requested`),
    which is what actually produces the coordinates handed to :attr:`grid_confirmed` in that
    case (via ``ui/main_window.py``'s conversion back from screen space).
    """

    grid_confirmed = Signal(int, int, int, int)  # left, top, width, height -- full-frame coordinates
    adjust_on_screen_requested = Signal()

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

        self._preview = _GridPreviewWidget(display_image, initial_rect, _default_max_display_size(), self)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        heading = QLabel("<h2>Does this look right?</h2>", self)
        layout.addWidget(heading)

        self._status_label = QLabel("", self)
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #a15c00; font-weight: bold;")
        self._status_label.setVisible(False)
        layout.addWidget(self._status_label)

        layout.addWidget(self._preview)

        buttons = QHBoxLayout()
        self._looks_right_button = QPushButton("Looks right", self)
        self._looks_right_button.clicked.connect(self._on_looks_right)
        buttons.addWidget(self._looks_right_button)

        self._adjust_button = QPushButton("Adjust on my screen", self)
        self._adjust_button.clicked.connect(self.adjust_on_screen_requested.emit)
        buttons.addWidget(self._adjust_button)
        layout.addLayout(buttons)

        if self._needs_manual_check:
            # A low-confidence detection must not be confirmable as-is -- see the module
            # docstring's "must not be confirmable without actually being checked" point.
            self._looks_right_button.setVisible(False)
            self._status_label.setText(
                "I couldn't confidently find the board on my own this time. A green box is "
                "opening directly on top of your game -- drag it (or its corners) until it "
                "lines up with just the 4x4 grid, then confirm there. If it doesn't appear, "
                "use Adjust on my screen below."
            )
            self._status_label.setVisible(True)

    def _on_looks_right(self) -> None:
        rect = self._preview.rect()
        offset_left, offset_top = self._crop_offset
        self.grid_confirmed.emit(rect.left() + offset_left, rect.top() + offset_top, rect.width(), rect.height())
