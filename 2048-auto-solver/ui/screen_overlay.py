"""A live, full-screen overlay for adjusting the board region directly on top of the real game.

Direct user feedback: dragging a rectangle on a small, scaled-down static screenshot inside a
dialog was fiddly and imprecise, especially when the board is a small fraction of a large
browser window -- exactly the situation that kept producing bad calibrations. The ask was to
draw the box directly on the real screen instead, the way a native OS screenshot-selection tool
works (drag directly over what you can actually see).

This is a separate, transparent, always-on-top window positioned over the screen the target
game window is on; the user drags directly on top of the live game, which stays visible through
the transparent background since this window paints nothing but a dimming scrim, the selection
rectangle, and its drag handles.

This window only exists for the duration of one interactive adjustment and must be closed
before any screenshot-based recognition runs -- otherwise it would end up in the capture the
same way the old floating HUD did (see ui/play_panel.py's docstring for that whole story).
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, QRect, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

_HANDLE_RADIUS = 9
_HANDLE_HIT_RADIUS = 16
_MIN_RECT_SIZE = 24  # widget-space pixels == screen-logical pixels here (no scaling involved)


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


class ScreenRegionOverlay(QWidget):
    """Covers ``screen_geometry`` (screen-logical pixel coordinates, i.e. ``QScreen.geometry()``
    space) with a transparent, always-on-top window; the user drags a rectangle directly on the
    live screen beneath it. :attr:`region_confirmed` reports the confirmed rectangle in that
    same absolute screen-logical coordinate space.
    """

    region_confirmed = Signal(QRect)
    cancelled = Signal()

    def __init__(self, screen_geometry: QRect, initial_rect: QRect, parent: QWidget | None = None) -> None:
        super().__init__(
            parent,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._screen_origin = screen_geometry.topLeft()
        self._dragging_corner: int | None = None  # 0=TL, 1=TR, 2=BL, 3=BR
        self._dragging_move = False
        self._move_start_pos = QPoint()
        self._move_start_rect = QRect()
        self.setMouseTracking(True)
        # Sized to the target screen *before* computing the initial rect, so the clamp below
        # measures against the real bounds rather than this widget's not-yet-resized default.
        self.setGeometry(screen_geometry)
        # Stored relative to this widget's own top-left -- matching mouse-event coordinates,
        # which Qt already reports relative to the widget, not the screen.
        self._rect = self._clamp_rect(QRect(initial_rect.translated(-self._screen_origin)))
        self._build_controls()

    def _build_controls(self) -> None:
        # A fixed corner, not pinned to the (movable) selection rectangle -- always reachable
        # regardless of where the user has dragged the box to.
        controls = QWidget(self)
        layout = QHBoxLayout(controls)
        confirm = QPushButton("Looks right", controls)
        confirm.clicked.connect(self._confirm)
        layout.addWidget(confirm)
        cancel = QPushButton("Cancel", controls)
        cancel.clicked.connect(self._cancel)
        layout.addWidget(cancel)
        controls.setStyleSheet(
            "background-color: rgba(20,20,26,235); border-radius: 8px; color: white;"
        )
        controls.move(20, 20)
        controls.adjustSize()
        controls.raise_()

    def _confirm(self) -> None:
        self.region_confirmed.emit(self._rect.translated(self._screen_origin))

    def _cancel(self) -> None:
        self.cancelled.emit()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override signature
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Dim everything outside the selection so it reads clearly, without painting over (and
        # thus hiding) the live game inside it.
        scrim = QColor(0, 0, 0, 90)
        full = self.rect()
        for band in (
            QRect(full.left(), full.top(), full.width(), self._rect.top() - full.top()),
            QRect(full.left(), self._rect.bottom(), full.width(), full.bottom() - self._rect.bottom()),
            QRect(full.left(), self._rect.top(), self._rect.left() - full.left(), self._rect.height()),
            QRect(self._rect.right(), self._rect.top(), full.right() - self._rect.right(), self._rect.height()),
        ):
            painter.fillRect(band, scrim)

        pen = QPen(QColor(80, 220, 120), 3)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(self._rect)

        handle_pen = QPen(QColor(255, 255, 255), 2)
        painter.setPen(handle_pen)
        painter.setBrush(QColor(80, 220, 120))
        for corner in self._corners():
            painter.drawEllipse(QPointF(corner), _HANDLE_RADIUS, _HANDLE_RADIUS)

    def _corners(self) -> list[QPoint]:
        r = self._rect
        return [r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight()]

    def _corner_at(self, pos: QPoint) -> int | None:
        for index, corner in enumerate(self._corners()):
            if (corner - pos).manhattanLength() <= _HANDLE_HIT_RADIUS:
                return index
        return None

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        pos = event.position().toPoint()
        corner = self._corner_at(pos)
        if corner is not None:
            self._dragging_corner = corner
            return
        if self._rect.contains(pos):
            self._dragging_move = True
            self._move_start_pos = pos
            self._move_start_rect = QRect(self._rect)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        pos = event.position().toPoint()

        if self._dragging_corner is not None:
            self._resize_corner(self._dragging_corner, self._clamp_to_widget(pos))
            self.update()
            return

        if self._dragging_move:
            delta = pos - self._move_start_pos
            self._rect = self._clamp_rect(self._move_start_rect.translated(delta))
            self.update()
            return

        if self._corner_at(pos) is not None:
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif self._rect.contains(pos):
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._dragging_corner = None
        self._dragging_move = False

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        # A full-screen(-ish), always-on-top, click-capturing window is disorienting to be
        # stuck behind if something looks wrong -- Escape is the universal "get me out of
        # this" key, and shouldn't require precisely clicking a small Cancel button to reach.
        if event.key() == Qt.Key.Key_Escape:
            self._cancel()
            return
        super().keyPressEvent(event)

    def _clamp_to_widget(self, point: QPoint) -> QPoint:
        bounds = self.rect()
        return QPoint(_clamp(point.x(), bounds.left(), bounds.right()), _clamp(point.y(), bounds.top(), bounds.bottom()))

    def _clamp_rect(self, rect: QRect) -> QRect:
        bounds = self.rect()
        width = min(rect.width(), bounds.width())
        height = min(rect.height(), bounds.height())
        x = _clamp(rect.x(), 0, max(0, bounds.width() - width))
        y = _clamp(rect.y(), 0, max(0, bounds.height() - height))
        return QRect(x, y, width, height)

    def _resize_corner(self, corner: int, pos: QPoint) -> None:
        r = QRect(self._rect)
        if corner == 0:
            r.setTopLeft(pos)
        elif corner == 1:
            r.setTopRight(pos)
        elif corner == 2:
            r.setBottomLeft(pos)
        elif corner == 3:
            r.setBottomRight(pos)
        r = r.normalized()
        if r.width() < _MIN_RECT_SIZE or r.height() < _MIN_RECT_SIZE:
            return  # ignore a resize that would collapse the box past a usable minimum
        self._rect = r
