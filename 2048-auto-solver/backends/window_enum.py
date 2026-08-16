"""Cross-platform window enumeration for the "pick your game" wizard step.

Three OS-specific backends, dispatched by :func:`enumerate_windows`: ``pygetwindow`` on
Windows, Quartz on macOS, and raw ``python-xlib`` calls on Linux/X11. ``pygetwindow`` looks
like a natural cross-platform choice but explicitly does not support Linux (it raises
``NotImplementedError`` at import time there), so X11 talks to the window manager's EWMH
``_NET_CLIENT_LIST`` directly, falling back to a raw window-tree walk when no EWMH-compliant
window manager is running.

Assumption: window geometry from this module is in whatever units the OS window-management
API natively reports (logical points on macOS via Quartz, pixels on Windows via
``pygetwindow`` and on Linux via Xlib). Callers that build a
:class:`backends.capture.base.CaptureRegion` from this geometry must multiply by
``CaptureBackend.get_scale_factor()`` first -- see the module docstring in
``backends/capture/base.py``.
"""

from __future__ import annotations

import difflib
import logging
import sys
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_TITLE_HINTS = ("2048", "twenty forty eight", "twenty-forty-eight")

# System chrome that shows up in raw window lists but is never the user's game.
_SYSTEM_TITLE_BLOCKLIST = {
    "dock",
    "window server",
    "menubar",
    "spotlight",
    "notification center",
    "control center",
    "program manager",
    "desktop window manager",
}


@dataclass(frozen=True)
class WindowInfo:
    """One enumerated, on-screen application window."""

    window_id: int
    title: str
    left: int
    top: int
    width: int
    height: int
    thumbnail: np.ndarray | None  # small BGR preview for the visual picker, or None if unavailable
    match_score: float  # 0.0-1.0, how strongly the title suggests this is a 2048 game


def _title_match_score(title: str) -> float:
    lowered = title.lower()
    best = 0.0
    for hint in _TITLE_HINTS:
        if hint in lowered:
            return 1.0
        ratio = difflib.SequenceMatcher(None, lowered, hint).ratio()
        best = max(best, ratio)
    return best


def _is_real_window(title: str, width: int, height: int) -> bool:
    if not title or not title.strip():
        return False
    if title.strip().lower() in _SYSTEM_TITLE_BLOCKLIST:
        return False
    # Filter out slivers and off-screen helper windows that aren't a plausible game surface.
    return width >= 200 and height >= 150


def _enumerate_windows_windows() -> list[tuple[int, str, int, int, int, int]]:
    """Windows enumeration via ``pygetwindow`` (its only supported non-macOS platform)."""
    import pygetwindow  # local import: optional dependency, not needed outside this function

    raw: list[tuple[int, str, int, int, int, int]] = []
    for win in pygetwindow.getAllWindows():
        try:
            if not win.visible or win.isMinimized:
                continue
            raw.append((hash(win._hWnd) if hasattr(win, "_hWnd") else id(win), win.title, win.left, win.top, win.width, win.height))
        except Exception:  # noqa: BLE001 - a single misbehaving window handle must not abort enumeration
            logger.debug("Skipping a window that raised during enumeration", exc_info=True)
    return raw


def _decode_wm_name(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value) if value else ""


def _window_title_x11(display, window) -> str:
    from Xlib.error import XError

    try:
        net_wm_name = display.intern_atom("_NET_WM_NAME")
        utf8_string = display.intern_atom("UTF8_STRING")
        prop = window.get_full_property(net_wm_name, utf8_string)
        if prop and prop.value:
            return _decode_wm_name(prop.value)
        wm_name = window.get_wm_name()
        return _decode_wm_name(wm_name)
    except XError:
        return ""


def _window_geometry_x11(display, root, window) -> tuple[int, int, int, int] | None:
    from Xlib.error import XError

    try:
        geometry = window.get_geometry()
        # python-xlib's translate_coords(self, src_window, src_x, src_y) maps src_wid to the
        # *argument* and dst_wid to `self` -- i.e. the object you call it on is the
        # destination, not the source, despite what the call reads like. To get `window`'s
        # origin in root's coordinate space (its absolute screen position, correct even after
        # a window manager reparents it into a decoration frame), call it on `root` with
        # `window` as the argument.
        translated = root.translate_coords(window, 0, 0)
        return translated.x, translated.y, geometry.width, geometry.height
    except XError:
        return None


def _enumerate_via_ewmh(display, root) -> list[tuple[int, str, int, int, int, int]]:
    """Preferred X11 path: ask the window manager for its authoritative client list."""
    from Xlib import X

    net_client_list = display.intern_atom("_NET_CLIENT_LIST")
    response = root.get_full_property(net_client_list, X.AnyPropertyType)
    if response is None or not response.value:
        return []

    raw: list[tuple[int, str, int, int, int, int]] = []
    for window_id in response.value:
        window = display.create_resource_object("window", window_id)
        title = _window_title_x11(display, window)
        geometry = _window_geometry_x11(display, root, window)
        if geometry is None:
            continue
        left, top, width, height = geometry
        raw.append((window_id, title, left, top, width, height))
    return raw


def _enumerate_via_tree_walk(display, root) -> list[tuple[int, str, int, int, int, int]]:
    """Fallback for window managers (or no window manager at all) that skip EWMH: walk the
    root window's direct children and keep the ones that are actually mapped on screen."""
    from Xlib import X
    from Xlib.error import XError

    raw: list[tuple[int, str, int, int, int, int]] = []
    try:
        tree = root.query_tree()
    except XError:
        return raw

    for window in tree.children:
        try:
            attrs = window.get_attributes()
            if attrs.map_state != X.IsViewable:
                continue
        except XError:
            continue
        title = _window_title_x11(display, window)
        geometry = _window_geometry_x11(display, root, window)
        if geometry is None:
            continue
        left, top, width, height = geometry
        raw.append((window.id, title, left, top, width, height))
    return raw


def _enumerate_windows_linux() -> list[tuple[int, str, int, int, int, int]]:
    """X11 enumeration via ``python-xlib``. No Linux support exists in ``pygetwindow`` (it
    raises ``NotImplementedError`` on import there), so this talks to the X server directly:
    first via the EWMH ``_NET_CLIENT_LIST`` a compliant window manager maintains, falling back
    to a raw window-tree walk when no such property is published (a minimal or absent WM).
    """
    from Xlib.display import Display

    display = Display()
    try:
        root = display.screen().root
        windows = _enumerate_via_ewmh(display, root)
        if not windows:
            windows = _enumerate_via_tree_walk(display, root)
        return windows
    finally:
        display.close()


def _enumerate_windows_macos() -> list[tuple[int, str, int, int, int, int]]:
    """macOS enumeration via Quartz's window server (no ``pygetwindow`` support there)."""
    import Quartz  # local import: optional dependency (pyobjc-framework-Quartz), macOS-only

    options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    window_list = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)
    raw: list[tuple[int, str, int, int, int, int]] = []
    for entry in window_list:
        title = entry.get("kCGWindowName", "") or ""
        owner = entry.get("kCGWindowOwnerName", "") or ""
        bounds = entry.get("kCGWindowBounds")
        window_id = entry.get("kCGWindowNumber", 0)
        if not bounds:
            continue
        display_title = title or owner
        raw.append(
            (
                int(window_id),
                display_title,
                int(bounds["X"]),
                int(bounds["Y"]),
                int(bounds["Width"]),
                int(bounds["Height"]),
            )
        )
    return raw


def enumerate_windows() -> list[tuple[int, str, int, int, int, int]]:
    """Return raw ``(window_id, title, left, top, width, height)`` tuples for visible windows."""
    if sys.platform == "darwin":
        return _enumerate_windows_macos()
    if sys.platform == "win32":
        return _enumerate_windows_windows()
    return _enumerate_windows_linux()


def list_candidate_windows(capture_thumbnail: bool = True, thumbnail_max_dim: int = 220) -> list[WindowInfo]:
    """Enumerate real, on-screen application windows, best 2048-title-match first.

    Thumbnails are captured through the normal capture backend (same code path the play loop
    uses) so what the picker shows the user is exactly what recognition will see, including
    any DPI scaling quirks.
    """
    from backends.capture.base import CaptureRegion
    from backends.capture.factory import create_capture_backend

    capture_backend = create_capture_backend() if capture_thumbnail else None
    scale = capture_backend.get_scale_factor() if capture_backend else 1.0

    results: list[WindowInfo] = []
    for window_id, title, left, top, width, height in enumerate_windows():
        if not _is_real_window(title, width, height):
            continue

        thumbnail: np.ndarray | None = None
        if capture_backend is not None:
            try:
                region = CaptureRegion(
                    left=int(left * scale),
                    top=int(top * scale),
                    width=max(1, int(width * scale)),
                    height=max(1, int(height * scale)),
                )
                frame = capture_backend.grab(region)
                thumbnail = _downscale(frame, thumbnail_max_dim)
            except Exception:  # noqa: BLE001 - a window we can't thumbnail is still selectable
                logger.debug("Failed to capture thumbnail for window %r", title, exc_info=True)

        results.append(
            WindowInfo(
                window_id=window_id,
                title=title,
                left=left,
                top=top,
                width=width,
                height=height,
                thumbnail=thumbnail,
                match_score=_title_match_score(title),
            )
        )

    if capture_backend is not None:
        capture_backend.close()

    results.sort(key=lambda w: w.match_score, reverse=True)
    return results


def _downscale(frame: np.ndarray, max_dim: int) -> np.ndarray:
    import cv2

    height, width = frame.shape[:2]
    scale = min(1.0, max_dim / max(height, width))
    if scale >= 1.0:
        return frame
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
