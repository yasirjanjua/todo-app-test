"""OS-wide global hotkeys (P pause/resume, Q stop, S save snapshot).

Registered through ``pynput`` so they fire even while the game window holds focus -- a Qt
shortcut alone would only work while the app itself is focused, which is useless once the bot
is actively playing a different window. ``pynput``'s listener runs on its own thread; Qt
signals are queued automatically across threads as long as the receiving :class:`QObject`
lives on a thread with a running event loop, which is always true here (the main/GUI thread).
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal

from app.config import HotkeyBindings

logger = logging.getLogger(__name__)


class GlobalHotkeys(QObject):
    """Wraps a ``pynput.keyboard.GlobalHotKeys`` listener and re-emits presses as Qt signals."""

    pause_resume_pressed = Signal()
    stop_pressed = Signal()
    save_snapshot_pressed = Signal()

    def __init__(self, bindings: HotkeyBindings, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._bindings = bindings
        self._listener = None

    def start(self) -> None:
        """Start listening. Safe to call once; call :meth:`stop_listening` before re-starting."""
        if self._listener is not None:
            logger.warning("GlobalHotkeys.start() called while already listening; ignoring.")
            return
        from pynput import keyboard

        hotkey_map = {
            self._bindings.pause_resume: self.pause_resume_pressed.emit,
            self._bindings.stop: self.stop_pressed.emit,
            self._bindings.save_snapshot: self.save_snapshot_pressed.emit,
        }
        try:
            self._listener = keyboard.GlobalHotKeys(hotkey_map)
            self._listener.start()
            logger.info("Global hotkeys active: %s", self._bindings)
        except Exception:  # noqa: BLE001 - e.g. missing Accessibility permission on macOS
            logger.error("Failed to register global hotkeys.", exc_info=True)
            self._listener = None

    def stop_listening(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
