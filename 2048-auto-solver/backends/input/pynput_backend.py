"""Universal key-injection backend built on ``pynput``.

Works unmodified on macOS, Windows, and X11 Linux, and is what the overwhelming majority of
targets (browsers, Electron apps, most native games) respond to correctly. See
``backends/input/pydirectinput_backend.py`` for the narrow class of DirectInput games that
need a different injection path.
"""

from __future__ import annotations

import logging
import threading
import time

from backends.input.base import Key

logger = logging.getLogger(__name__)

_KEY_MAP = {
    Key.UP: "up",
    Key.DOWN: "down",
    Key.LEFT: "left",
    Key.RIGHT: "right",
}


class PynputInputBackend:
    """:class:`backends.input.base.InputBackend` implementation using ``pynput``."""

    def __init__(self) -> None:
        from pynput.keyboard import Controller, Key as PynputKey  # local import: optional dependency

        self._controller = Controller()
        self._pynput_key_cls = PynputKey
        self._key_lookup = {
            Key.UP: PynputKey.up,
            Key.DOWN: PynputKey.down,
            Key.LEFT: PynputKey.left,
            Key.RIGHT: PynputKey.right,
        }
        self._lock = threading.Lock()
        self._held: set[Key] = set()

    @staticmethod
    def is_available() -> bool:
        try:
            import pynput  # noqa: F401
        except ImportError:
            return False
        return True

    def tap_key(self, key: Key, hold_ms: int = 40) -> None:
        pynput_key = self._key_lookup[key]
        with self._lock:
            try:
                self._controller.press(pynput_key)
                self._held.add(key)
                time.sleep(max(hold_ms, 0) / 1000.0)
            finally:
                self._controller.release(pynput_key)
                self._held.discard(key)

    def release_all(self) -> None:
        with self._lock:
            for key in list(self._held):
                try:
                    self._controller.release(self._key_lookup[key])
                except Exception:  # noqa: BLE001 - best-effort cleanup, must never raise
                    logger.warning("Failed to release key %s during cleanup", key, exc_info=True)
                finally:
                    self._held.discard(key)
