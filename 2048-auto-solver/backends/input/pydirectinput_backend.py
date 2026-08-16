"""Optional, Windows-only key-injection backend built on ``pydirectinput``.

Some DirectInput titles (mostly native Windows games, never browsers or Electron apps) ignore
the ``SendInput`` calls ``pynput`` issues for standard input and only respond to
``pydirectinput``'s scan-code-based injection. Offered as an explicit fallback the user (or a
saved profile) can opt into per-game, never auto-selected, since ``pydirectinput`` also moves
the real mouse cursor as part of some of its calls and is a strictly narrower tool than
``pynput``.
"""

from __future__ import annotations

import logging
import sys
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


class PyDirectInputBackend:
    """:class:`backends.input.base.InputBackend` implementation using ``pydirectinput``."""

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("PyDirectInputBackend is only available on Windows")
        import pydirectinput  # local import: optional dependency, Windows-only

        pydirectinput.FAILSAFE = False
        self._pydirectinput = pydirectinput
        self._lock = threading.Lock()
        self._held: set[Key] = set()

    @staticmethod
    def is_available() -> bool:
        if sys.platform != "win32":
            return False
        try:
            import pydirectinput  # noqa: F401
        except ImportError:
            return False
        return True

    def tap_key(self, key: Key, hold_ms: int = 40) -> None:
        scan_key = _KEY_MAP[key]
        with self._lock:
            try:
                self._pydirectinput.keyDown(scan_key)
                self._held.add(key)
                time.sleep(max(hold_ms, 0) / 1000.0)
            finally:
                self._pydirectinput.keyUp(scan_key)
                self._held.discard(key)

    def release_all(self) -> None:
        with self._lock:
            for key in list(self._held):
                try:
                    self._pydirectinput.keyUp(_KEY_MAP[key])
                except Exception:  # noqa: BLE001 - best-effort cleanup, must never raise
                    logger.warning("Failed to release key %s during cleanup", key, exc_info=True)
                finally:
                    self._held.discard(key)
