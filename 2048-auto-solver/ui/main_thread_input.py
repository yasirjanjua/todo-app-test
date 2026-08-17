"""Marshals keyboard injection from a background thread onto the Qt main thread.

The play loop (``app/play_loop.py``) runs on its own background thread so it never blocks the
GUI's event loop -- but on macOS, ``pynput``'s ``Controller.press()``/``release()`` can call
into HIToolbox's Text Services Manager (keyboard-layout lookup) to resolve which physical key
to send. TSM asserts it is only ever called from the main thread; calling it from any other
thread hits ``dispatch_assert_queue_fail`` and aborts the whole process outright on current
macOS -- there is no exception to catch, no way to recover in Python. See the crash report
this class was added to fix: a SIGTRAP inside ``TSMGetInputSourceProperty``, reached via
``pynput`` from the play loop's background thread.

Qt's own event loop already lives on the main thread. Wrapping the real
:class:`backends.input.base.InputBackend` in a :class:`QObject` and marshaling every call
through a signal connected with :attr:`Qt.ConnectionType.BlockingQueuedConnection` is the
standard, portable way to guarantee every actual keypress happens there -- it costs one thread
hop (negligible next to the 100-150ms animation wait the play loop already budgets per move)
and is a correctness requirement on macOS, not just a nicety.

A plain ``Qt.ConnectionType.BlockingQueuedConnection`` (rather than a signal) would need
``Q_ARG`` values with a registered Qt meta-type; a :class:`Signal` declared with ``object``
arguments accepts arbitrary Python payloads (here, a :class:`backends.input.base.Key`) without
that registration, which is why this uses signals instead.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Qt, Signal

from backends.input.base import InputBackend, Key


class MainThreadInputBackend(QObject):
    """Wraps any :class:`InputBackend` so every call executes on the thread that constructed
    this object -- expected to be the Qt main/GUI thread (construct it there, e.g. from a
    button-click handler, and its thread affinity is set correctly automatically)."""

    _tap_key_requested = Signal(object, int)
    _release_all_requested = Signal()

    def __init__(self, delegate: InputBackend, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._delegate = delegate
        # Explicit BlockingQueuedConnection (rather than AutoConnection) so this always
        # dispatches through the receiver thread's event loop -- Qt would otherwise resolve
        # AutoConnection to a same-thread DirectConnection when sender/receiver share a
        # thread, which is fine, but here the signal is only ever emitted from the *other*
        # (background) thread in the first place (see tap_key/release_all below), so a
        # same-thread emission -- and the deadlock a same-thread BlockingQueuedConnection
        # would cause -- never happens.
        self._tap_key_requested.connect(self._tap_key_on_main_thread, Qt.ConnectionType.BlockingQueuedConnection)
        self._release_all_requested.connect(self._release_all_on_main_thread, Qt.ConnectionType.BlockingQueuedConnection)

    def tap_key(self, key: Key, hold_ms: int = 40) -> None:
        if QThread.currentThread() is self.thread():
            self._delegate.tap_key(key, hold_ms)
        else:
            self._tap_key_requested.emit(key, hold_ms)

    def release_all(self) -> None:
        if QThread.currentThread() is self.thread():
            self._delegate.release_all()
        else:
            self._release_all_requested.emit()

    def _tap_key_on_main_thread(self, key: Key, hold_ms: int) -> None:
        self._delegate.tap_key(key, hold_ms)

    def _release_all_on_main_thread(self) -> None:
        self._delegate.release_all()
