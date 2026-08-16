"""Runtime selection of the best available :class:`InputBackend`."""

from __future__ import annotations

import logging

from backends.input.base import InputBackend

logger = logging.getLogger(__name__)


def create_input_backend(use_pydirectinput: bool = False) -> InputBackend:
    """Return an input backend.

    ``pynput`` is the universal default and is correct for browsers, Electron apps, and the
    overwhelming majority of native games. ``use_pydirectinput`` is an explicit, per-profile
    opt-in (see ``backends/input/pydirectinput_backend.py``) for the narrow set of DirectInput
    titles that ignore standard injection; it is never selected automatically.
    """
    if use_pydirectinput:
        try:
            from backends.input.pydirectinput_backend import PyDirectInputBackend

            if PyDirectInputBackend.is_available():
                return PyDirectInputBackend()
            logger.warning("pydirectinput requested but unavailable on this platform; using pynput")
        except Exception:  # noqa: BLE001 - any failure falls back to pynput
            logger.warning("Failed to construct pydirectinput backend; falling back to pynput", exc_info=True)

    from backends.input.pynput_backend import PynputInputBackend

    return PynputInputBackend()
