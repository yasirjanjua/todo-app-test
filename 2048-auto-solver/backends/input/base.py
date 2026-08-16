"""OS-agnostic keystroke-injection contract.

Assumption: input is injected at the OS level (synthetic key events), never by attaching to
the target process, patching it, or sending it window messages that bypass the normal input
pipeline. That is what makes it work identically against a browser tab, an Electron app, a
native app, or an emulator (see the "Universality over cleverness" principle in the project
README).
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable


class Key(Enum):
    """The only keys this app ever needs to send."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


@runtime_checkable
class InputBackend(Protocol):
    """Contract every input implementation (pynput, pydirectinput, ...) must satisfy.

    Implementations must guarantee that :meth:`release_all` leaves no key in a held-down
    state, even if it is called after a previous ``press_key`` without a matching
    ``release_key`` -- this is what backs the "stopping must never leave a modifier key held
    down" requirement (Part 3 / Part 5 robustness requirement 5).
    """

    def tap_key(self, key: Key, hold_ms: int = 40) -> None:
        """Press and release ``key``, holding it down for ``hold_ms`` milliseconds.

        Most 2048 implementations (DOM keydown/keyup listeners, canvas games polling key
        state) need a real hold duration rather than an instantaneous press+release to
        register reliably.
        """
        ...

    def release_all(self) -> None:
        """Release every key this backend may currently be holding. Always safe to call."""
        ...
