"""macOS Screen Recording and Accessibility permission handling.

macOS requires both permissions to be granted manually in System Settings and the app must be
relaunched afterward for the grant to take effect in-process -- there is no API to "just work"
after the user clicks Allow. This module is the whole of the app's response to that: detect,
explain, deep-link, poll, relaunch. Windows and Linux have no equivalent and must not import
this module at all (``app/state_machine.py`` skips this wizard step entirely off-macOS).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

SCREEN_RECORDING_PANE_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"
ACCESSIBILITY_PANE_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"


class PermissionKind(Enum):
    SCREEN_RECORDING = "screen_recording"
    ACCESSIBILITY = "accessibility"


@dataclass(frozen=True)
class PermissionStatus:
    kind: PermissionKind
    granted: bool
    explanation: str
    settings_url: str


def _require_macos() -> None:
    if sys.platform != "darwin":
        raise RuntimeError("macOS permission checks were called on a non-macOS platform")


def check_screen_recording() -> bool:
    """Return whether this process currently has Screen Recording access."""
    _require_macos()
    try:
        import Quartz

        # CGPreflightScreenCaptureAccess never prompts; it only reports current state.
        return bool(Quartz.CGPreflightScreenCaptureAccess())
    except Exception:  # noqa: BLE001 - pyobjc missing or API unavailable on this macOS version
        logger.warning("Could not query Screen Recording permission", exc_info=True)
        return False


def check_accessibility() -> bool:
    """Return whether this process currently has Accessibility (input injection) access."""
    _require_macos()
    try:
        from ApplicationServices import AXIsProcessTrusted

        return bool(AXIsProcessTrusted())
    except Exception:  # noqa: BLE001 - pyobjc missing or API unavailable on this macOS version
        logger.warning("Could not query Accessibility permission", exc_info=True)
        return False


def get_permission_statuses() -> list[PermissionStatus]:
    """Return plain-language status for every permission this app needs, for the wizard screen."""
    return [
        PermissionStatus(
            kind=PermissionKind.SCREEN_RECORDING,
            granted=check_screen_recording(),
            explanation="Lets the app see your game on screen, the same way a screenshot does.",
            settings_url=SCREEN_RECORDING_PANE_URL,
        ),
        PermissionStatus(
            kind=PermissionKind.ACCESSIBILITY,
            granted=check_accessibility(),
            explanation="Lets the app press arrow keys for you while it plays.",
            settings_url=ACCESSIBILITY_PANE_URL,
        ),
    ]


def open_settings_pane(url: str) -> None:
    """Deep-link straight into the relevant System Settings pane."""
    _require_macos()
    try:
        subprocess.run(["open", url], check=True)
    except (OSError, subprocess.CalledProcessError):
        logger.error("Failed to open System Settings pane: %s", url, exc_info=True)


def request_screen_recording_prompt() -> None:
    """Trigger the OS's native one-time Screen Recording prompt, if it hasn't been shown yet."""
    _require_macos()
    try:
        import Quartz

        Quartz.CGRequestScreenCaptureAccess()
    except Exception:  # noqa: BLE001 - best-effort; the settings pane deep link is the fallback
        logger.warning("Could not trigger the Screen Recording prompt", exc_info=True)


def poll_until_granted(
    kind: PermissionKind, timeout_seconds: float = 120.0, interval_seconds: float = 1.0
) -> bool:
    """Poll a permission until granted or ``timeout_seconds`` elapses.

    Meant to run on a background thread from the permissions wizard screen so the UI can offer
    a one-click relaunch the moment the grant is detected, instead of leaving the user staring
    at a dead window (see the project's first-run experience spec).
    """
    _require_macos()
    checker = check_screen_recording if kind is PermissionKind.SCREEN_RECORDING else check_accessibility
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if checker():
            return True
        time.sleep(interval_seconds)
    return checker()


def relaunch_app() -> None:
    """Relaunch the current executable/script and exit this process.

    Required after a permission grant because ``CGPreflightScreenCaptureAccess`` and
    ``AXIsProcessTrusted`` are cached per-process at launch on macOS; a granted permission
    only takes effect for a freshly started process.
    """
    _require_macos()
    python = sys.executable
    args = [python] + sys.argv
    logger.info("Relaunching app after permission grant: %s", args)
    subprocess.Popen(args, close_fds=True, start_new_session=True)
    os._exit(0)
