"""App-wide configuration: paths and the top-level settings dataclass.

Assumption: per-game calibration (ROI, tile templates, settle timing) lives in a
:class:`app.profile.GameProfile`, not here -- this module only holds the settings that apply
across every game (solver tuning, confidence threshold, hotkeys, dry-run toggle), which is
what the UI's "Advanced" panel edits.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from core.heuristics import DEFAULT_WEIGHTS, HeuristicWeights

APP_NAME = "2048AutoSolver"


def get_app_data_dir() -> Path:
    """Return the per-OS directory for profiles, logs, and settings, creating it if needed."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / APP_NAME
    elif sys.platform == "win32":
        import os

        appdata = os.environ.get("APPDATA")
        base = Path(appdata) / APP_NAME if appdata else Path.home() / APP_NAME
    else:
        import os

        xdg_data_home = os.environ.get("XDG_DATA_HOME")
        base = (Path(xdg_data_home) if xdg_data_home else Path.home() / ".local" / "share") / APP_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_profiles_dir() -> Path:
    directory = get_app_data_dir() / "profiles"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_log_dir() -> Path:
    directory = get_app_data_dir() / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@dataclass
class HotkeyBindings:
    """Global OS-wide hotkeys, active even while the game window holds focus."""

    pause_resume: str = "p"
    stop: str = "q"
    save_snapshot: str = "s"


@dataclass
class AppConfig:
    """Cross-game settings, editable from the UI's Advanced panel."""

    base_search_depth: int = 3
    max_search_depth: int = 6
    confidence_threshold: float = 0.6
    cell_inset_ratio: float = 0.15
    settle_frames: int = 2
    settle_timeout_ms: float = 1500.0
    dry_run: bool = False
    prefer_dxcam: bool = True
    use_pydirectinput: bool = False
    key_hold_ms: int = 40
    hotkeys: HotkeyBindings = field(default_factory=HotkeyBindings)
    heuristic_weights: HeuristicWeights = field(default_factory=lambda: DEFAULT_WEIGHTS)

    # pynput's GlobalHotKeys resolves each hotkey character (e.g. "p") via KeyCode.from_char(),
    # which on macOS can call into the same HIToolbox Text Services Manager API that crashes
    # the process when touched off the main thread (see ui/main_thread_input.py's docstring
    # for the keystroke-injection half of this same class of bug). pynput's listener manages
    # its own internal OS thread for the event tap -- there is no call site of ours to marshal
    # onto the main thread the way there was for tap_key/release_all, so global hotkeys are
    # off by default on macOS until this is confirmed fixed upstream. The HUD's Pause/Stop/
    # Save Snapshot buttons cover the same functionality without touching pynput's listener at
    # all. Flip this on to try global hotkeys anyway; expect a possible hard crash on macOS.
    enable_macos_global_hotkeys: bool = False
