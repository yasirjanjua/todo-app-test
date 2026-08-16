"""First-run wizard state machine: permissions -> pick game -> confirm grid -> learn tiles -> play.

This module holds *only* the state transitions and the calibration data accumulated along the
way; ``ui/main_window.py`` renders each step and calls into this class. Keeping the sequencing
here (rather than in the UI layer) is what makes "second run must be zero-setup" easy to
implement: :meth:`WizardStateMachine.try_resume_from_profile` can jump straight to
``READY_TO_PLAY`` without the UI needing to know why.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from enum import Enum, auto

from app.profile import GameProfile, ProfileStore, RoiOffset
from vision.grid_detect import GridDetectionResult
from vision.recognition import TileRecognizer
from vision.tile_learning import TileLearner

logger = logging.getLogger(__name__)


class WizardStep(Enum):
    PERMISSIONS = auto()  # macOS only; skipped entirely elsewhere
    PICK_WINDOW = auto()
    CONFIRM_GRID = auto()
    LEARN_TILES = auto()
    READY_TO_PLAY = auto()


@dataclass
class WizardData:
    """Calibration data accumulated as the user moves through the wizard."""

    window_title: str | None = None
    window_bounds: tuple[int, int, int, int] | None = None  # left, top, width, height (logical)
    scale_factor: float = 1.0
    grid_detection: GridDetectionResult | None = None
    roi: RoiOffset | None = None  # physical pixels, relative to the window's own origin
    tile_learner: TileLearner = field(default_factory=TileLearner)
    settle_frames: int = 2
    settle_timeout_ms: float = 1500.0
    confidence_threshold: float = 0.6
    use_pydirectinput: bool = False


class WizardStateMachine:
    """Drives the first-run flow described in the project spec, Part 1."""

    def __init__(self, profile_store: ProfileStore | None = None) -> None:
        self.profile_store = profile_store or ProfileStore()
        self.data = WizardData()
        self.step = self._first_step()

    def _first_step(self) -> WizardStep:
        # Windows and Linux have no OS permission gate; macOS shows it only if something is
        # actually missing, per the "skip the screen entirely rather than a satisfied
        # checklist" rule.
        if sys.platform == "darwin":
            from backends.permissions_macos import get_permission_statuses

            statuses = get_permission_statuses()
            if any(not s.granted for s in statuses):
                return WizardStep.PERMISSIONS
        return WizardStep.PICK_WINDOW

    def try_resume_from_profile(self, window_title: str) -> GameProfile | None:
        """Look up a saved profile for ``window_title``; if found, the wizard can be skipped."""
        profile = self.profile_store.load(window_title)
        if profile is not None:
            logger.info("Recognized %r from a saved profile; skipping the wizard.", window_title)
        return profile

    def advance_from_permissions(self) -> None:
        if self.step is not WizardStep.PERMISSIONS:
            raise RuntimeError(f"advance_from_permissions() called from {self.step}")
        self.step = WizardStep.PICK_WINDOW

    def select_window(self, window_title: str, window_bounds: tuple[int, int, int, int], scale_factor: float) -> None:
        if self.step is not WizardStep.PICK_WINDOW:
            raise RuntimeError(f"select_window() called from {self.step}")
        self.data.window_title = window_title
        self.data.window_bounds = window_bounds
        self.data.scale_factor = scale_factor
        self.step = WizardStep.CONFIRM_GRID

    def set_grid_detection(self, detection: GridDetectionResult) -> None:
        self.data.grid_detection = detection

    def confirm_grid(self, left: int, top: int, width: int, height: int) -> None:
        """Accept a grid ROI (either the auto-detected one or a user-adjusted one).

        Coordinates are physical pixels *within the captured window frame*; they get stored
        relative to the window's own origin so the profile survives the window moving later.
        """
        if self.step is not WizardStep.CONFIRM_GRID:
            raise RuntimeError(f"confirm_grid() called from {self.step}")
        self.data.roi = RoiOffset(left=left, top=top, width=width, height=height)
        self.step = WizardStep.LEARN_TILES
        self.data.tile_learner.start_new_game()

    def readjust_grid(self) -> None:
        """User clicked "Let me adjust"; stay on this step so the UI can show drag handles."""
        if self.step is not WizardStep.CONFIRM_GRID:
            raise RuntimeError(f"readjust_grid() called from {self.step}")

    def finish_tile_learning(self) -> TileRecognizer:
        if self.step is not WizardStep.LEARN_TILES:
            raise RuntimeError(f"finish_tile_learning() called from {self.step}")
        self.step = WizardStep.READY_TO_PLAY
        return self.data.tile_learner.recognizer

    def build_profile(self) -> GameProfile:
        if self.step is not WizardStep.READY_TO_PLAY:
            raise RuntimeError(f"build_profile() called from {self.step}")
        if self.data.window_title is None or self.data.roi is None:
            raise RuntimeError("Cannot build a profile before window and grid are set")
        window_width = int(self.data.window_bounds[2] * self.data.scale_factor) if self.data.window_bounds else 0
        window_height = int(self.data.window_bounds[3] * self.data.scale_factor) if self.data.window_bounds else 0
        return GameProfile(
            window_title=self.data.window_title,
            roi=self.data.roi,
            window_size_at_calibration=(window_width, window_height),
            tile_templates=dict(self.data.tile_learner.recognizer.templates),
            settle_frames=self.data.settle_frames,
            settle_timeout_ms=self.data.settle_timeout_ms,
            confidence_threshold=self.data.confidence_threshold,
            use_pydirectinput=self.data.use_pydirectinput,
        )

    def save_profile(self) -> GameProfile:
        profile = self.build_profile()
        self.profile_store.save(profile)
        return profile

    def request_recalibration(self) -> None:
        """Reset back to the window-picker step, keeping the same profile store."""
        self.data = WizardData()
        self.step = WizardStep.PICK_WINDOW
