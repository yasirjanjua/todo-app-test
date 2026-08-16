"""The top-level window: a wizard stack (permissions -> pick game -> confirm grid -> learn
tiles -> play), per the spec's five-step first-run flow.

Recognizing a previously-calibrated game (via a saved :class:`app.profile.GameProfile`) skips
straight to the play view, satisfying "second run must be zero-setup".
"""

from __future__ import annotations

import logging
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QMessageBox, QPushButton, QStackedWidget, QVBoxLayout, QWidget

from app.config import AppConfig
from app.play_loop import PlayController, PlayEvent, PlayState
from app.profile import GameProfile, ProfileStore
from app.state_machine import WizardStateMachine, WizardStep
from backends.capture.base import CaptureRegion
from backends.capture.factory import create_capture_backend
from backends.input.factory import create_input_backend
from ui.advanced_panel import AdvancedPanel
from ui.hotkeys import GlobalHotkeys
from ui.play_hud import PlayHud
from ui.wizard_grid_confirm import GridConfirmPage
from ui.wizard_window_picker import WindowPickerPage
from vision.capture import Roi
from vision.grid_detect import detect_grid
from vision.recognition import TileRecognizer

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Owns the wizard's QStackedWidget and, once calibrated, the play HUD and controller."""

    def __init__(self, config: AppConfig | None = None, profile_store: ProfileStore | None = None) -> None:
        super().__init__()
        self.setWindowTitle("2048 Auto-Solver")
        self.resize(720, 640)

        self.config = config or AppConfig()
        self.profile_store = profile_store or ProfileStore()
        self.wizard = WizardStateMachine(self.profile_store)
        self.capture_backend = create_capture_backend(prefer_dxcam=self.config.prefer_dxcam)

        self.hud: PlayHud | None = None
        self.play_controller: PlayController | None = None
        self.hotkeys = GlobalHotkeys(self.config.hotkeys, self)
        self.hotkeys.pause_resume_pressed.connect(self._toggle_pause)
        self.hotkeys.stop_pressed.connect(self._stop_play)
        self.hotkeys.save_snapshot_pressed.connect(self._save_debug_snapshot)

        self._stack = QStackedWidget(self)
        self.setCentralWidget(self._stack)

        self._advanced_panel = AdvancedPanel(self.config, self)
        self._advanced_panel.config_changed.connect(self._on_config_changed)

        self._enter_current_step()

    # -- Step routing --------------------------------------------------------------------

    def _enter_current_step(self) -> None:
        if self.wizard.step is WizardStep.PERMISSIONS:
            self._show_permissions_step()
        elif self.wizard.step is WizardStep.PICK_WINDOW:
            self._show_window_picker_step()
        elif self.wizard.step is WizardStep.CONFIRM_GRID:
            self._show_grid_confirm_step()
        elif self.wizard.step is WizardStep.LEARN_TILES:
            self._show_tile_learning_step()
        elif self.wizard.step is WizardStep.READY_TO_PLAY:
            self._show_ready_to_play_step()

    def _show_permissions_step(self) -> None:
        from ui.wizard_permissions import PermissionsWizardPage

        page = PermissionsWizardPage(self)
        page.all_granted.connect(self._advance_from_permissions)
        self._swap_page(page)

    def _advance_from_permissions(self) -> None:
        self.wizard.advance_from_permissions()
        self._enter_current_step()

    def _show_window_picker_step(self) -> None:
        page = WindowPickerPage(self)
        page.window_selected.connect(self._on_window_selected)
        self._swap_page(page)

    def _on_window_selected(self, window_info) -> None:
        scale_factor = self.capture_backend.get_scale_factor()
        bounds = (window_info.left, window_info.top, window_info.width, window_info.height)

        existing_profile = self.wizard.try_resume_from_profile(window_info.title)
        if existing_profile is not None:
            self._load_existing_profile(existing_profile, bounds, scale_factor)
            return

        self.wizard.select_window(window_info.title, bounds, scale_factor)
        self._enter_current_step()

    def _load_existing_profile(self, profile: GameProfile, bounds: tuple[int, int, int, int], scale_factor: float) -> None:
        self.wizard.data.window_title = profile.window_title
        self.wizard.data.window_bounds = bounds
        self.wizard.data.scale_factor = scale_factor
        self.wizard.data.roi = profile.roi
        self.wizard.data.settle_frames = profile.settle_frames
        self.wizard.data.settle_timeout_ms = profile.settle_timeout_ms
        self.wizard.data.confidence_threshold = profile.confidence_threshold
        self.wizard.data.use_pydirectinput = profile.use_pydirectinput
        recognizer = TileRecognizer(dict(profile.tile_templates), confidence_threshold=profile.confidence_threshold)
        self.wizard.data.tile_learner.recognizer = recognizer
        self.wizard.step = WizardStep.READY_TO_PLAY
        self._enter_current_step()

    def _show_grid_confirm_step(self) -> None:
        left, top, width, height = self.wizard.data.window_bounds
        scale = self.wizard.data.scale_factor
        region = CaptureRegion(
            left=int(left * scale), top=int(top * scale),
            width=max(1, int(width * scale)), height=max(1, int(height * scale)),
        )
        try:
            frame = self.capture_backend.grab(region)
        except Exception:  # noqa: BLE001 - surface capture failures instead of crashing the wizard
            logger.error("Failed to capture the selected window for grid detection.", exc_info=True)
            QMessageBox.warning(self, "Capture failed", "Couldn't capture that window. Please try again.")
            self.wizard.step = WizardStep.PICK_WINDOW
            self._enter_current_step()
            return

        detection = detect_grid(frame)
        self.wizard.set_grid_detection(detection)
        page = GridConfirmPage(frame, detection.left, detection.top, detection.width, detection.height, self)
        page.grid_confirmed.connect(self._on_grid_confirmed)
        self._swap_page(page)

    def _on_grid_confirmed(self, left: int, top: int, width: int, height: int) -> None:
        self.wizard.confirm_grid(left, top, width, height)
        self._enter_current_step()

    def _show_tile_learning_step(self) -> None:
        from ui.wizard_tile_learning import TileLearningPage

        roi = self._current_roi()
        page = TileLearningPage(self.capture_backend, roi, self.wizard.data.tile_learner, self)
        page.learning_finished.connect(self._on_tile_learning_finished)
        self._swap_page(page)

    def _on_tile_learning_finished(self) -> None:
        self.wizard.finish_tile_learning()
        self.wizard.save_profile()
        self._enter_current_step()

    def _show_ready_to_play_step(self) -> None:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        heading = QPushButton("Start", container)
        heading.setMinimumHeight(60)
        heading.clicked.connect(self._start_playing)
        layout.addWidget(heading)

        recalibrate = QPushButton("Recalibrate", container)
        recalibrate.clicked.connect(self._recalibrate)
        layout.addWidget(recalibrate)

        layout.addWidget(self._advanced_panel)
        self._swap_page(container)

    def _recalibrate(self) -> None:
        self.wizard.request_recalibration()
        self._enter_current_step()

    # -- Playing ---------------------------------------------------------------------------

    def _current_roi(self) -> Roi:
        # window_bounds is in logical (pre-DPI-scale) coordinates from window enumeration;
        # the stored ROI offset is in physical pixels relative to the window's physical
        # top-left corner (see app/profile.py). Convert the window origin to physical pixels
        # first, then add the physical-pixel offset -- never mix the two unit systems.
        left, top, _w, _h = self.wizard.data.window_bounds
        scale = self.wizard.data.scale_factor
        roi = self.wizard.data.roi
        window_physical_left = int(left * scale)
        window_physical_top = int(top * scale)
        return Roi(
            left=window_physical_left + roi.left,
            top=window_physical_top + roi.top,
            width=roi.width,
            height=roi.height,
        )

    def _start_playing(self) -> None:
        input_backend = create_input_backend(use_pydirectinput=self.wizard.data.use_pydirectinput)
        self.hud = PlayHud()
        self.hud.pause_button.clicked.connect(self._toggle_pause)
        self.hud.stop_button.clicked.connect(self._stop_play)

        self.play_controller = PlayController(
            capture_backend=self.capture_backend,
            input_backend=input_backend,
            recognizer=self.wizard.data.tile_learner.recognizer,
            roi=self._current_roi(),
            app_config=self.config,
            on_event=self._on_play_event,
        )
        self.hotkeys.start()
        self.play_controller.start()
        self.hud.show()

    def _on_play_event(self, event: PlayEvent) -> None:
        # PlayController runs its loop on a background thread; forward to the GUI thread via
        # Qt's queued-connection machinery rather than touching widgets directly here.
        if self.hud is not None:
            from PySide6.QtCore import QMetaObject, Q_ARG, Qt as QtNS

            QMetaObject.invokeMethod(self.hud, "on_play_event", QtNS.ConnectionType.QueuedConnection, Q_ARG(object, event))
        if event.kind == "game_over":
            logger.info(event.message)

    def _toggle_pause(self) -> None:
        if self.play_controller is None:
            return
        if self.play_controller.state is PlayState.RUNNING:
            self.play_controller.pause()
        else:
            self.play_controller.resume()

    def _stop_play(self) -> None:
        if self.play_controller is not None:
            self.play_controller.stop()
        self.hotkeys.stop_listening()
        if self.hud is not None:
            self.hud.close()

    def _save_debug_snapshot(self) -> None:
        if self.play_controller is None:
            return
        try:
            frame = self.play_controller.capture_debug_frame()
        except Exception:  # noqa: BLE001 - a failed debug snapshot must not affect play
            logger.error("Failed to save debug snapshot.", exc_info=True)
            return
        import time

        import cv2

        from app.config import get_log_dir

        path = get_log_dir() / f"snapshot-{int(time.time())}.png"
        cv2.imwrite(str(path), frame)
        logger.info("Saved debug snapshot to %s", path)

    # -- Config -------------------------------------------------------------------------

    def _on_config_changed(self, new_config: AppConfig) -> None:
        self.config = new_config

    def _swap_page(self, widget: QWidget) -> None:
        while self._stack.count():
            old = self._stack.widget(0)
            self._stack.removeWidget(old)
            old.deleteLater()
        self._stack.addWidget(widget)
        self._stack.setCurrentWidget(widget)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override signature
        self._stop_play()
        super().closeEvent(event)
