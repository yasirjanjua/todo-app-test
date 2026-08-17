"""The top-level window: a wizard stack (permissions -> pick game -> confirm grid -> learn
tiles -> play), per the spec's five-step first-run flow.

Recognizing a previously-calibrated game (via a saved :class:`app.profile.GameProfile`) skips
straight to the play view, satisfying "second run must be zero-setup".
"""

from __future__ import annotations

import logging
import sys

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QMainWindow, QMessageBox, QPushButton, QStackedWidget, QVBoxLayout, QWidget

from app.config import AppConfig
from app.play_loop import PlayController, PlayEvent, PlayState
from app.profile import GameProfile, ProfileStore
from app.state_machine import WizardStateMachine, WizardStep
from backends.capture.base import CaptureBackend, CaptureRegion
from backends.capture.factory import create_capture_backend
from backends.input.factory import create_input_backend
from ui.main_thread_input import MainThreadInputBackend
from backends.window_enum import OWN_WINDOW_TITLE, enumerate_windows
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

    # PlayController invokes its on_event callback from the play loop's background thread.
    # A Signal(object) accepts an arbitrary Python payload (a PlayEvent) and Qt's default
    # AutoConnection resolves to a queued, cross-thread-safe delivery automatically since the
    # emitting thread differs from this QObject's own (main) thread -- unlike
    # QMetaObject.invokeMethod with Q_ARG(object, ...), which raises at call time because
    # plain Python objects have no registered Qt meta-type (see ui/main_thread_input.py for
    # the same lesson applied to keystroke injection).
    _play_event_received = Signal(object)

    def __init__(
        self,
        config: AppConfig | None = None,
        profile_store: ProfileStore | None = None,
        capture_backend: CaptureBackend | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle(OWN_WINDOW_TITLE)
        self.resize(720, 640)
        self._play_event_received.connect(self._handle_play_event)

        self.config = config or AppConfig()
        self.profile_store = profile_store or ProfileStore()
        self.wizard = WizardStateMachine(self.profile_store)
        # Injectable so tests can exercise the wizard/UI without a real display server (the
        # default mss backend needs one to open a screen-capture connection at all).
        self.capture_backend = capture_backend or create_capture_backend(prefer_dxcam=self.config.prefer_dxcam)

        self.hud: PlayHud | None = None
        self.play_controller: PlayController | None = None
        self.hotkeys = GlobalHotkeys(self.config.hotkeys, self)
        self.hotkeys.pause_resume_pressed.connect(self._toggle_pause)
        self.hotkeys.stop_pressed.connect(self._stop_play)
        self.hotkeys.save_snapshot_pressed.connect(self._save_debug_snapshot)

        self._stack = QStackedWidget(self)
        self.setCentralWidget(self._stack)

        self._enter_current_step()

    # -- Step routing --------------------------------------------------------------------

    def _enter_current_step(self) -> None:
        # Every wizard navigation funnels through here, so this is the one place that must
        # never let an unexpected exception escape silently: a real user hit exactly that (a
        # widget-lifecycle bug elsewhere) and was left staring at a screen that looked frozen,
        # with the only evidence in a terminal they weren't watching -- repeatedly clicking
        # with no idea anything had gone wrong. Whatever the underlying bug turns out to be,
        # the UI's job is to say so, not to go silent.
        try:
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
        except Exception:  # noqa: BLE001 - the wizard must never fail silently and invisibly
            logger.error("Failed to show wizard step %s", self.wizard.step, exc_info=True)
            QMessageBox.critical(
                self,
                "Something went wrong",
                "This step ran into an unexpected problem and couldn't be shown.\n\n"
                "Check the app's log file for details. Try again, or restart the app if it "
                "keeps happening.",
            )

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
        # A resumed profile has no live TileLearner left over from its original calibration
        # session -- only the templates it produced. Put the learner back into steady
        # auto-promote mode from those templates so a brand-new tile tier encountered mid-play
        # (inevitable once merges pass the highest tier seen during calibration) gets learned
        # silently instead of pausing forever with no way to teach it. See
        # app/play_loop.py's _try_learn_unknown_tiles and vision/tile_learning.py's
        # resume_for_play for the mechanism.
        self.wizard.data.tile_learner.resume_for_play()
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

        # Built fresh on every visit to this screen, parented to the disposable `container`
        # (like every other wizard page) rather than kept as a long-lived MainWindow
        # attribute: a previous version reused a single AdvancedPanel instance across visits,
        # but _swap_page()'s deleteLater() on the *previous* container also destroys whatever
        # is still parented to it -- including a reused-and-reparented AdvancedPanel, which
        # left its Python wrapper referencing an already-deleted C++ object on the next visit
        # ("libshiboken: Internal C++ object (AdvancedPanel) already deleted", reproduced by a
        # real user hitting Recalibrate and getting permanently stuck). Sourcing its initial
        # values from self.config -- already kept current via config_changed on every edit --
        # means nothing is lost by rebuilding it each time.
        advanced_panel = AdvancedPanel(self.config, container)
        advanced_panel.config_changed.connect(self._on_config_changed)
        layout.addWidget(advanced_panel)
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

    def _probe_window_bounds(self) -> tuple[int, int, int, int] | None:
        """Re-locate the calibrated window by title and report its *current* bounds.

        Passed to :class:`PlayController` as ``window_bounds_probe`` so it can detect the
        window having moved, been resized, or closed mid-play (Part 5.1) -- matched by title
        rather than a raw window handle/ID, consistent with how the rest of the app already
        identifies "the calibrated window" (see backends/window_enum.py and
        app/profile.py), since IDs aren't guaranteed stable or even available across every
        platform's enumeration backend.
        """
        title = self.wizard.data.window_title
        if title is None:
            return None
        scale = self.wizard.data.scale_factor
        try:
            for _window_id, win_title, left, top, width, height, _pid in enumerate_windows():
                if win_title == title:
                    return int(left * scale), int(top * scale), int(width * scale), int(height * scale)
        except Exception:  # noqa: BLE001 - a failed probe should pause play, not crash it
            logger.warning("Failed to probe the calibrated window's current position.", exc_info=True)
            return None
        return None

    def _start_playing(self) -> None:
        try:
            self._start_playing_unsafe()
        except Exception:  # noqa: BLE001 - see _enter_current_step's docstring comment
            logger.error("Failed to start playing.", exc_info=True)
            QMessageBox.critical(
                self,
                "Couldn't start",
                "Something went wrong trying to start. Check the app's log file for details.",
            )

    def _start_playing_unsafe(self) -> None:
        # Constructed here, on the main/GUI thread, so its Qt thread affinity is the main
        # thread -- required for the cross-thread marshaling in MainThreadInputBackend to land
        # calls where macOS needs them. See ui/main_thread_input.py.
        raw_input_backend = create_input_backend(use_pydirectinput=self.wizard.data.use_pydirectinput)
        input_backend = MainThreadInputBackend(raw_input_backend, self)
        self.hud = PlayHud()
        self.hud.pause_button.clicked.connect(self._toggle_pause)
        self.hud.stop_button.clicked.connect(self._stop_play)
        self.hud.save_snapshot_button.clicked.connect(self._save_debug_snapshot)

        self.play_controller = PlayController(
            capture_backend=self.capture_backend,
            input_backend=input_backend,
            tile_learner=self.wizard.data.tile_learner,
            roi=self._current_roi(),
            app_config=self.config,
            on_event=self._on_play_event,
            window_bounds_probe=self._probe_window_bounds,
        )
        if sys.platform != "darwin" or self.config.enable_macos_global_hotkeys:
            self.hotkeys.start()
        else:
            logger.info(
                "Skipping global hotkey registration on macOS (see AppConfig."
                "enable_macos_global_hotkeys); use the HUD's Pause/Stop/Save Snapshot buttons instead."
            )
        self.play_controller.start()
        self.hud.show()

    def _on_play_event(self, event: PlayEvent) -> None:
        # Called directly from PlayController on the play loop's background thread; hop onto
        # the main thread via _play_event_received before touching any widget.
        self._play_event_received.emit(event)

    def _handle_play_event(self, event: PlayEvent) -> None:
        if self.hud is not None:
            self.hud.on_play_event(event)
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
        self._save_profile_if_learned_new_tiles()

    def _save_profile_if_learned_new_tiles(self) -> None:
        # PlayController shares wizard.data.tile_learner by reference, so any tiles learned
        # mid-play (see app/play_loop.py's _try_learn_unknown_tiles) are already reflected
        # here; re-saving is cheap and idempotent, so it's simplest to always do it rather
        # than track a dirty flag -- this is what keeps a session's live-learned tiles from
        # being silently lost if the app is closed without recalibrating.
        #
        # Exception: if this session tripped the recognition-anomaly guard (too many "new"
        # tiles in one read -- almost certainly a drifted ROI, not real gameplay), the
        # in-memory recognizer may already contain a handful of bogus templates learned before
        # the anomaly was detected. Writing those over the last-known-good profile would turn a
        # recoverable-by-recalibrating session into a permanently corrupted one every future
        # launch loads. Leave the saved profile untouched in that case.
        if self.play_controller is not None and self.play_controller.had_recognition_anomaly:
            logger.warning(
                "Not saving the profile: this session's recognition looked unreliable "
                "(see the recognition-anomaly pause). Recalibrate to fix it properly."
            )
            return
        try:
            self.wizard.save_profile()
        except RuntimeError:
            logger.debug("Nothing to save (play was stopped before a profile existed).", exc_info=True)

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
