"""Regression tests for the Qt UI layer (ui/main_window.py, ui/wizard_grid_confirm.py):
widget lifecycle, error handling, and wizard-page behavior.

Runs Qt in offscreen mode (no real display server needed) by setting QT_QPA_PLATFORM before
PySide6 is imported anywhere in the process -- this module must stay the only one in the suite
that touches PySide6, since the platform plugin can only be selected once per process. A fake
capture backend (MainWindow's capture_backend is injectable specifically for this) means these
tests don't need a real screen either, unlike backends.capture.mss's default, which needs a
live display connection just to construct.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from app.profile import ProfileStore
from app.state_machine import WizardStep
from backends.capture.base import CaptureRegion
from backends.window_enum import WindowInfo


class _FakeCaptureBackend:
    def list_monitors(self):
        return []

    def get_scale_factor(self, monitor_index: int = 0) -> float:
        return 1.0

    def grab(self, region: CaptureRegion):
        return np.zeros((region.height, region.width, 3), dtype=np.uint8)

    def close(self) -> None:
        pass


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window_info() -> WindowInfo:
    return WindowInfo(
        window_id=1,
        title="2048 Test Window",
        left=50,
        top=60,
        width=400,
        height=400,
        thumbnail=None,
        match_score=1.0,
        pid=None,
    )


def test_revisiting_ready_to_play_repeatedly_does_not_crash(qapp, tmp_path, window_info) -> None:
    """Regression test for a real crash report: recalibrating and landing back on the
    "ready to play" screen a second time raised
    'RuntimeError: libshiboken: Internal C++ object (AdvancedPanel) already deleted.'

    Root cause: a single AdvancedPanel instance was built once in MainWindow.__init__ and
    reparented into a disposable container every time the ready-to-play screen was shown;
    _swap_page()'s deleteLater() on the *previous* container also destroyed whatever was still
    parented to it -- including the reused-and-reparented AdvancedPanel -- so the second visit
    held a Python reference to an already-destroyed C++ object. The fix builds a fresh
    AdvancedPanel on every visit, parented to that visit's own disposable container, matching
    every other wizard page.
    """
    from app.profile import GameProfile, RoiOffset
    from ui.main_window import MainWindow
    from vision.recognition import make_template

    store = ProfileStore(tmp_path)
    # Not under test here (see test_arrange_screen_step_is_shown_once_then_skipped for that) --
    # skip straight past the one-time screen-arrangement step so this test can isolate the
    # widget-lifecycle bug it's actually about.
    store.mark_screen_setup_complete()
    templates = {1: make_template(1, np.full((64, 64, 3), (200, 190, 180), dtype=np.uint8))}
    profile = GameProfile(
        window_title=window_info.title,
        roi=RoiOffset(0, 0, 320, 320),
        window_size_at_calibration=(400, 400),
        tile_templates=templates,
        settle_frames=1,
        settle_timeout_ms=300.0,
        confidence_threshold=0.6,
    )
    store.save(profile)

    window = MainWindow(profile_store=store, capture_backend=_FakeCaptureBackend())
    try:
        window._on_window_selected(window_info)
        assert window.wizard.step is WizardStep.READY_TO_PLAY

        # Revisit the ready-to-play screen several times (as happens naturally via
        # Stop -> Start, or navigating away and back), each with a real event-loop pump so the
        # deferred deletion that caused the crash actually gets a chance to run (a plain
        # QApplication.processEvents() loop doesn't reliably trigger it; QTest.qWait does).
        # Recalibrate isn't used to loop back here since it deliberately routes through the
        # full wizard now (see test_recalibrate_actually_redoes_the_wizard_instead_of_bouncing_back);
        # jumping wizard.step directly isolates this test to the widget-lifecycle bug alone.
        for _ in range(4):
            QTest.qWait(200)
            window.wizard.step = WizardStep.READY_TO_PLAY
            window._enter_current_step()
            assert window.wizard.step is WizardStep.READY_TO_PLAY
    finally:
        window.close()


def test_step_handler_exception_shows_message_instead_of_crashing(qapp, tmp_path, monkeypatch) -> None:
    """A step-handler exception must be caught, logged, and surfaced to the user -- not left
    to escape silently. Before this fix, a real user hit exactly this: an unhandled exception
    printed only to a terminal they weren't watching, leaving the app looking frozen with zero
    on-screen feedback about what had gone wrong.
    """
    from ui.main_window import MainWindow

    window = MainWindow(profile_store=ProfileStore(tmp_path), capture_backend=_FakeCaptureBackend())
    try:
        shown = []
        monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: shown.append((a, k))))

        def boom() -> None:
            raise RuntimeError("simulated internal bug")

        window._show_window_picker_step = boom
        window.wizard.step = WizardStep.PICK_WINDOW

        window._enter_current_step()  # must not raise

        assert len(shown) == 1
    finally:
        window.close()


def test_recalibrate_actually_redoes_the_wizard_instead_of_bouncing_back(qapp, tmp_path, window_info) -> None:
    """Regression test for a real report: clicking Recalibrate, then picking the same window
    again, landed straight back on the "ready to play" screen with the old profile still
    loaded -- Recalibrate was a silent no-op. Root cause: _on_window_selected() always checked
    for and auto-resumed an existing saved profile match, with no way to tell "the user just
    clicked Recalibrate for this exact window" apart from an ordinary first-time selection.
    """
    from app.profile import GameProfile, RoiOffset
    from ui.main_window import MainWindow
    from vision.recognition import make_template

    store = ProfileStore(tmp_path)
    store.mark_screen_setup_complete()  # not under test here; see the ARRANGE_SCREEN-specific test
    # A profile with templates already present, standing in for the corrupted profile a real
    # user would be trying to get away from by recalibrating.
    templates = {t: make_template(t, np.full((64, 64, 3), (200 - t, 190, 180), dtype=np.uint8)) for t in range(1, 4)}
    profile = GameProfile(
        window_title=window_info.title,
        roi=RoiOffset(0, 0, 320, 320),
        window_size_at_calibration=(400, 400),
        tile_templates=templates,
        settle_frames=1,
        settle_timeout_ms=300.0,
        confidence_threshold=0.6,
    )
    store.save(profile)

    window = MainWindow(profile_store=store, capture_backend=_FakeCaptureBackend())
    try:
        # Ordinary first selection: normal auto-resume behavior must still work.
        window._on_window_selected(window_info)
        assert window.wizard.step is WizardStep.READY_TO_PLAY
        assert set(window.wizard.data.tile_learner.recognizer.templates) == {1, 2, 3}

        # Recalibrate, then pick the *same* window again -- exactly what the user did.
        window._recalibrate()
        assert window.wizard.step is WizardStep.PICK_WINDOW
        assert window._force_recalibration is True

        window._on_window_selected(window_info)

        assert window.wizard.step is WizardStep.CONFIRM_GRID, (
            "must go through grid confirmation and tile learning again, not bounce back to "
            "ready-to-play with the old profile"
        )
        assert window._force_recalibration is False, "the flag must not leak into later selections"
        assert window.wizard.data.tile_learner.recognizer.templates == {}, (
            "recalibrating must start tile learning from a clean slate, not carry over the "
            "old (possibly corrupted) templates"
        )
    finally:
        window.close()


def test_low_confidence_grid_detection_hides_looks_right_and_requests_the_overlay(qapp) -> None:
    """Regression test for a real report: a "container" (non-lattice) grid detection
    confidently proposed almost the entire captured window as the board, and nothing in the
    confirmation screen stopped the user from proceeding with it -- every tile learned
    afterward was noise from the page around the actual board, not the board itself. A
    non-lattice detection must hide "Looks right" (so it can't be confirmed unchecked) and
    request the live on-screen overlay -- see MainWindow._show_grid_confirm_step, which opens
    it automatically for exactly this case rather than requiring an extra click.
    """
    from ui.wizard_grid_confirm import GridConfirmPage

    frame = np.full((800, 1200, 3), (240, 235, 225), dtype=np.uint8)

    low_confidence_page = GridConfirmPage(frame, 0, 0, 1200, 800, detection_method="failed")
    assert low_confidence_page._needs_manual_check is True
    assert low_confidence_page._looks_right_button.isVisibleTo(low_confidence_page) is False

    high_confidence_page = GridConfirmPage(frame, 100, 100, 400, 400, detection_method="lattice")
    assert high_confidence_page._needs_manual_check is False
    assert high_confidence_page._looks_right_button.isVisibleTo(high_confidence_page) is True

    # Confirming without adjustment still works for the trustworthy (lattice) case, using the
    # auto-detected box translated back into full-frame coordinates.
    confirmed = []
    high_confidence_page.grid_confirmed.connect(lambda *args: confirmed.append(args))
    high_confidence_page._looks_right_button.click()
    assert confirmed == [(100, 100, 400, 400)]

    # "Adjust on my screen" is available either way, as an escape hatch for a lattice detection
    # that's slightly off too.
    requests = []
    high_confidence_page.adjust_on_screen_requested.connect(lambda: requests.append(True))
    high_confidence_page._adjust_button.click()
    assert requests == [True]


def test_screen_region_overlay_confirms_in_absolute_screen_coordinates(qapp) -> None:
    """Regression test for direct user feedback: dragging a box on a small, scaled-down static
    screenshot was fiddly and imprecise. The overlay is drawn directly on the real screen
    instead, working throughout in absolute screen-logical (QScreen.geometry()) coordinates --
    verify a confirmed rectangle round-trips back to the same absolute coordinates it was
    seeded with.
    """
    from PySide6.QtCore import QRect
    from ui.screen_overlay import ScreenRegionOverlay

    screen_geometry = QRect(100, 50, 1200, 800)  # a non-(0,0)-origin monitor, e.g. a secondary display
    initial_rect = QRect(300, 200, 400, 300)  # absolute screen coordinates, already inside the screen

    overlay = ScreenRegionOverlay(screen_geometry, initial_rect)
    try:
        confirmed = []
        overlay.region_confirmed.connect(lambda r: confirmed.append(r))
        overlay._confirm()
        assert confirmed == [initial_rect]

        cancelled = []
        overlay.cancelled.connect(lambda: cancelled.append(True))
        overlay._cancel()
        assert cancelled == [True]
    finally:
        overlay.close()


def test_open_screen_overlay_round_trips_through_window_and_scale_conversion(qapp, tmp_path, monkeypatch) -> None:
    """Regression test for a real report: after grid auto-detection failed repeatedly, the
    static-screenshot preview showed only a tiny sliver of real content -- symptomatic of a
    capture-region bug, and exactly the kind of geometry math this test pins down. The detected
    box (physical pixels relative to the window's own origin, what vision.grid_detect works in)
    must convert to the overlay's absolute screen-logical coordinates and back to *exactly* the
    original numbers when the box isn't moved, through the window's logical bounds and DPI
    scale factor both ways.
    """
    from ui.main_window import MainWindow
    from vision.grid_detect import GridDetectionResult

    store = ProfileStore(tmp_path)
    store.mark_screen_setup_complete()
    window = MainWindow(profile_store=store, capture_backend=_FakeCaptureBackend())
    try:
        window.wizard.data.window_bounds = (50, 60, 400, 400)  # logical
        window.wizard.data.scale_factor = 2.0
        window.wizard.data.grid_detection = GridDetectionResult(
            left=40, top=80, width=320, height=320, cells=(), confidence=0.0, method="failed",
        )

        confirmed = []
        monkeypatch.setattr(window, "_on_grid_confirmed", lambda l, t, w, h: confirmed.append((l, t, w, h)))

        window._open_screen_overlay()
        assert window._screen_overlay is not None

        window._screen_overlay._confirm()  # confirm without moving the box

        assert confirmed == [(40, 80, 320, 320)]
        assert window._screen_overlay is None, "must close itself once confirmed"
    finally:
        window.close()


def test_screen_overlay_covers_only_the_window_area_not_the_whole_screen(qapp, tmp_path) -> None:
    """Regression test for a real report with a screenshot: with the overlay covering the
    *entire* screen, this app's own window -- which had ended up sitting over part of the game
    -- became completely unreachable, because the full-screen, always-on-top, click-capturing
    overlay was swallowing every click on the whole screen, including ones meant for that
    unrelated window. The overlay must only cover the target window's own area plus padding, not
    the whole screen, so anything positioned outside that (including this app's own window, once
    it isn't overlapping the game) stays clickable while adjusting.
    """
    from PySide6.QtWidgets import QApplication
    from ui.main_window import MainWindow
    from vision.grid_detect import GridDetectionResult

    store = ProfileStore(tmp_path)
    store.mark_screen_setup_complete()
    window = MainWindow(profile_store=store, capture_backend=_FakeCaptureBackend())
    try:
        window.wizard.data.window_bounds = (50, 60, 400, 400)
        window.wizard.data.scale_factor = 1.0
        window.wizard.data.grid_detection = GridDetectionResult(
            left=40, top=80, width=320, height=320, cells=(), confidence=0.0, method="failed",
        )

        window._open_screen_overlay()
        overlay_geometry = window._screen_overlay.geometry()
        screen_geometry = QApplication.primaryScreen().geometry()

        assert overlay_geometry.width() < screen_geometry.width() or overlay_geometry.height() < screen_geometry.height(), (
            "the overlay must not cover the entire screen"
        )
        # Still generous enough to drag the box well outside a badly-off detection.
        assert overlay_geometry.width() >= 400 + 200
        assert overlay_geometry.height() >= 400 + 200
    finally:
        window.close()


def test_tile_learning_page_shows_preview_and_blocks_on_mid_game_anomaly(qapp) -> None:
    """Regression test for a real report (with screenshot): the tile-learning screen showed 14
    sequential "Learned a new tile" toasts with no indication anything was wrong, no live view
    of what was actually being captured, and let the user proceed straight to Start with a
    profile built from what turned out to be webpage noise, not real tiles.
    """
    from backends.capture.base import CaptureRegion
    from ui.wizard_tile_learning import TileLearningPage
    from vision.capture import Roi
    from vision.tile_learning import TileLearner

    class _NoiseCaptureBackend:
        def grab(self, region: CaptureRegion):
            return np.full((320, 320, 3), (200, 190, 180), dtype=np.uint8)

    learner = TileLearner()
    page = TileLearningPage(_NoiseCaptureBackend(), Roi(0, 0, 320, 320), learner)

    # The live preview must actually update from a poll -- this is the concrete "show what
    # you're looking at" fix, not just a cosmetic placeholder.
    assert page._preview_label.pixmap().isNull()
    page._on_start_new_game()
    page._poll_once()
    assert not page._preview_label.pixmap().isNull()

    recalibrate_requests = []
    page.recalibrate_requested.connect(lambda: recalibrate_requests.append(True))

    for i in range(14):
        rng = np.random.default_rng(i)
        crop = rng.integers(30, 220, (64, 64, 3), dtype=np.uint8)
        learner._observe_mid_game([crop])
    page._finish_mid_game_observation()

    assert page._anomaly_label.text() != ""
    assert page._fix_grid_button.isVisibleTo(page)
    assert page._finish_button.isEnabled() is False

    page._fix_grid_button.click()
    assert recalibrate_requests == [True]


def test_tile_learning_page_shows_a_live_recognized_grid_not_just_toasts(qapp) -> None:
    """Regression test for direct user feedback: "why not display the game grid with
    identified tiles instead of showing traces?" -- the tile-learning page must show a real
    4x4 grid of what's currently recognized in each cell, updated live from each poll, not
    just a scrolling list of "Learned a new tile" toasts with no spatial context.
    """
    import cv2

    from backends.capture.base import CaptureRegion
    from ui.wizard_tile_learning import TileLearningPage
    from vision.capture import Roi, flatten_cells, split_cells
    from vision.recognition import make_template
    from vision.tile_learning import TileLearner

    def _frame_with_one_tile() -> np.ndarray:
        # A flat color alone reads as an empty background cell (see is_empty_cell's std
        # threshold); a real tile sprite always has some internal texture, so stamp a digit
        # onto it, matching how test_play_loop_integration.py's synthetic board renders tiles.
        frame = np.full((320, 320, 3), (187, 173, 160), dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (80, 80), (200, 190, 180), -1)
        cv2.putText(frame, "2", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (50, 50, 50), 3, cv2.LINE_AA)
        return frame

    class _KnownTileCaptureBackend:
        def grab(self, region: CaptureRegion):
            return _frame_with_one_tile()

    seed_crop = flatten_cells(split_cells(_frame_with_one_tile()))[0]

    learner = TileLearner()
    learner.recognizer.add_template(make_template(1, seed_crop))
    page = TileLearningPage(_KnownTileCaptureBackend(), Roi(0, 0, 320, 320), learner)

    # Before any poll, the grid must not falsely claim to recognize anything.
    assert all(cell.text() == "" for row in page._recognized_grid._labels for cell in row)

    page._poll_once()

    top_left_label = page._recognized_grid._labels[0][0]
    assert top_left_label.text() == "2", "the known tier-1 tile must show as its board value"
    other_labels = [
        cell for r, row in enumerate(page._recognized_grid._labels) for c, cell in enumerate(row) if (r, c) != (0, 0)
    ]
    assert all(cell.text() == "" for cell in other_labels), "empty cells must show as empty, not unknown"


def test_starting_play_embeds_the_panel_instead_of_a_separate_floating_window(
    qapp, tmp_path, window_info, monkeypatch
) -> None:
    """Regression test for a real report with a screenshot: the play status used to be a
    separate always-on-top floating window (PlayHud) that landed directly on top of several
    board tiles. Since screen capture reads the literal compositor output, that overlap wasn't
    just visually distracting -- the recognizer was reading the floating window's own buttons
    and stats as if they were tiles. The fix removes the separate window entirely: play status
    now lives inside MainWindow's own central stack (ui/play_panel.py's PlayPanel), so it can
    never be positioned on top of the game board by anything other than the user dragging this
    whole app window there themselves.
    """
    import ui.main_window as main_window_module
    from app.config import AppConfig
    from app.profile import GameProfile, RoiOffset
    from ui.main_window import MainWindow
    from ui.play_panel import PlayPanel
    from vision.recognition import make_template

    class _NoOpInputBackend:
        def tap_key(self, key, hold_ms: int = 40) -> None:
            pass

        def release_all(self) -> None:
            pass

    # The real pynput-backed input backend needs a live X/Wayland display to even construct
    # (raises ImportError otherwise, as it does in this offscreen test environment); swapping
    # it for a no-op keeps this test about panel embedding, not display availability. dry_run
    # additionally guarantees tap_key is never called at all regardless.
    monkeypatch.setattr(main_window_module, "create_input_backend", lambda use_pydirectinput=False: _NoOpInputBackend())

    store = ProfileStore(tmp_path)
    store.mark_screen_setup_complete()
    templates = {1: make_template(1, np.full((64, 64, 3), (200, 190, 180), dtype=np.uint8))}
    profile = GameProfile(
        window_title=window_info.title,
        roi=RoiOffset(0, 0, 320, 320),
        window_size_at_calibration=(400, 400),
        tile_templates=templates,
        settle_frames=1,
        settle_timeout_ms=300.0,
        confidence_threshold=0.6,
    )
    store.save(profile)

    window = MainWindow(profile_store=store, capture_backend=_FakeCaptureBackend())
    window.config = AppConfig(dry_run=True, settle_timeout_ms=200.0, settle_frames=1)
    # Global hotkey registration also needs a live X/Wayland display (pynput.keyboard's
    # GlobalHotKeys) for the same reason the input backend does; irrelevant to what this test
    # is checking, so it's stubbed out rather than left to fail into a blocking QMessageBox.
    window.hotkeys.start = lambda: None
    try:
        window._on_window_selected(window_info)
        assert window.wizard.step is WizardStep.READY_TO_PLAY

        before = {id(w) for w in QApplication.topLevelWidgets()}
        window._start_playing()
        after = {id(w) for w in QApplication.topLevelWidgets()}

        assert isinstance(window.play_panel, PlayPanel)
        assert window._stack.currentWidget() is window.play_panel, "must be embedded in this window's own stack"
        assert after == before, "starting play must not create any new top-level window"

        window._stop_play()
        assert window.play_panel is None
        assert window.wizard.step is WizardStep.READY_TO_PLAY, "Stop must return to the ready-to-play screen"
    finally:
        window.close()


def test_arrange_screen_step_is_shown_once_then_skipped(qapp, tmp_path) -> None:
    """Regression test for direct user feedback: the app must guide the user to lay out their
    screen (game + this app side by side, never overlapping) once, up front -- but per the
    wizard's "second run must be zero-setup" rule, it must never ask again once that's done.
    """
    from ui.main_window import MainWindow
    from ui.wizard_arrange_screen import ArrangeScreenPage

    store = ProfileStore(tmp_path)
    assert store.has_completed_screen_setup() is False

    window = MainWindow(profile_store=store, capture_backend=_FakeCaptureBackend())
    try:
        assert window.wizard.step is WizardStep.ARRANGE_SCREEN
        page = window._stack.currentWidget()
        assert isinstance(page, ArrangeScreenPage)

        page.arranged.emit()
        assert window.wizard.step is WizardStep.PICK_WINDOW
        assert store.has_completed_screen_setup() is True
    finally:
        window.close()

    # A fresh MainWindow against the same (now-marked) store must skip straight past it.
    second_window = MainWindow(profile_store=store, capture_backend=_FakeCaptureBackend())
    try:
        assert second_window.wizard.step is WizardStep.PICK_WINDOW
    finally:
        second_window.close()


def test_tile_learning_page_shows_tile_thumbnails_and_lets_user_reject_a_wrong_ranking(qapp) -> None:
    """Regression test for direct user feedback: the tile-learning confirmation only ever
    showed a tier number in a text toast, and the mid-game ranking's one confirmation step
    could only ever be accepted -- "the only option for the user is to accept the tiles - while
    not understanding if they are correct or not." Every learned tile must now show the actual
    picture that was learned, and a wrong ranking must be rejectable and redoable rather than
    something the user is stuck with.
    """
    from backends.capture.base import CaptureRegion
    from ui.wizard_tile_learning import TileLearningPage
    from vision.capture import Roi
    from vision.tile_learning import TileLearner

    class _StubCaptureBackend:
        def grab(self, region: CaptureRegion):
            return np.full((320, 320, 3), (200, 190, 180), dtype=np.uint8)

    learner = TileLearner()
    page = TileLearningPage(_StubCaptureBackend(), Roi(0, 0, 320, 320), learner)

    page._on_start_mid_game()
    for i in range(3):
        rng = np.random.default_rng(i)
        crop = rng.integers(30, 220, (64, 64, 3), dtype=np.uint8)
        learner._observe_mid_game([crop])
    page._finish_mid_game_observation()

    # Every learned tile shows the actual picture, not just a tier number in text.
    assert page._toast_list.count() == 3
    for i in range(3):
        assert not page._toast_list.item(i).icon().isNull()

    assert page._confirm_no_button.isVisibleTo(page) is True
    assert set(learner.recognizer.templates) == {1, 2, 3}

    page._confirm_no_button.click()

    # Rejecting must undo exactly what this batch learned and let the user redo it, not leave
    # a possibly-wrong ranking as the only option on the table.
    assert learner.recognizer.templates == {}
    assert page._confirm_no_button.isVisibleTo(page) is False
    assert page._confirm_yes_button.isVisibleTo(page) is False
    assert page._finish_button.isEnabled() is False
    assert page._toast_list.count() == 0
    assert learner.phase.name == "MID_GAME_FALLBACK", "must have restarted observation, not just cleared state"
