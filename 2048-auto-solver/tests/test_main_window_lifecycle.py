"""Regression tests for ui/main_window.py's widget lifecycle and error handling.

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
