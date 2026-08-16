"""macOS-only wizard step: explain, deep-link, poll, and offer a one-click relaunch.

Never instantiated off-macOS; ``ui/main_window.py`` skips straight past this step there (see
``app/state_machine.py``'s platform check).
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)


class _PermissionPollThread(QThread):
    granted = Signal()

    def __init__(self, kind, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._kind = kind

    def run(self) -> None:
        from backends.permissions_macos import poll_until_granted

        if poll_until_granted(self._kind, timeout_seconds=300.0):
            self.granted.emit()


class PermissionRow(QWidget):
    """One permission's explanation, status, and "Open Settings" button."""

    granted_changed = Signal()

    def __init__(self, status, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._status = status
        self._poll_thread: _PermissionPollThread | None = None
        layout = QVBoxLayout(self)

        title = "Screen Recording" if status.kind.value == "screen_recording" else "Accessibility"
        self._title_label = QLabel(f"<b>{title}</b>", self)
        layout.addWidget(self._title_label)

        self._explanation_label = QLabel(status.explanation, self)
        self._explanation_label.setWordWrap(True)
        layout.addWidget(self._explanation_label)

        self._status_label = QLabel(self)
        layout.addWidget(self._status_label)

        self._open_settings_button = QPushButton("Open Settings", self)
        self._open_settings_button.clicked.connect(self._open_settings)
        layout.addWidget(self._open_settings_button)

        self._refresh_status_label()

    def _refresh_status_label(self) -> None:
        if self._status.granted:
            self._status_label.setText("Granted")
            self._status_label.setStyleSheet("color: green;")
            self._open_settings_button.setEnabled(False)
        else:
            self._status_label.setText("Not granted yet")
            self._status_label.setStyleSheet("color: #cc6600;")

    def _open_settings(self) -> None:
        from backends.permissions_macos import open_settings_pane

        open_settings_pane(self._status.settings_url)
        self._start_polling()

    def _start_polling(self) -> None:
        if self._poll_thread is not None and self._poll_thread.isRunning():
            return
        self._poll_thread = _PermissionPollThread(self._status.kind, self)
        self._poll_thread.granted.connect(self._on_granted)
        self._poll_thread.start()

    def _on_granted(self) -> None:
        self._status = self._status.__class__(
            kind=self._status.kind,
            granted=True,
            explanation=self._status.explanation,
            settings_url=self._status.settings_url,
        )
        self._refresh_status_label()
        self.granted_changed.emit()


class PermissionsWizardPage(QWidget):
    """The permissions explanation screen: one :class:`PermissionRow` per required permission,
    plus a relaunch button that activates once everything is granted."""

    all_granted = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from backends.permissions_macos import get_permission_statuses

        self._statuses = get_permission_statuses()
        layout = QVBoxLayout(self)

        heading = QLabel(
            "<h2>Before we start</h2><p>This app needs two permissions to watch your game "
            "and play it for you. Nothing else on your Mac is affected.</p>",
            self,
        )
        heading.setWordWrap(True)
        layout.addWidget(heading)

        self._rows = [PermissionRow(status, self) for status in self._statuses]
        for row in self._rows:
            row.granted_changed.connect(self._check_all_granted)
            layout.addWidget(row)

        self._relaunch_button = QPushButton("Relaunch now", self)
        self._relaunch_button.setEnabled(all(s.granted for s in self._statuses))
        self._relaunch_button.clicked.connect(self._relaunch)
        layout.addWidget(self._relaunch_button)

    def _check_all_granted(self) -> None:
        from backends.permissions_macos import get_permission_statuses

        statuses = get_permission_statuses()
        if all(s.granted for s in statuses):
            self._relaunch_button.setEnabled(True)
            self.all_granted.emit()

    def _relaunch(self) -> None:
        from backends.permissions_macos import relaunch_app

        relaunch_app()
