"""Entry point: launches the wizard/play GUI. Packaged into the per-OS binary by PyInstaller
(see .github/workflows/build.yml and the PyInstaller spec files alongside this module).
"""

from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from app.config import get_log_dir


def _configure_logging() -> None:
    log_path = get_log_dir() / "app.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
    )


def main() -> int:
    _configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting 2048 Auto-Solver")

    app = QApplication(sys.argv)
    app.setApplicationName("2048 Auto-Solver")
    app.setQuitOnLastWindowClosed(True)

    from ui.main_window import MainWindow

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
