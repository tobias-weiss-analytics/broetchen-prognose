"""Programmstart."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from . import __version__
from .ui.main_window import RESOURCES, MainWindow


def run(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    app = QApplication(argv)
    app.setApplicationName("Brötchen-Prognose")
    app.setApplicationVersion(__version__)
    qss = RESOURCES / "app.qss"
    if qss.exists():
        app.setStyleSheet(qss.read_text(encoding="utf-8"))
    data = Path(argv[1]) if len(argv) > 1 else None
    window = MainWindow(data) if data else MainWindow()
    window.show()
    return app.exec()
