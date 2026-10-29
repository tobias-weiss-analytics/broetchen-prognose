import time

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")


def test_main_window_trains_and_forecasts():
    from broetchen.ui.main_window import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    deadline = time.time() + 120
    while win.result is None and time.time() < deadline:
        app.processEvents()
        time.sleep(0.05)
    assert win.result is not None
    assert win.plan is not None and len(win.plan) == 14 * 3
    win.spn_horizon.setValue(7)
    assert len(win.plan) == 7 * 3
    win.close()
