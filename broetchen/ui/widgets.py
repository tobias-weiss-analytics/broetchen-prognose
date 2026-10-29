"""Wiederverwendbare Bausteine: Diagramm-Fläche, Tabellenmodell, Hintergrund-Worker."""

from __future__ import annotations

import numbers
import traceback
from collections.abc import Callable

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QAbstractTableModel, QObject, QRunnable, Qt, Signal

BG = "#1e1f22"
WEEKDAYS = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
DARK_STYLE = {
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": "#3a3d42", "axes.labelcolor": "#d7dae0", "text.color": "#e6e6e6",
    "xtick.color": "#9aa3af", "ytick.color": "#9aa3af", "grid.color": "#3a3d42",
    "axes.titlesize": 11, "axes.titleweight": "bold", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
}


class ChartCanvas(FigureCanvasQTAgg):
    """Matplotlib-Fläche im dunklen App-Stil."""

    def __init__(self, height: float = 3.2):
        super().__init__(Figure(figsize=(8, height), dpi=100, facecolor=BG, layout="constrained"))
        self.setMinimumHeight(220)

    def draw_with(self, func: Callable, *args) -> None:
        self.figure.clear()
        with plt.rc_context(DARK_STYLE):
            ax = self.figure.add_subplot(111)
            func(ax, *args)
        self.draw_idle()

    def message(self, text: str) -> None:
        self.figure.clear()
        self.figure.text(0.5, 0.5, text, ha="center", va="center", color="#7b7f86", fontsize=11)
        self.draw_idle()


class DataFrameModel(QAbstractTableModel):
    """Zeigt einen DataFrame in einer QTableView an (nur lesend, mit Formatierung)."""

    def __init__(self, df: pd.DataFrame | None = None, formats: dict[str, str] | None = None):
        super().__init__()
        self._df = pd.DataFrame() if df is None else df
        self._formats = formats or {}

    def set_frame(self, df: pd.DataFrame, formats: dict[str, str] | None = None) -> None:
        self.beginResetModel()
        self._df = df.reset_index(drop=True)
        if formats is not None:
            self._formats = formats
        self.endResetModel()

    def rowCount(self, parent=None):  # noqa: N802 - Qt-API
        return len(self._df)

    def columnCount(self, parent=None):  # noqa: N802
        return len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        col = self._df.columns[index.column()]
        value = self._df.iat[index.row(), index.column()]
        if role == Qt.DisplayRole:
            if isinstance(value, pd.Timestamp):
                return f"{WEEKDAYS[value.weekday()]} {value:%d.%m.%Y}"
            fmt = self._formats.get(col)
            if fmt and pd.notna(value):
                return fmt.format(value).replace(",", " ").replace(".", ",").replace(" ", ".")
            return "" if pd.isna(value) else str(value)
        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignRight | Qt.AlignVCenter) if isinstance(value, numbers.Number) else int(Qt.AlignLeft | Qt.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802
        if role != Qt.DisplayRole:
            return None
        return str(self._df.columns[section]) if orientation == Qt.Horizontal else str(section + 1)


class WorkerSignals(QObject):
    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)


class Worker(QRunnable):
    """Führt eine Funktion im Hintergrund aus, damit die Oberfläche bedienbar bleibt."""

    def __init__(self, fn: Callable, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            result = self.fn(*self.args, progress=self.signals.progress.emit, **self.kwargs)
        except Exception as exc:  # noqa: BLE001 - wird dem Nutzer angezeigt
            self.signals.failed.emit(f"{exc}\n\n{traceback.format_exc(limit=3)}")
        else:
            self.signals.finished.emit(result)
