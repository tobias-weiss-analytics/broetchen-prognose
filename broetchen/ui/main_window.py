"""Hauptfenster der Brötchen-Prognose."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QStandardPaths, Qt, QThreadPool
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .. import __author__, __version__, charts
from ..data import SAMPLE_DATA, DataError, holiday_names, load_sales
from ..insights import driver_effects, summary_text, weekday_profile
from ..model import MODELS, TrainResult, default_assumptions, forecast, naive_forecast, train
from ..planning import Costs, build_plan, compare_strategies, safety_factor
from ..report import ReportInput, export_report
from .widgets import ChartCanvas, DataFrameModel, Worker

RESOURCES = Path(__file__).resolve().parent.parent / "resources"


def de(x: float, digits: int = 1, sign: bool = False) -> str:
    """Zahl im deutschen Format, z. B. 1.234,5."""
    text = f"{x:{'+' if sign else ''},.{digits}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def _table_view(model: DataFrameModel) -> QTableView:
    view = QTableView()
    view.setModel(model)
    view.setAlternatingRowColors(True)
    view.setSelectionBehavior(QAbstractItemView.SelectRows)
    view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    view.horizontalHeader().setStretchLastSection(True)
    view.verticalHeader().setVisible(False)
    return view


class MainWindow(QMainWindow):
    def __init__(self, data_path: Path | None = SAMPLE_DATA):
        super().__init__()
        self.setWindowTitle(f"Brötchen-Prognose {__version__}")
        self.setWindowIcon(QIcon(str(RESOURCES / "app.ico")))
        self.resize(1360, 860)

        self.data: pd.DataFrame | None = None
        self.source_name = ""
        self.result: TrainResult | None = None
        self._forecast: pd.DataFrame | None = None
        self.strategies: pd.DataFrame | None = None
        self.plan: pd.DataFrame | None = None
        self.effects: pd.DataFrame | None = None
        self.pool = QThreadPool.globalInstance()
        self._worker: Worker | None = None

        self._build_ui()
        self._set_busy(False)
        if data_path is not None and Path(data_path).exists():
            self.load_data(Path(data_path))
            self.start_training()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_sidebar())
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_data_tab(), "Daten")
        self.tabs.addTab(self._build_forecast_tab(), "Prognose && Backplan")
        self.tabs.addTab(self._build_quality_tab(), "Modellgüte")
        self.tabs.addTab(self._build_drivers_tab(), "Treiber")
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 1020])
        self.setCentralWidget(splitter)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumWidth(160)
        self.statusBar().addPermanentWidget(self.progress)

        open_action = QAction("CSV öffnen …", self, shortcut=QKeySequence.Open, triggered=self.on_open)
        report_action = QAction("PDF-Report exportieren …", self, shortcut=QKeySequence("Ctrl+P"), triggered=self.on_report)
        quit_action = QAction("Beenden", self, shortcut=QKeySequence.Quit, triggered=self.close)
        about_action = QAction("Über", self, triggered=self.on_about)
        menu = self.menuBar().addMenu("&Datei")
        menu.addActions([open_action, report_action])
        menu.addSeparator()
        menu.addAction(quit_action)
        self.menuBar().addMenu("&Hilfe").addAction(about_action)

    def _build_sidebar(self) -> QWidget:
        side = QWidget()
        lay = QVBoxLayout(side)
        lay.setContentsMargins(10, 10, 10, 10)

        box = QGroupBox("Daten")
        bl = QVBoxLayout(box)
        self.lbl_source = QLabel("Keine Datei geladen")
        self.lbl_source.setWordWrap(True)
        self.btn_open = QPushButton("📁  CSV öffnen …")
        self.btn_open.clicked.connect(self.on_open)
        bl.addWidget(self.lbl_source)
        bl.addWidget(self.btn_open)
        lay.addWidget(box)

        box = QGroupBox("Filialen")
        bl = QVBoxLayout(box)
        self.store_list = QListWidget()
        self.store_list.setMaximumHeight(110)
        self.store_list.itemChanged.connect(self.on_stores_changed)
        bl.addWidget(self.store_list)
        lay.addWidget(box)

        box = QGroupBox("Modell")
        fl = QFormLayout(box)
        self.cmb_model = QComboBox()
        for key, label in MODELS.items():
            self.cmb_model.addItem(label, key)
        self.btn_train = QPushButton("⚙️  Trainieren && prüfen")
        self.btn_train.setProperty("variant", "primary")
        self.btn_train.clicked.connect(self.start_training)
        self.lbl_model = QLabel("–")
        self.lbl_model.setWordWrap(True)
        fl.addRow(self.cmb_model)
        fl.addRow(self.btn_train)
        fl.addRow(self.lbl_model)
        lay.addWidget(box)

        box = QGroupBox("Prognose")
        fl = QFormLayout(box)
        self.spn_horizon = QSpinBox()
        self.spn_horizon.setRange(1, 28)
        self.spn_horizon.setValue(14)
        self.spn_horizon.setSuffix(" Tage")
        self.spn_horizon.valueChanged.connect(self.on_horizon_changed)
        fl.addRow("Zeitraum", self.spn_horizon)
        lay.addWidget(box)

        box = QGroupBox("Kosten je Brötchen")
        fl = QFormLayout(box)
        self.spn_price = self._euro(0.35)
        self.spn_cost = self._euro(0.12)
        self.spn_salvage = self._euro(0.00)
        fl.addRow("Verkaufspreis", self.spn_price)
        fl.addRow("Herstellkosten", self.spn_cost)
        fl.addRow("Restwert", self.spn_salvage)
        self.lbl_costs = QLabel()
        self.lbl_costs.setWordWrap(True)
        fl.addRow(self.lbl_costs)
        lay.addWidget(box)

        self.btn_report = QPushButton("🧾  PDF-Report exportieren …")
        self.btn_report.clicked.connect(self.on_report)
        lay.addWidget(self.btn_report)
        lay.addStretch(1)
        credit = QLabel(f"{__author__} · v{__version__}")
        credit.setStyleSheet("color: #7b7f86; font-size: 9pt;")
        lay.addWidget(credit)

        scroll = QScrollArea()
        scroll.setWidget(side)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(330)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return scroll

    def _euro(self, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0, 5)
        spin.setDecimals(2)
        spin.setSingleStep(0.01)
        spin.setSuffix(" €")
        spin.setValue(value)
        spin.valueChanged.connect(self.on_costs_changed)
        return spin

    def _build_data_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.chart_history = ChartCanvas()
        self.data_model = DataFrameModel()
        split = QSplitter(Qt.Vertical)
        split.addWidget(self.chart_history)
        split.addWidget(_table_view(self.data_model))
        split.setSizes([380, 380])
        lay.addWidget(split)
        return w

    def _build_forecast_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.chart_forecast = ChartCanvas(height=3.6)

        assumptions = QWidget()
        al = QVBoxLayout(assumptions)
        al.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("<b>Annahmen</b> – Aktionen, Ereignisse und Wetter für die kommenden Tage anpassen:")
        hint.setWordWrap(True)
        self.tbl_assumptions = QTableWidget(0, 5)
        self.tbl_assumptions.setHorizontalHeaderLabels(["Datum", "Aktion", "Ereignis", "Temp. °C", "Regen"])
        self.tbl_assumptions.verticalHeader().setVisible(False)
        self.tbl_assumptions.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl_assumptions.horizontalHeader().setStretchLastSection(True)
        self.btn_recalc = QPushButton("🔄  Prognose mit diesen Annahmen berechnen")
        self.btn_recalc.setProperty("variant", "primary")
        self.btn_recalc.clicked.connect(self.run_forecast)
        al.addWidget(hint)
        al.addWidget(self.tbl_assumptions)
        al.addWidget(self.btn_recalc)

        plan_box = QWidget()
        pl = QVBoxLayout(plan_box)
        pl.setContentsMargins(0, 0, 0, 0)
        head = QHBoxLayout()
        head.addWidget(QLabel("<b>Backplan</b> – Prognose und empfohlene Backmenge (Stück)"))
        head.addStretch(1)
        self.btn_csv = QPushButton("Als CSV speichern …")
        self.btn_csv.clicked.connect(self.on_export_plan)
        head.addWidget(self.btn_csv)
        self.plan_model = DataFrameModel()
        pl.addLayout(head)
        pl.addWidget(_table_view(self.plan_model))

        bottom = QSplitter(Qt.Horizontal)
        bottom.addWidget(assumptions)
        bottom.addWidget(plan_box)
        bottom.setSizes([420, 620])
        split = QSplitter(Qt.Vertical)
        split.addWidget(self.chart_forecast)
        split.addWidget(bottom)
        split.setSizes([400, 360])
        lay.addWidget(split)
        return w

    def _build_quality_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.chart_backtest = ChartCanvas()
        self.comparison_model = DataFrameModel()
        self.strategy_model = DataFrameModel()
        tables = QSplitter(Qt.Horizontal)
        for title, model in [("Fehler im Backtest (letzte 8 Wochen, 7 Tage im Voraus)", self.comparison_model),
                             ("Backstrategien im selben Zeitraum", self.strategy_model)]:
            box = QWidget()
            bl = QVBoxLayout(box)
            bl.setContentsMargins(0, 0, 0, 0)
            bl.addWidget(QLabel(f"<b>{title}</b>"))
            bl.addWidget(_table_view(model))
            tables.addWidget(box)
        split = QSplitter(Qt.Vertical)
        split.addWidget(self.chart_backtest)
        split.addWidget(tables)
        split.setSizes([430, 300])
        lay.addWidget(split)
        return w

    def _build_drivers_tab(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        self.chart_effects = ChartCanvas(height=4.5)
        self.chart_weekdays = ChartCanvas(height=2.6)
        self.txt_insights = QTextBrowser()
        right = QSplitter(Qt.Vertical)
        right.addWidget(self.chart_weekdays)
        right.addWidget(self.txt_insights)
        right.setSizes([330, 420])
        lay.addWidget(self.chart_effects, 3)
        lay.addWidget(right, 2)
        return w

    # ------------------------------------------------------------ Helfer
    def _set_busy(self, busy: bool, message: str = "") -> None:
        for widget in (self.btn_open, self.btn_train, self.btn_recalc, self.btn_report, self.btn_csv, self.cmb_model):
            widget.setEnabled(not busy)
        if not busy:
            ready = self.result is not None
            for widget in (self.btn_recalc, self.btn_report, self.btn_csv):
                widget.setEnabled(ready)
        self.progress.setVisible(busy)
        if message:
            self.statusBar().showMessage(message)

    def _error(self, title: str, text: str) -> None:
        QMessageBox.critical(self, title, text)

    def selected_stores(self) -> list[str]:
        items = (self.store_list.item(i) for i in range(self.store_list.count()))
        return [it.text() for it in items if it.checkState() == Qt.Checked]

    def costs(self) -> Costs | None:
        try:
            return Costs(self.spn_price.value(), self.spn_cost.value(), self.spn_salvage.value())
        except ValueError:
            return None

    def _documents(self) -> str:
        return QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)

    # ------------------------------------------------------------- Daten
    def on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Verkaufsdaten öffnen", self._documents(), "CSV-Dateien (*.csv)")
        if path and self.load_data(Path(path)):
            self.start_training()

    def load_data(self, path: Path) -> bool:
        try:
            df = load_sales(path)
        except DataError as exc:
            self._error("Datei passt nicht", str(exc))
            return False
        self.data, self.source_name = df, path.name
        self.result = self.plan = self._forecast = None
        stores = sorted(df["store"].unique())
        self.store_list.blockSignals(True)
        self.store_list.clear()
        for store in stores:
            item = QListWidgetItem(store)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.store_list.addItem(item)
        self.store_list.blockSignals(False)
        self.spn_price.blockSignals(True)
        self.spn_price.setValue(round(float(df.sort_values("date").groupby("store")["price"].last().mean()), 2))
        self.spn_price.blockSignals(False)

        self.lbl_source.setText(
            f"<b>{path.name}</b><br>{df['date'].min():%d.%m.%Y} – {df['date'].max():%d.%m.%Y}<br>"
            f"{len(stores)} Filialen · {len(df):,} Zeilen".replace(",", "."))
        view = df.sort_values(["date", "store"], ascending=[False, True])
        self.data_model.set_frame(view.rename(columns={
            "date": "Datum", "store": "Filiale", "sales": "Absatz", "price": "Preis €", "promo": "Aktion",
            "temp_c": "Temp. °C", "rain_mm": "Regen mm", "event": "Ereignis"}),
            {"Absatz": "{:,.0f}", "Preis €": "{:.2f}", "Temp. °C": "{:.1f}", "Regen mm": "{:.1f}"})
        self.chart_history.draw_with(charts.history, df, stores)
        self.effects = driver_effects(df)
        self.chart_effects.draw_with(charts.effects, self.effects)
        self.chart_weekdays.draw_with(charts.weekday_profile, weekday_profile(df))
        lines = summary_text(df, self.effects)
        self.txt_insights.setHtml("<h3>Was treibt den Absatz?</h3><ul>" + "".join(f"<li>{t}</li>" for t in lines) + "</ul>"
                                  "<p style='color:#9aa3af'>Geschätzt mit einer Regression über alle Filialen; jeder Effekt "
                                  "ist um die übrigen Einflüsse, das Wochenmuster der Filiale und die Saison bereinigt. "
                                  "Feiertage wirken je Filiale verschieden und werden daher getrennt ausgewiesen.</p>")
        for chart in (self.chart_forecast, self.chart_backtest):
            chart.message("Modell wird trainiert …")
        self.fill_assumptions()
        self.on_costs_changed()
        self.statusBar().showMessage(f"{path.name} geladen.")
        return True

    def on_stores_changed(self, _item=None) -> None:
        if self.data is None:
            return
        self.chart_history.draw_with(charts.history, self.data, self.selected_stores() or [])
        if self.result is not None:
            self.run_forecast()

    # ---------------------------------------------------------- Training
    def start_training(self) -> None:
        if self.data is None:
            self._error("Keine Daten", "Bitte zuerst eine CSV-Datei öffnen.")
            return
        key = self.cmb_model.currentData()
        self._set_busy(True, "Training gestartet …")
        self._worker = Worker(train, self.data, key)
        self._worker.signals.progress.connect(self.statusBar().showMessage)
        self._worker.signals.finished.connect(self.on_trained)
        self._worker.signals.failed.connect(self.on_failed)
        self.pool.start(self._worker)

    def on_trained(self, result: TrainResult) -> None:
        self.result = result
        self._set_busy(False)
        err = result.comparison.loc[result.name]
        naive = result.comparison.loc["Vorwoche (gleicher Wochentag)"]
        self.lbl_model.setText(f"Aktiv: <b>{result.name}</b><br>Ø Abweichung {de(err['MAPE'])} % "
                               f"(Regel „wie Vorwoche“: {de(naive['MAPE'])} %)")
        self.comparison_model.set_frame(
            result.comparison.reset_index().rename(columns={"index": "Verfahren", "MAE": "MAE (Stk.)",
                                                            "MAPE": "MAPE (%)", "Bias": "Bias (Stk.)"}),
            {"MAE (Stk.)": "{:.1f}", "MAPE (%)": "{:.1f}", "Bias (Stk.)": "{:+.1f}"})
        self.chart_backtest.draw_with(charts.backtest, result.backtest)
        self.run_forecast()
        self.on_costs_changed()
        self.statusBar().showMessage(f"Fertig: {result.name}, Ø Abweichung im Backtest {de(err['MAPE'])} %.", 8000)

    def on_failed(self, message: str) -> None:
        self._set_busy(False, "Fehler.")
        self._error("Berechnung fehlgeschlagen", message)

    # ---------------------------------------------------------- Prognose
    def on_horizon_changed(self, _value: int) -> None:
        self.fill_assumptions()
        if self.result is not None:
            self.run_forecast()

    def fill_assumptions(self) -> None:
        if self.data is None:
            return
        a = default_assumptions(self.data, self.spn_horizon.value())
        t = self.tbl_assumptions
        t.setRowCount(len(a))
        weekdays = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
        names = holiday_names(a["date"])
        for r, row in a.iterrows():
            label = f"{weekdays[row['date'].weekday()]} {row['date']:%d.%m.}"
            item = QTableWidgetItem(f"{label}  🎉 {names[r]}" if names[r] else label)
            if names[r]:
                item.setToolTip(f"Feiertag: {names[r]} – wird automatisch berücksichtigt")
            item.setData(Qt.UserRole, row["date"])
            item.setFlags(Qt.ItemIsEnabled)
            t.setItem(r, 0, item)
            for c, col in ((1, "promo"), (2, "event")):
                check = QTableWidgetItem()
                check.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                check.setCheckState(Qt.Checked if row[col] else Qt.Unchecked)
                t.setItem(r, c, check)
            t.setItem(r, 3, QTableWidgetItem(f"{row['temp_c']:.1f}"))
            t.setItem(r, 4, QTableWidgetItem(f"{row['rain_mm']:.1f}"))

    def read_assumptions(self) -> pd.DataFrame:
        t = self.tbl_assumptions
        rows = []
        def num(r: int, c: int) -> float:
            text = t.item(r, c).text().replace(",", ".").strip()
            try:
                return float(text)
            except ValueError as exc:
                raise ValueError(f"Zeile {r + 1}: „{t.item(r, c).text()}“ ist keine Zahl.") from exc

        for r in range(t.rowCount()):
            rows.append({
                "date": t.item(r, 0).data(Qt.UserRole),
                "promo": int(t.item(r, 1).checkState() == Qt.Checked),
                "event": int(t.item(r, 2).checkState() == Qt.Checked),
                "temp_c": num(r, 3),
                "rain_mm": max(num(r, 4), 0.0),
            })
        return pd.DataFrame(rows)

    def run_forecast(self) -> None:
        if self.result is None or self.data is None:
            return
        stores = self.selected_stores()
        if not stores:
            self.chart_forecast.message("Bitte mindestens eine Filiale auswählen.")
            self.plan_model.set_frame(pd.DataFrame())
            return
        try:
            assumptions = self.read_assumptions()
        except ValueError as exc:
            self._error("Ungültige Annahme", str(exc))
            return
        self._forecast = forecast(self.data, self.result, assumptions, stores)
        self.update_plan()
        self.tabs.setCurrentIndex(1)

    def on_costs_changed(self, _value: float = 0) -> None:
        costs = self.costs()
        if costs is None:
            self.lbl_costs.setText("<span style='color:#e5484d'>Es muss gelten: Restwert &lt; Herstellkosten &lt; Preis.</span>")
            return
        text = (f"Kritisches Verhältnis: <b>{de(costs.critical_ratio * 100, 0)} %</b><br>"
                f"Ein fehlendes Brötchen kostet {de(costs.underage, 2)} € Gewinn, ein übriges {de(costs.overage, 2)} €.")
        if self.result is not None:
            buffer = (safety_factor(self.result.ratios, costs) - 1) * 100
            text += f"<br>Puffer auf die Prognose: <b>{de(buffer, 1, sign=True)} %</b>"
        self.lbl_costs.setText(text)
        self.update_plan()

    def update_plan(self) -> None:
        costs = self.costs()
        if self.result is None or costs is None or self._forecast is None:
            return
        self.plan = build_plan(self._forecast, self.result.ratios, costs)
        self.chart_forecast.draw_with(charts.forecast, self.data, self.plan, self.plan)
        wide = self.plan.pivot(index="date", columns="store", values=["forecast", "recommended"])
        names = holiday_names(pd.Series(wide.index))
        table = pd.DataFrame({"Datum": wide.index, "Feiertag": names.to_numpy()})
        for store in sorted(self.plan["store"].unique()):
            table[f"{store} Prognose"] = wide[("forecast", store)].round().astype(int).to_numpy()
            table[f"{store} backen"] = wide[("recommended", store)].to_numpy()
        self.plan_model.set_frame(table, {c: "{:,.0f}" for c in table.columns if c not in ("Datum", "Feiertag")})

        bt = self.result.backtest
        naive = naive_forecast(self.data, bt)
        self.strategies = compare_strategies(bt, naive, self.result.ratios, costs)
        self.strategy_model.set_frame(
            self.strategies.reset_index().rename(columns={"index": "Strategie"}),
            {"Gewinn (€)": "{:,.0f}", "Abschriften (Stk.)": "{:,.0f}", "Fehlmenge (Stk.)": "{:,.0f}",
             "Ausverkauft (Tage %)": "{:.0f}"})

    # ----------------------------------------------------------- Export
    def on_export_plan(self) -> None:
        if self.plan is None:
            return
        default = str(Path(self._documents()) / f"Backplan_{date.today():%Y-%m-%d}.csv")
        path, _ = QFileDialog.getSaveFileName(self, "Backplan speichern", default, "CSV-Dateien (*.csv)")
        if path:
            out = self.plan.rename(columns={"date": "Datum", "store": "Filiale", "forecast": "Prognose",
                                            "low": "untere_Grenze", "high": "obere_Grenze", "recommended": "Backmenge"})
            out.round(0).to_csv(path, sep=";", decimal=",", index=False, date_format="%d.%m.%Y", encoding="utf-8-sig")
            self.statusBar().showMessage(f"Backplan gespeichert: {path}", 8000)

    def on_report(self) -> None:
        if self.result is None or self.plan is None or self.costs() is None:
            self._error("Noch kein Ergebnis", "Bitte zuerst trainieren und eine Prognose berechnen.")
            return
        default = str(Path(self._documents()) / f"Broetchen-Report_{date.today():%Y-%m-%d}.pdf")
        path, _ = QFileDialog.getSaveFileName(self, "Report speichern", default, "PDF (*.pdf)")
        if not path:
            return
        inp = ReportInput(self.data, self.source_name, self.result, self.plan, self.strategies,
                          self.effects, summary_text(self.data, self.effects), self.costs())
        try:
            export_report(path, inp)
        except PermissionError:
            self._error("Speichern nicht möglich", "Die Datei ist vermutlich noch in einem PDF-Programm geöffnet.")
            return
        self.statusBar().showMessage(f"Report gespeichert: {path}", 8000)

    def on_about(self) -> None:
        QMessageBox.about(self, "Über", f"<b>Brötchen-Prognose</b> {__version__}<br>{__author__}<br><br>"
                          "Absatzprognose und Backmengenplanung je Filiale.")
