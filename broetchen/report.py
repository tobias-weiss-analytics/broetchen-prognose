"""PDF-Report mit Prognose, Backmengenempfehlung, Modellgüte und Treibern."""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4, landscape  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

from . import __version__, charts  # noqa: E402
from .data import holiday_names, weekday_name  # noqa: E402
from .insights import weekday_profile  # noqa: E402
from .model import TrainResult  # noqa: E402
from .planning import Costs  # noqa: E402

HEADER = colors.HexColor("#2f3b52")


@dataclass
class ReportInput:
    data: pd.DataFrame
    source_name: str
    result: TrainResult
    plan: pd.DataFrame            # date, store, forecast, low, high, recommended
    strategies: pd.DataFrame
    effects: pd.DataFrame
    insight_lines: list[str]
    costs: Costs


def _figure(draw, width=25, height=8.5) -> Image:
    fig, ax = plt.subplots(figsize=(width / 2.54, height / 2.54), dpi=150)
    draw(ax)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=width * cm, height=height * cm)


def _table(df: pd.DataFrame, col_widths=None, font_size=8.5) -> Table:
    data = [list(df.columns)] + df.astype(str).values.tolist()
    t = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", font_size),
        ("FONT", (0, 1), (-1, -1), "Helvetica", font_size),
        ("BACKGROUND", (0, 0), (-1, 0), HEADER),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f4f7")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d0d5dd")),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def _num(x: float, digits: int = 0) -> str:
    return f"{x:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def plan_table(plan: pd.DataFrame) -> pd.DataFrame:
    """Je Datum eine Zeile, je Filiale Prognose und empfohlene Backmenge."""
    wide = plan.pivot(index="date", columns="store", values=["forecast", "recommended"])
    names = holiday_names(pd.Series(wide.index))
    out = pd.DataFrame({"Datum": [f"{weekday_name(pd.Series([d])).iloc[0]} {d:%d.%m.%Y}" for d in wide.index],
                        "Feiertag": names.to_numpy()})
    for store in sorted(plan["store"].unique()):
        out[f"{store} Prognose"] = [_num(v) for v in wide[("forecast", store)]]
        out[f"{store} backen"] = [_num(v) for v in wide[("recommended", store)]]
    return out


def export_report(path: str | Path, inp: ReportInput) -> Path:
    path = Path(path)
    doc = SimpleDocTemplate(str(path), pagesize=landscape(A4), leftMargin=1.4 * cm, rightMargin=1.4 * cm,
                            topMargin=1.2 * cm, bottomMargin=1.2 * cm, title="Brötchen-Prognose – Report",
                            author="Brötchen-Prognose")
    base = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=base["Title"], alignment=0, fontSize=20, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=base["Heading2"], fontSize=13, spaceBefore=8, spaceAfter=4)
    body = ParagraphStyle("body", parent=base["Normal"], fontSize=9.5, leading=13)
    meta = ParagraphStyle("meta", parent=body, textColor=colors.HexColor("#555555"))

    r, df, c = inp.result, inp.data, inp.costs
    model_err = r.comparison.loc[r.name]
    naive_err = r.comparison.loc["Vorwoche (gleicher Wochentag)"]
    start, end = inp.plan["date"].min(), inp.plan["date"].max()
    story = [
        Paragraph("Brötchen-Prognose", h1),
        Paragraph(
            f"Daten: {inp.source_name} ({df['date'].min():%d.%m.%Y} – {df['date'].max():%d.%m.%Y}, "
            f"{df['store'].nunique()} Filialen) &nbsp;·&nbsp; Modell: <b>{r.name}</b> &nbsp;·&nbsp; "
            f"erstellt {datetime.now():%d.%m.%Y %H:%M} mit Version {__version__}", meta),
        Spacer(1, 0.2 * cm),
        Paragraph(
            f"Im Backtest der letzten 8 Wochen lag die Prognose im Mittel um <b>{_num(model_err['MAPE'], 1)} %</b> "
            f"daneben (Regel „wie Vorwoche“: {_num(naive_err['MAPE'], 1)} %). Die empfohlene Backmenge enthält einen "
            f"Puffer für das kritische Verhältnis von {c.critical_ratio:.0%} "
            f"(Preis {_num(c.price, 2)} €, Herstellkosten {_num(c.unit_cost, 2)} €, Restwert {_num(c.salvage, 2)} €).",
            body),
        Paragraph(f"Prognose {start:%d.%m.} – {end:%d.%m.%Y}", h2),
        _figure(lambda ax: charts.forecast(ax, df, inp.plan, inp.plan), height=5.6),
        Paragraph("Backplan (Prognose und empfohlene Backmenge in Stück)", h2),
        _table(plan_table(inp.plan)),
        PageBreak(),
        Paragraph("Modellgüte im Backtest", h2),
        _table(r.comparison.round(1).map(_num_1).reset_index().rename(columns={"index": "Verfahren",
                                                                            "MAE": "MAE (Stk.)",
                                                                            "MAPE": "MAPE (%)",
                                                                            "Bias": "Bias (Stk.)"})),
        Spacer(1, 0.2 * cm),
        _figure(lambda ax: charts.backtest(ax, r.backtest), height=7),
        Paragraph("Backstrategien im Backtest-Zeitraum", h2),
        _table(inp.strategies.map(lambda v: _num(v)).reset_index().rename(columns={"index": "Strategie"})),
        PageBreak(),
        Paragraph("Was treibt den Absatz?", h2),
        Table([[_figure(lambda ax: charts.effects(ax, inp.effects), width=15, height=9),
                 _figure(lambda ax: charts.weekday_profile(ax, weekday_profile(inp.data)), width=10, height=9)]],
              hAlign="LEFT"),
        Spacer(1, 0.2 * cm),
        *[Paragraph("• " + line, body) for line in inp.insight_lines],
    ]
    doc.build(story)
    return path


def _num_1(x: float) -> str:
    return _num(x, 1)
