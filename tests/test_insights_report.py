from broetchen.insights import driver_effects, summary_text
from broetchen.model import default_assumptions, forecast, naive_forecast
from broetchen.planning import Costs, build_plan, compare_strategies
from broetchen.report import ReportInput, export_report


def test_driver_effects(sales):
    eff = driver_effects(sales)
    assert eff.loc["Aktion", "Effekt %"] > 0
    assert eff.loc["Feiertag – Wohngebiet", "Effekt %"] > 0 > eff.loc["Feiertag – Bahnhof", "Effekt %"]
    assert any("Aktion" in line for line in summary_text(sales, eff))


def test_pdf_report(tmp_path, sales, trained):
    costs = Costs(0.35, 0.12)
    plan = build_plan(forecast(sales, trained, default_assumptions(sales, 7)), trained.ratios, costs)
    strategies = compare_strategies(trained.backtest, naive_forecast(sales, trained.backtest), trained.ratios, costs)
    eff = driver_effects(sales)
    out = export_report(tmp_path / "r.pdf",
                        ReportInput(sales, "test.csv", trained, plan, strategies, eff, summary_text(sales, eff), costs))
    assert out.read_bytes()[:4] == b"%PDF"
    assert out.stat().st_size > 50_000
