import numpy as np
import pandas as pd
import pytest

from broetchen.planning import Costs, compare_strategies, evaluate, recommend


def test_costs_validation():
    assert Costs(0.40, 0.10).critical_ratio == pytest.approx(0.75)
    with pytest.raises(ValueError):
        Costs(0.10, 0.20)


def test_evaluate():
    r = evaluate(pd.Series([10, 10]), pd.Series([12, 8]), Costs(1.0, 0.4, 0.1))
    # Tag 1: 10 verkauft, 2 übrig; Tag 2: 8 verkauft, 2 fehlen
    assert r["Gewinn (€)"] == pytest.approx(18 * 1.0 - 20 * 0.4 + 2 * 0.1)
    assert r["Abschriften (Stk.)"] == 2 and r["Fehlmenge (Stk.)"] == 2


def test_higher_margin_means_bigger_buffer():
    ratios = np.linspace(0.8, 1.2, 101)
    fc = pd.Series([100.0])
    assert recommend(fc, ratios, Costs(1.0, 0.1))[0] > recommend(fc, ratios, Costs(1.0, 0.8))[0]


def test_recommended_strategy_is_profitable(sales, trained):
    from broetchen.model import naive_forecast
    bt = trained.backtest
    table = compare_strategies(bt, naive_forecast(sales, bt), trained.ratios, Costs(0.35, 0.12))
    profit = table["Gewinn (€)"]
    assert profit["Prognose + Puffer (empfohlen)"] > profit["Wie Vorwoche backen"]
    assert profit["Perfekte Voraussicht"] == profit.max()
