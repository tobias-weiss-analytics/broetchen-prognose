import pandas as pd

from broetchen.model import add_features, backtest, default_assumptions, error_metrics, forecast


def test_features_use_only_the_past(sales):
    feats = add_features(sales)
    s1 = feats[feats["store"] == "Innenstadt"].reset_index(drop=True)
    assert s1.loc[20, "lag_7"] == s1.loc[13, "sales"]
    assert s1.loc[40, "mean_7"] == s1.loc[33:39, "sales"].mean()


def test_backtest_is_out_of_sample(sales):
    bt = backtest(sales, "ridge", weeks=2)
    assert len(bt) == 2 * 7 * 3
    assert bt["date"].min() == sales["date"].max() - pd.Timedelta(days=13)


def test_model_beats_naive_rule(trained):
    comp = trained.comparison
    assert comp.loc[trained.name, "MAPE"] < comp.loc["Vorwoche (gleicher Wochentag)", "MAPE"]
    assert comp.loc[trained.name, "MAPE"] < 10


def test_forecast_shape_and_promo_effect(sales, trained):
    a = default_assumptions(sales, 7)
    base = forecast(sales, trained, a, ["Innenstadt"])
    promo = forecast(sales, trained, a.assign(promo=1), ["Innenstadt"])
    assert len(base) == 7
    assert base["date"].min() == sales["date"].max() + pd.Timedelta(days=1)
    assert (base["low"] <= base["forecast"]).all() and (base["forecast"] <= base["high"]).all()
    assert (promo["forecast"] > base["forecast"]).all()


def test_error_metrics():
    m = error_metrics(pd.DataFrame({"actual": [100, 200], "forecast": [110, 180]}))
    assert m["MAE"] == 15 and m["Bias"] == -5 and round(m["MAPE"], 6) == 10


def test_holiday_features():
    dates = pd.to_datetime(["2026-10-31", "2026-11-01", "2026-11-02"])  # Allerheiligen am 1.11.
    df = pd.DataFrame({"date": dates, "store": "A", "sales": 1.0})
    f = add_features(df)
    assert f["holiday"].tolist() == [0, 1, 0]
    assert f["pre_holiday"].tolist() == [1, 0, 0]
    assert f["day_type"].tolist()[1] == "Feiertag"
