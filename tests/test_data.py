import pandas as pd
import pytest

from broetchen.data import DataError, clean_sales, seasonal_weather


def test_sample_data(sales):
    assert set(sales.columns) == {"date", "store", "sales", "price", "promo", "temp_c", "rain_mm", "event"}
    assert sales["store"].nunique() == 3
    assert not sales.isna().any().any()


def test_aliases_and_gap_filling():
    dates = pd.date_range("2024-01-01", periods=70).delete(10)  # ein fehlender Tag
    raw = pd.DataFrame({"Datum": dates, "store_id": "A", "sales_units": range(69), "Preis": 0.4, "temp_c": 5.0})
    df = clean_sales(raw)
    assert len(df) == 70
    assert df.loc[10, "sales"] == pytest.approx(9.5)
    assert (df["promo"] == 0).all()


def test_missing_columns_raise():
    with pytest.raises(DataError, match="fehlen"):
        clean_sales(pd.DataFrame({"date": ["2024-01-01"], "store": ["A"]}))


def test_duplicates_raise():
    raw = pd.DataFrame({"date": ["2024-01-01"] * 2, "store": "A", "sales": 1, "price": 1, "temp_c": 1})
    with pytest.raises(DataError, match="mehrfach"):
        clean_sales(raw)


def test_seasonal_weather(sales):
    w = seasonal_weather(sales, pd.date_range("2026-01-01", periods=3))
    january = sales[sales["date"].dt.month == 1]["temp_c"].mean()
    assert w["temp_c"].iloc[0] == pytest.approx(january, abs=0.05)
