"""Absatzprognose: Merkmale, Modelle, zeitlicher Backtest und rekursive Vorhersage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import holiday_names, seasonal_weather

HISTORY_DAYS = 28          # so viel Vorlauf brauchen die Lag-Merkmale
BACKTEST_WEEKS = 8         # Länge des Testzeitraums am Ende der Daten
BACKTEST_HORIZON = 7       # jede Woche wird 7 Tage im Voraus prognostiziert

EXOGENOUS = ["price", "promo", "event", "temp_c", "rain_mm"]
LAGS = ["lag_7", "lag_14", "mean_7", "mean_28"]
CALENDAR = ["month", "doy_sin", "doy_cos", "holiday", "pre_holiday"]
# Tagestyp = Wochentag oder „Feiertag“; zusammen mit der Filiale, weil jede Filiale
# ihr eigenes Wochenmuster hat (Bahnhof: werktags stark, Wohngebiet: am Wochenende).
CATEGORICAL = ["store", "day_type", "store_day"]
NUMERIC = EXOGENOUS + LAGS + CALENDAR

MODELS: dict[str, str] = {
    "auto": "Automatisch (bestes im Backtest)",
    "hgb": "Gradient Boosting",
    "rf": "Random Forest",
    "ridge": "Lineare Regression (Ridge)",
}


def make_estimator(key: str) -> Pipeline:
    if key == "ridge":
        pre = ColumnTransformer([
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ])
        return Pipeline([("pre", pre), ("model", Ridge(alpha=1.0))])
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    if key == "hgb":
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=20, random_state=42)
    elif key == "rf":
        model = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, max_features=0.5, n_jobs=1, random_state=42)
    else:
        raise ValueError(f"Unbekanntes Modell: {key}")
    return Pipeline([("pre", pre), ("model", model)])


# --------------------------------------------------------------------------- #
# Merkmale
# --------------------------------------------------------------------------- #
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Kalender- und Verlaufsmerkmale. Verlaufswerte nutzen nur Tage *vor* dem Zieltag."""
    out = df.sort_values(["store", "date"]).copy()
    by_store = out.groupby("store")["sales"]
    out["lag_7"] = by_store.shift(7)
    out["lag_14"] = by_store.shift(14)
    out["mean_7"] = by_store.transform(lambda s: s.shift(1).rolling(7).mean())
    out["mean_28"] = by_store.transform(lambda s: s.shift(1).rolling(28).mean())
    holiday = holiday_names(out["date"]).ne("")
    next_day_holiday = holiday_names(out["date"] + pd.Timedelta(days=1)).ne("")
    out["holiday"] = holiday.astype(int)
    out["pre_holiday"] = (next_day_holiday & ~holiday).astype(int)
    out["day_type"] = np.where(holiday, "Feiertag", out["date"].dt.weekday.astype(str))
    out["store_day"] = out["store"] + "|" + out["day_type"]
    out["month"] = out["date"].dt.month
    doy = out["date"].dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def training_rows(df: pd.DataFrame) -> pd.DataFrame:
    return add_features(df).dropna(subset=LAGS)


def recursive_forecast(model: Pipeline, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    """Prognose Tag für Tag; jede Prognose dient als Verlaufswert für die folgenden Tage.

    ``history`` enthält Ist-Absätze, ``future`` die Annahmen (Preis, Aktion, Wetter …)
    für die zu prognostizierenden Tage, ohne ``sales``.
    """
    window = history.groupby("store").tail(HISTORY_DAYS + 1)
    frame = pd.concat([window, future.assign(sales=np.nan)], ignore_index=True)
    frame = frame.sort_values(["store", "date"], ignore_index=True)
    for day in sorted(future["date"].unique()):
        feats = add_features(frame)
        rows = feats["date"] == day
        pred = model.predict(feats.loc[rows, NUMERIC + CATEGORICAL])
        frame.loc[feats.index[rows], "sales"] = np.maximum(pred, 0)
    result = frame[frame["date"].isin(future["date"])][["date", "store", "sales"]]
    return result.rename(columns={"sales": "forecast"}).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def backtest(df: pd.DataFrame, key: str, weeks: int = BACKTEST_WEEKS, horizon: int = BACKTEST_HORIZON) -> pd.DataFrame:
    """Rollierender Backtest über die letzten ``weeks`` Wochen.

    Das Modell wird nur auf Daten *vor* dem Testzeitraum trainiert. Ab jedem
    Wochenbeginn wird ``horizon`` Tage im Voraus prognostiziert, wie es eine
    Filiale bei der Wochenplanung tun würde. Für das Wetter werden die
    tatsächlichen Werte verwendet (entspricht einer perfekten Wettervorhersage).
    """
    test_start = df["date"].max() - pd.Timedelta(days=weeks * 7 - 1)
    train = df[df["date"] < test_start]
    if key == "naive":
        out = df[df["date"] >= test_start][["date", "store", "sales"]].reset_index(drop=True)
        out["forecast"] = naive_forecast(df, out)
        return out.rename(columns={"sales": "actual"})

    model = make_estimator(key).fit(*_xy(training_rows(train)))
    parts = []
    for origin in pd.date_range(test_start, df["date"].max(), freq=f"{horizon}D"):
        days = pd.date_range(origin, min(origin + pd.Timedelta(days=horizon - 1), df["date"].max()))
        history = df[df["date"] < origin]
        future = df[df["date"].isin(days)].drop(columns="sales")
        fc = recursive_forecast(model, history, future)
        parts.append(fc.merge(df[["date", "store", "sales"]], on=["date", "store"]))
    out = pd.concat(parts, ignore_index=True).rename(columns={"sales": "actual"})
    return out[["date", "store", "actual", "forecast"]]


def error_metrics(bt: pd.DataFrame) -> dict[str, float]:
    err = bt["forecast"] - bt["actual"]
    return {
        "MAE": float(err.abs().mean()),
        "MAPE": float((err.abs() / bt["actual"].clip(lower=1)).mean() * 100),
        "Bias": float(err.mean()),
    }


def _xy(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return rows[NUMERIC + CATEGORICAL], rows["sales"]


# --------------------------------------------------------------------------- #
# Training und Prognose
# --------------------------------------------------------------------------- #
@dataclass
class TrainResult:
    key: str
    model: Pipeline
    comparison: pd.DataFrame            # Fehlermaße aller Modelle und der Vorwochen-Regel
    backtest: pd.DataFrame              # Backtest des gewählten Modells
    ratios: np.ndarray = field(repr=False)  # Ist / Prognose im Backtest, für Puffer und Intervalle

    @property
    def name(self) -> str:
        return MODELS[self.key]


def train(df: pd.DataFrame, key: str = "auto", progress: Callable[[str], None] | None = None) -> TrainResult:
    progress = progress or (lambda _msg: None)
    candidates = ["hgb", "rf", "ridge"] if key == "auto" else [key]
    rows, backtests = {}, {}
    for cand in candidates + ["naive"]:
        label = "Vorwoche (gleicher Wochentag)" if cand == "naive" else MODELS[cand]
        progress(f"Backtest: {label} …")
        backtests[cand] = backtest(df, cand)
        rows[label] = error_metrics(backtests[cand])
    comparison = pd.DataFrame(rows).T.sort_values("MAE")
    best = min(candidates, key=lambda c: rows[MODELS[c]]["MAE"])

    progress(f"Training auf allen Daten: {MODELS[best]} …")
    model = make_estimator(best).fit(*_xy(training_rows(df)))
    bt = backtests[best]
    ratios = (bt["actual"] / bt["forecast"].clip(lower=1)).to_numpy()
    return TrainResult(best, model, comparison, bt, ratios)


def default_assumptions(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Annahmen für die kommenden Tage: keine Aktion/kein Ereignis, saisonübliches Wetter."""
    start = df["date"].max() + pd.Timedelta(days=1)
    dates = pd.date_range(start, periods=horizon, freq="D")
    out = seasonal_weather(df, dates)
    out["promo"] = 0
    out["event"] = 0
    return out[["date", "promo", "event", "temp_c", "rain_mm"]]


def forecast(df: pd.DataFrame, result: TrainResult, assumptions: pd.DataFrame,
             stores: list[str] | None = None) -> pd.DataFrame:
    """Prognose für die Tage in ``assumptions`` (gelten für alle Filialen; Preis = zuletzt gültiger)."""
    stores = stores or sorted(df["store"].unique())
    last_price = df.sort_values("date").groupby("store")["price"].last()
    future = pd.concat([assumptions.assign(store=s, price=last_price[s]) for s in stores], ignore_index=True)
    history = df[df["store"].isin(stores)]
    fc = recursive_forecast(result.model, history, future)
    lo, hi = np.quantile(result.ratios, [0.1, 0.9])
    fc["low"] = fc["forecast"] * lo
    fc["high"] = fc["forecast"] * hi
    return fc


def naive_forecast(df: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    """Regel „wie Vorwoche“: Absatz derselben Filiale am selben Wochentag eine Woche zuvor."""
    lag = df.set_index(["store", "date"])["sales"]
    keys = list(zip(rows["store"], rows["date"] - pd.Timedelta(days=7), strict=True))
    return pd.Series(lag.reindex(keys).to_numpy(), index=rows.index)
