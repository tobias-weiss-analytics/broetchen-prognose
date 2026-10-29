"""Was treibt den Absatz? Effekte aus einer Regression mit Kontrollvariablen.

Geschätzt wird der Logarithmus des Absatzes, damit jeder Effekt als prozentuale
Veränderung lesbar ist. Alle Einflüsse werden gemeinsam geschätzt: Der
Aktionseffekt etwa ist um Wochentag, Filiale, Wetter, Preis und Saison bereinigt.
Das Wochenmuster jeder Filiale geht als eigene Kontrollvariable ein.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import holiday_names

WEEKDAYS = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]


def _design(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    holiday = holiday_names(df["date"]).ne("").to_numpy()
    pre = holiday_names(df["date"] + pd.Timedelta(days=1)).ne("").to_numpy() & ~holiday
    X = pd.DataFrame(index=df.index)
    X["Aktion"] = df["promo"]
    X["Lokales Ereignis"] = df["event"]
    X["Vortag eines Feiertags"] = pre.astype(float)
    stores = sorted(df["store"].unique())
    for store in stores:
        X[f"Feiertag – {store}"] = (holiday & (df["store"] == store)).astype(float)
    X["+5 mm Regen"] = df["rain_mm"].clip(upper=15) / 5
    X["+10 °C Temperatur"] = df["temp_c"] / 10
    X["+10 % Preis"] = np.log(df["price"]) / np.log(1.1)
    X["Trend je Jahr"] = (df["date"] - df["date"].min()).dt.days / 365.25
    shown = list(X.columns)
    # Kontrollvariablen (nicht angezeigt): Wochenmuster je Filiale und Saison
    store_day = df["store"] + "|" + df["date"].dt.weekday.astype(str)
    X = X.join(pd.get_dummies(store_day, dtype=float))
    doy = df["date"].dt.dayofyear
    for k in (1, 2):
        X[f"sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        X[f"cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    return X, shown


def driver_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Effekt jedes Treibers in Prozent (mit 95-%-Intervall) und in Brötchen am Durchschnittstag."""
    X, shown = _design(df)
    A, y = X.to_numpy(float), np.log(df["sales"].clip(lower=1).to_numpy(float))
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    sigma2 = resid @ resid / (len(y) - np.linalg.matrix_rank(A))
    se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(A.T @ A)))
    idx = [X.columns.get_loc(c) for c in shown]
    b, s = coef[idx], se[idx]
    pct = (np.exp(b) - 1) * 100
    out = pd.DataFrame({
        "Effekt %": pct,
        "von %": (np.exp(b - 1.96 * s) - 1) * 100,
        "bis %": (np.exp(b + 1.96 * s) - 1) * 100,
        "Brötchen": pct / 100 * df["sales"].mean(),
    }, index=shown)
    return out


def weekday_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Ø Absatz je Wochentag und Filiale (ohne Feiertage)."""
    normal = df[holiday_names(df["date"]).eq("").to_numpy()]
    table = normal.groupby([normal["date"].dt.weekday, "store"])["sales"].mean().unstack()
    table.index = [WEEKDAYS[i] for i in table.index]
    return table


def summary_text(df: pd.DataFrame, effects: pd.DataFrame) -> list[str]:
    avg = df["sales"].mean()
    significant = effects[(effects["von %"] > 0) | (effects["bis %"] < 0)]
    weak = effects.index.difference(significant.index)
    lines = [f"Ø Absatz: {avg:,.0f} Brötchen pro Tag und Filiale.".replace(",", ".")]
    order = significant["Effekt %"].abs().sort_values(ascending=False).index
    for name in order:
        e = significant.loc[name]
        sign = "+" if e["Effekt %"] > 0 else "−"
        lines.append(f"{name}: {sign}{abs(e['Effekt %']):.0f} % (≈ {sign}{abs(e['Brötchen']):.0f} Brötchen)")
    if len(weak):
        lines.append("Kein belastbarer Effekt: " + ", ".join(sorted(weak)) + ".")
    lines.append("Hinweis: Effekte sind statistische Zusammenhänge, keine bewiesenen Ursachen.")
    return lines
