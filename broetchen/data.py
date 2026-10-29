"""Einlesen, Prüfen und Aufbereiten der Verkaufsdaten."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

SAMPLE_DATA = Path(__file__).resolve().parent.parent / "data" / "broetchen_bedarf.csv"

# Interne Spaltennamen. Alternative Namen aus älteren Exporten werden umbenannt.
COLUMNS = ["date", "store", "sales", "price", "promo", "temp_c", "rain_mm", "event"]
ALIASES = {
    "store_id": "store", "filiale": "store",
    "sales_units": "sales", "y": "sales", "absatz": "sales",
    "event_local": "event",
    "datum": "date", "preis": "price", "aktion": "promo",
}
OPTIONAL_DEFAULTS = {"promo": 0, "rain_mm": 0.0, "event": 0}


# Feiertage: Bayern einschließlich Mariä Himmelfahrt (gilt in überwiegend katholischen
# Gemeinden wie Passau). Für ein anderes Bundesland das Kürzel ändern, z. B. "BW" oder "NW".
REGION = "BY"


@lru_cache(maxsize=8)
def _calendar(first_year: int, last_year: int) -> holidays.HolidayBase:
    return holidays.Germany(subdiv=REGION, years=range(first_year, last_year + 1), categories=("public", "catholic"))


def holiday_names(dates: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """Name des gesetzlichen Feiertags je Datum, sonst leerer Text (Index wie ``dates``)."""
    dates = pd.Series(pd.to_datetime(dates)) if not isinstance(dates, pd.Series) else pd.to_datetime(dates)
    calendar = _calendar(int(dates.dt.year.min()), int(dates.dt.year.max()) + 1)
    return dates.map(lambda d: calendar.get(d.date(), "")).astype(str)


class DataError(ValueError):
    """Die Datei hat nicht das erwartete Format."""


def load_sales(path: str | Path = SAMPLE_DATA) -> pd.DataFrame:
    """CSV einlesen und auf das interne Schema bringen (eine Zeile je Tag und Filiale)."""
    try:
        raw = pd.read_csv(path, sep=None, engine="python")
    except Exception as exc:  # noqa: BLE001 - Meldung wird in der Oberfläche angezeigt
        raise DataError(f"Datei konnte nicht gelesen werden: {exc}") from exc
    return clean_sales(raw)


def clean_sales(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=lambda c: ALIASES.get(str(c).strip().lower(), str(c).strip().lower()))
    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise DataError(
            "Es fehlen Spalten: " + ", ".join(missing)
            + ". Erwartet werden mindestens date, store, sales, price, temp_c."
        )
    df = df[COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["store"] = df["store"].astype(str).str.strip()
    for col in ["sales", "price", "temp_c", "rain_mm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["promo", "event"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(0, 1).astype(int)

    bad = df["date"].isna() | df["sales"].isna()
    if bad.all():
        raise DataError("Keine gültigen Zeilen gefunden (Datum oder Absatz fehlen überall).")
    df = df[~bad]
    if df.duplicated(["date", "store"]).any():
        raise DataError("Datum und Filiale kommen mehrfach vor. Bitte eine Zeile je Tag und Filiale.")

    # Lückenlose Tagesreihe je Filiale; fehlende Werte werden fortgeschrieben.
    frames = []
    for store, g in df.groupby("store"):
        g = g.set_index("date").asfreq("D")
        g["store"] = store
        g["sales"] = g["sales"].interpolate(limit_direction="both")
        g[["price", "temp_c", "rain_mm"]] = g[["price", "temp_c", "rain_mm"]].ffill().bfill()
        g[["promo", "event"]] = g[["promo", "event"]].fillna(0).astype(int)
        frames.append(g.reset_index())
    out = pd.concat(frames, ignore_index=True).sort_values(["store", "date"], ignore_index=True)
    if len(out) < 60 * out["store"].nunique():
        raise DataError("Zu wenig Historie: mindestens 60 Tage je Filiale werden benötigt.")
    return out


def seasonal_weather(history: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Typisches Wetter für zukünftige Tage: Mittel der Historie im selben Kalendermonat."""
    by_month = history.groupby(history["date"].dt.month)[["temp_c", "rain_mm"]].mean()
    overall = history[["temp_c", "rain_mm"]].mean()
    rows = [by_month.loc[d.month] if d.month in by_month.index else overall for d in dates]
    return pd.DataFrame(rows, index=dates).round(1).rename_axis("date").reset_index()


def weekday_name(dates: pd.Series) -> pd.Series:
    names = np.array(["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"])
    return pd.Series(names[pd.to_datetime(dates).dt.weekday], index=dates.index)
