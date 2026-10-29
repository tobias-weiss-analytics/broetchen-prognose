"""Erzeugt die Demo-Daten in data/broetchen_bedarf.csv.

Die Daten sind simuliert, folgen aber einem realistischen Muster für drei
Filialen einer Bäckerei in einer bayerischen Kleinstadt. Weil die eingebauten
Effekte bekannt sind (siehe EFFECTS), lässt sich prüfen, ob die App sie
wiederfindet – das ist der Nachweis, dass Prognose und Treiberanalyse funktionieren.

    uv run python scripts/generate_demo_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from broetchen.data import holiday_names  # noqa: E402

START, END = "2023-01-01", "2026-10-28"
SEED = 7

# Eingebaute Effekte (multiplikativ auf die Nachfrage), für alle Filialen gleich.
EFFECTS = {
    "Aktion": 0.12,                 # +12 %
    "Vortag eines Feiertags": 0.15,  # +15 %
    "Regen je mm": -0.012,          # −1,2 % je mm Niederschlag
    "Preiselastizität": -0.4,       # +1 % Preis → −0,4 % Absatz
    "Trend je Jahr": 0.02,          # +2 % pro Jahr (Neubaugebiet, Zuzug)
}

# Filialprofile: Grundabsatz, Wochentagsmuster Mo–So, Faktor an Feiertagen,
# Effekt lokaler Ereignisse (Markt, Dult, Stadtfest) und Rückgang in den Sommerferien.
STORES = {
    "Innenstadt": dict(base=640, weekday=[0.95, 0.93, 1.02, 0.98, 1.08, 1.32, 0.72], holiday=0.70, event=0.25, summer=-0.06),
    "Bahnhof":    dict(base=520, weekday=[1.12, 1.10, 1.10, 1.10, 1.14, 0.78, 0.62], holiday=0.60, event=0.10, summer=-0.15),
    "Wohngebiet": dict(base=470, weekday=[0.88, 0.86, 0.88, 0.90, 1.04, 1.42, 1.48], holiday=1.40, event=0.05, summer=-0.10),
}

# Brötchenpreis mit Preiserhöhungen
PRICES = [("2023-01-01", 0.35), ("2024-03-01", 0.39), ("2025-03-01", 0.42), ("2026-02-01", 0.45)]


def weather(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Wetter wie in Niederbayern: Jahresgang der Temperatur, wechselhafter Regen."""
    doy = dates.dayofyear.to_numpy()
    seasonal = 9.0 + 9.8 * np.sin(2 * np.pi * (doy - 105) / 365.25)
    anomaly = np.zeros(len(dates))
    for i in range(1, len(dates)):
        anomaly[i] = 0.75 * anomaly[i - 1] + rng.normal(0, 2.8)
    summer = np.sin(2 * np.pi * (doy - 105) / 365.25) > 0
    rainy = rng.random(len(dates)) < np.where(summer, 0.50, 0.42)
    rain = np.where(rainy, np.minimum(rng.gamma(0.8, 5.0, len(dates)), 40), 0.0)
    return pd.DataFrame({"date": dates, "temp_c": (seasonal + anomaly).round(1), "rain_mm": rain.round(1)})


def price_on(dates: pd.DatetimeIndex) -> np.ndarray:
    steps = pd.Series([p for _, p in PRICES], index=pd.to_datetime([d for d, _ in PRICES]))
    return steps.reindex(dates, method="ffill").to_numpy()


def local_events(dates: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Maidult und Herbstdult (je 9 Tage) plus einzelne Markt- und Festtage."""
    event = np.zeros(len(dates), dtype=int)
    for year in sorted(set(dates.year)):
        for start in (f"{year}-05-02", f"{year}-09-22"):
            event |= ((dates >= start) & (dates < pd.Timestamp(start) + pd.Timedelta(days=9))).astype(int)
    event |= (rng.random(len(dates)) < 0.015).astype(int)
    return event


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(START, END, freq="D")
    wx = weather(dates, rng)
    price = price_on(dates)
    event = local_events(dates, rng)
    holiday = holiday_names(dates).ne("").to_numpy()
    pre_holiday = np.roll(holiday, -1) & ~holiday
    years = (dates - dates[0]).days / 365.25
    doy = dates.dayofyear.to_numpy()
    summer_holidays = (doy >= 213) & (doy <= 256)                  # ca. August bis Mitte September
    christmas = (dates.month == 12) & (dates.day >= 20) & (dates.day <= 24)

    frames = []
    for name, p in STORES.items():
        # Aktionen vor allem an schwachen Tagen (Mo–Mi), ca. 8 % der Tage
        promo = (rng.random(len(dates)) < np.where(dates.weekday <= 2, 0.14, 0.04)).astype(int)
        demand = (
            p["base"]
            * np.where(holiday, p["holiday"], np.array(p["weekday"])[dates.weekday])   # Feiertag statt Wochentag
            * (1 + EFFECTS["Vortag eines Feiertags"] * pre_holiday)
            * (1 + EFFECTS["Aktion"] * promo)
            * (1 + p["event"] * event)
            * (1 + EFFECTS["Regen je mm"] * np.minimum(wx["rain_mm"].to_numpy(), 15))
            * (1 - 0.02 * np.clip(wx["temp_c"].to_numpy() - 26, 0, None))   # Hitze: −2 % je °C über 26 °C
            * (price / PRICES[0][1]) ** EFFECTS["Preiselastizität"]
            * (1 + EFFECTS["Trend je Jahr"]) ** years
            * (1 + p["summer"] * summer_holidays)
            * (1 + 0.15 * christmas)
            * np.exp(rng.normal(0, 0.06, len(dates)))
        )
        frames.append(pd.DataFrame({
            "date": dates, "store": name, "sales": rng.poisson(demand), "price": price,
            "promo": promo, "temp_c": wx["temp_c"], "rain_mm": wx["rain_mm"], "event": event,
        }))
    return pd.concat(frames).sort_values(["date", "store"], ignore_index=True)


if __name__ == "__main__":
    out = ROOT / "data" / "broetchen_bedarf.csv"
    df = generate()
    df.to_csv(out, index=False, date_format="%Y-%m-%d")
    print(f"{len(df)} Zeilen geschrieben: {out}")
    print(df.groupby("store")["sales"].describe().round(0))
