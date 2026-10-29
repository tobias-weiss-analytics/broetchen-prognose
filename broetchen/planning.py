"""Backmengenplanung nach dem Zeitungsjungen-Modell (Newsvendor).

Jedes nicht gebackene, aber nachgefragte Brötchen kostet den entgangenen
Deckungsbeitrag (Preis − Herstellkosten). Jedes übrig gebliebene Brötchen kostet
die Herstellkosten abzüglich eines Restwerts (z. B. Verkauf am Folgetag, Tafel).
Die gewinnoptimale Backmenge ist das Quantil der Nachfrage bei

    kritisches Verhältnis = Unterdeckungskosten / (Unterdeckungskosten + Überdeckungskosten)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Costs:
    price: float          # Verkaufspreis je Brötchen (€)
    unit_cost: float      # Herstellkosten je Brötchen (€)
    salvage: float = 0.0  # Restwert eines übrig gebliebenen Brötchens (€)

    def __post_init__(self):
        if not 0 <= self.salvage < self.unit_cost < self.price:
            raise ValueError("Es muss gelten: 0 ≤ Restwert < Herstellkosten < Verkaufspreis.")

    @property
    def underage(self) -> float:
        return self.price - self.unit_cost

    @property
    def overage(self) -> float:
        return self.unit_cost - self.salvage

    @property
    def critical_ratio(self) -> float:
        return self.underage / (self.underage + self.overage)


def safety_factor(ratios: np.ndarray, costs: Costs) -> float:
    """Faktor auf die Prognose: Quantil von Ist/Prognose zum kritischen Verhältnis."""
    return float(np.quantile(ratios, costs.critical_ratio))


def recommend(forecast: pd.Series, ratios: np.ndarray, costs: Costs) -> pd.Series:
    """Empfohlene Backmenge (aufgerundet auf ganze Brötchen)."""
    return np.ceil(forecast * safety_factor(ratios, costs)).astype(int)


def evaluate(actual: pd.Series, baked: pd.Series, costs: Costs) -> dict[str, float]:
    """Wirtschaftliches Ergebnis, wenn ``baked`` gebacken und höchstens ``actual`` verkauft wird."""
    baked = np.round(np.asarray(baked, dtype=float))
    actual = np.asarray(actual, dtype=float)
    sold = np.minimum(baked, actual)
    leftover = baked - sold
    short = actual - sold
    profit = sold * costs.price - baked * costs.unit_cost + leftover * costs.salvage
    return {
        "Gewinn (€)": float(profit.sum()),
        "Abschriften (Stk.)": float(leftover.sum()),
        "Fehlmenge (Stk.)": float(short.sum()),
        "Ausverkauft (Tage %)": float((short > 0).mean() * 100),
    }


def compare_strategies(bt: pd.DataFrame, naive: pd.Series, ratios: np.ndarray, costs: Costs) -> pd.DataFrame:
    """Vergleich der Backstrategien im Backtest-Zeitraum."""
    strategies = {
        "Wie Vorwoche backen": naive,
        "Prognose backen": bt["forecast"],
        "Prognose + Puffer (empfohlen)": recommend(bt["forecast"], ratios, costs),
        "Perfekte Voraussicht": bt["actual"],
    }
    return pd.DataFrame({name: evaluate(bt["actual"], baked, costs) for name, baked in strategies.items()}).T


def build_plan(fc: pd.DataFrame, ratios: np.ndarray, costs: Costs) -> pd.DataFrame:
    """Prognose um die empfohlene Backmenge ergänzen."""
    plan = fc.copy()
    plan["recommended"] = recommend(plan["forecast"], ratios, costs)
    return plan
