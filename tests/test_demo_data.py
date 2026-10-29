"""Nachweis: Die Treiberanalyse findet die in den Demo-Daten eingebauten Effekte wieder."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from generate_demo_data import EFFECTS, generate  # noqa: E402

from broetchen.insights import driver_effects  # noqa: E402

EXPECTED = {  # eingebauter Effekt in Prozent, umgerechnet auf die Einheit der Anzeige
    "Aktion": EFFECTS["Aktion"] * 100,
    "Vortag eines Feiertags": EFFECTS["Vortag eines Feiertags"] * 100,
    "+5 mm Regen": ((1 + EFFECTS["Regen je mm"]) ** 5 - 1) * 100,
    "+10 % Preis": (1.1 ** EFFECTS["Preiselastizität"] - 1) * 100,
    "Trend je Jahr": EFFECTS["Trend je Jahr"] * 100,
}


def test_demo_file_matches_generator(sales):
    regenerated = generate()
    assert len(regenerated) == len(sales)
    assert np.allclose(regenerated["sales"].to_numpy(), sales.sort_values(["date", "store"])["sales"].to_numpy())


@pytest.mark.parametrize("name", EXPECTED)
def test_built_in_effects_are_recovered(sales, name):
    eff = driver_effects(sales).loc[name]
    assert eff["von %"] - 1 <= EXPECTED[name] <= eff["bis %"] + 1, (name, EXPECTED[name], eff.round(1).to_dict())


def test_store_specific_holidays(sales):
    eff = driver_effects(sales)["Effekt %"]
    assert eff["Feiertag – Wohngebiet"] > 30
    assert eff["Feiertag – Bahnhof"] < -30
    assert pd.notna(eff["Feiertag – Innenstadt"])
