"""Diagramme, gemeinsam genutzt von Oberfläche und PDF-Report."""

from __future__ import annotations

import matplotlib.dates as mdates
import pandas as pd
from matplotlib.axes import Axes

PALETTE = ["#4f8ef7", "#f5a524", "#3fbf7f", "#e5484d", "#a78bfa", "#2dd4bf"]


def _date_axis(ax: Axes) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m."))
    ax.grid(alpha=0.25)
    for label in ax.get_xticklabels():
        label.set_rotation(0)


def history(ax: Axes, df: pd.DataFrame, stores: list[str], days: int = 180) -> None:
    recent = df[df["date"] > df["date"].max() - pd.Timedelta(days=days)]
    for i, store in enumerate(stores):
        g = recent[recent["store"] == store]
        ax.plot(g["date"], g["sales"].rolling(7, min_periods=1).mean(), color=PALETTE[i % 6], lw=2, label=f"{store} (7-Tage-Mittel)")
        ax.plot(g["date"], g["sales"], color=PALETTE[i % 6], lw=0.8, alpha=0.35)
    ax.set_title(f"Absatz der letzten {days} Tage")
    ax.set_ylabel("Brötchen pro Tag")
    ax.legend(frameon=False, fontsize=8)
    _date_axis(ax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))


def forecast(ax: Axes, df: pd.DataFrame, fc: pd.DataFrame, plan: pd.DataFrame | None = None, context_days: int = 28) -> None:
    for i, store in enumerate(sorted(fc["store"].unique())):
        color = PALETTE[i % 6]
        hist = df[(df["store"] == store) & (df["date"] > df["date"].max() - pd.Timedelta(days=context_days))]
        f = fc[fc["store"] == store]
        ax.plot(hist["date"], hist["sales"], color=color, lw=1.5, alpha=0.6)
        ax.plot(f["date"], f["forecast"], color=color, lw=2, marker="o", ms=3, label=f"{store} Prognose")
        ax.fill_between(f["date"], f["low"], f["high"], color=color, alpha=0.15)
        if plan is not None:
            p = plan[plan["store"] == store]
            ax.step(p["date"], p["recommended"], where="mid", color=color, lw=1, ls="--")
    ax.axvline(df["date"].max() + pd.Timedelta(hours=12), color="grey", lw=1, ls=":")
    ax.set_title("Prognose (Fläche: 80-%-Intervall, gestrichelt: empfohlene Backmenge)")
    ax.set_ylabel("Brötchen pro Tag")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    _date_axis(ax)


def backtest(ax: Axes, bt: pd.DataFrame) -> None:
    for i, store in enumerate(sorted(bt["store"].unique())):
        g = bt[bt["store"] == store].sort_values("date")
        ax.plot(g["date"], g["actual"], color=PALETTE[i % 6], lw=1.5, label=f"{store} Ist")
        ax.plot(g["date"], g["forecast"], color=PALETTE[i % 6], lw=1.5, ls="--", label=f"{store} Prognose")
    ax.set_title("Backtest: Ist vs. Prognose (7 Tage im Voraus)")
    ax.set_ylabel("Brötchen pro Tag")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    _date_axis(ax)


def effects(ax: Axes, eff: pd.DataFrame) -> None:
    eff = eff.sort_values("Effekt %")
    colors = [PALETTE[2] if v > 0 else PALETTE[3] for v in eff["Effekt %"]]
    err = [eff["Effekt %"] - eff["von %"], eff["bis %"] - eff["Effekt %"]]
    ax.barh(eff.index, eff["Effekt %"], xerr=err, color=colors, alpha=0.85, ecolor="grey", capsize=2)
    ax.axvline(0, color="grey", lw=1)
    ax.set_title("Treiber des Absatzes (in %, mit 95-%-Intervall)")
    ax.xaxis.set_major_formatter(lambda x, _pos: f"{x:+.0f} %")
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(axis="y", labelsize=8)


def weekday_profile(ax: Axes, profile: pd.DataFrame) -> None:
    for i, store in enumerate(profile.columns):
        ax.plot(profile.index, profile[store], color=PALETTE[i % 6], lw=2, marker="o", ms=4, label=store)
    ax.set_title("Wochenmuster (Ø ohne Feiertage)")
    ax.set_ylabel("Brötchen pro Tag")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
