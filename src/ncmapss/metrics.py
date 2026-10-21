"""Métricas de error y de incertidumbre para el RUL.

d = predicho − real. Un d > 0 es SOBREESTIMAR la vida útil (el caso peligroso: se planifica
una inspección para cuando el motor ya falló).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))


def nasa_score(y, p, a_under: float = 13.0, a_over: float = 10.0, reduce: str = "mean") -> float:
    """Score asimétrico de NASA (PHM08): exp(-d/13)-1 si d<0, exp(d/10)-1 si d>=0. Menor es mejor."""
    d = np.asarray(p, dtype=float) - np.asarray(y, dtype=float)
    s = np.where(d < 0, np.exp(-d / a_under) - 1, np.exp(d / a_over) - 1)
    return float(s.mean() if reduce == "mean" else s.sum())


def overestimation_rate(y, p) -> float:
    return float(np.mean(np.asarray(p) > np.asarray(y)))


def interval_metrics(y, lo, hi) -> dict:
    y, lo, hi = (np.asarray(v, dtype=float) for v in (y, lo, hi))
    return {"coverage": float(np.mean((y >= lo) & (y <= hi))),
            "width": float(np.mean(hi - lo)),
            "miss_over": float(np.mean(lo > y)),     # la cota inferior ya está por encima del real: sobreestimación
            "miss_under": float(np.mean(hi < y))}


def summary(y, p, lo=None, hi=None, nasa: dict | None = None) -> dict:
    nasa = nasa or {}
    out = {"rmse": rmse(y, p), "mae": mae(y, p), "nasa": nasa_score(y, p, **nasa),
           "over_point": overestimation_rate(y, p), "n": int(len(y))}
    if lo is not None:
        out.update(interval_metrics(y, lo, hi))
        out["nasa_lower"] = nasa_score(y, lo, **nasa)      # si se planifica con la cota inferior
    return out


def by_bucket(df: pd.DataFrame, edges: list[int], lo: str = "lo", hi: str = "hi", p: str = "pred") -> pd.DataFrame:
    """Métricas por tramo de RUL real."""
    lab = [f"{a}–{b - 1}" if b < 10_000 else f"≥{a}" for a, b in zip(edges[:-1], edges[1:])]
    b = pd.cut(df["RUL"], edges, right=False, labels=lab)
    rows = []
    for k, g in df.groupby(b, observed=True):
        m = summary(g["RUL"], g[p], g[lo], g[hi])
        rows.append({"tramo": str(k), **m})
    return pd.DataFrame(rows)
