"""Tabla de modelado: una fila por vuelo con variables que sólo miran hacia atrás.

Para el vuelo t de un motor se usa únicamente lo observado hasta t:
  - res_<s>        residuo medio del sensor s en el vuelo (medido − motor sano)
  - d_<s>          residuo menos la línea base del propio motor (media de sus primeros vuelos,
                   expandiéndose mientras no haya suficientes): quita la variabilidad de fábrica
  - d_<s>_m<w>     media móvil de d_<s> en los últimos w vuelos
  - d_<s>_slope    pendiente de d_<s> en los últimos w vuelos (velocidad del desgaste)
  - cycle, Fc, horas de vuelo y condiciones medias del vuelo
No se usa el subconjunto (DSxx): en un motor real no se conoce el modo de falla.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from .config import Paths
from .splits import load_splits

log = logging.getLogger(__name__)


def sensor_names(paths: Paths) -> list[str]:
    return json.loads((paths.processed / "variable_names.json").read_text())["X_s"]


def _rolling_slope(s: pd.Series, w: int) -> pd.Series:
    """Pendiente de mínimos cuadrados en una ventana móvil hacia atrás."""
    def slope(y):
        x = np.arange(len(y), dtype=float)
        x -= x.mean()
        return float((x * (y - y.mean())).sum() / (x ** 2).sum()) if len(y) > 1 else 0.0
    return s.rolling(w, min_periods=2).apply(slope, raw=True).fillna(0.0)


def build(flights: pd.DataFrame, sensors: list[str], cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    f = flights.sort_values(["uid", "cycle"]).copy()
    b = cfg["line3"]["baseline_flights"]
    wins = cfg["features"]["rolling_windows"]
    f["hours"] = f["n_sec"] / 3600
    cols = ["cycle", "Fc", "hours", "alt_max", "Mach_mean", "TRA_mean", "T2_mean"]
    g = f.groupby("uid", sort=False)
    new = {}
    for s in sensors:
        r = f[f"res_{s}_mean"]
        new[f"res_{s}"] = r
        base = g[f"res_{s}_mean"].transform(lambda x: x.expanding().mean().where(np.arange(len(x)) < b, x.iloc[:b].mean()))
        d = r - base
        new[f"d_{s}"] = d
        dg = d.groupby(f["uid"], sort=False)
        for w in wins:
            new[f"d_{s}_m{w}"] = dg.transform(lambda x, w=w: x.rolling(w, min_periods=1).mean())
        new[f"d_{s}_slope"] = dg.transform(lambda x: _rolling_slope(x, max(wins)))
    f = pd.concat([f, pd.DataFrame(new, index=f.index)], axis=1)
    feats = cols + list(new)
    return f, feats


def run(cfg: dict, paths: Paths) -> None:
    flights = pd.read_parquet(paths.processed / "flights.parquet")
    splits = load_splits(paths)
    f, feats = build(flights, sensor_names(paths), cfg)
    f = f.merge(splits[["uid", "role"]], on="uid", validate="m:1")
    f["eol"] = f.groupby("uid")["cycle"].transform("max")
    keep = ["uid", "ds", "split", "role", "unit", "cycle", "Fc", "hs", "RUL", "eol"] + [c for c in feats if c not in ("cycle", "Fc")]
    f[keep].to_parquet(paths.processed / "model_table.parquet")
    (paths.processed / "features.json").write_text(json.dumps(feats))
    log.info("Tabla de modelado: %d vuelos · %d variables · roles %s",
             len(f), len(feats), f.drop_duplicates("uid")["role"].value_counts().to_dict())
