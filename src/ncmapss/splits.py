"""Partición por motor, sin fuga de información entre unidades.

- test : los motores de prueba oficiales de NASA (partición `*_test` de cada archivo).
- calib: parte de los motores de desarrollo, reservada para calibrar los intervalos conformales.
- val  : parte de los motores de desarrollo, para early stopping.
- train: el resto de los motores de desarrollo.

La división se hace dentro de cada subconjunto para que todos los escenarios de falla
estén representados en cada rol.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Paths


def make_splits(flights: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    units = flights[["uid", "ds", "split", "unit", "Fc"]].drop_duplicates("uid")
    for ds, g in units.groupby("ds", sort=True):
        test = g[g["split"] == "test"]
        rows += [(u, ds, "test") for u in test["uid"]]
        dev = g[g["split"] == "dev"]["uid"].to_numpy()
        dev = dev[rng.permutation(len(dev))]
        n_cal = max(1, int(round(cfg["split"]["calib_frac"] * len(dev))))
        n_val = max(1, int(round(cfg["split"]["val_frac"] * len(dev))))
        rows += [(u, ds, "calib") for u in dev[:n_cal]]
        rows += [(u, ds, "val") for u in dev[n_cal:n_cal + n_val]]
        rows += [(u, ds, "train") for u in dev[n_cal + n_val:]]
    out = pd.DataFrame(rows, columns=["uid", "ds", "role"])
    return out.merge(units[["uid", "Fc"]], on="uid")


def load_splits(paths: Paths) -> pd.DataFrame:
    return pd.read_parquet(paths.processed / "splits.parquet")
