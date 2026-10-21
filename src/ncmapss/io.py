"""Lectura de los archivos HDF5 de N-CMAPSS por bloques.

Cada archivo trae dos particiones (dev = desarrollo, test = prueba) con seis bloques
de variables: W (condiciones de vuelo), X_s (sensores), X_v (sensores virtuales),
T (parámetros de salud), Y (RUL) y A (unit, cycle, Fc, hs). Hay un registro por
segundo de vuelo, así que se lee por tramos de filas para no agotar la memoria.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import pandas as pd

from .config import dataset_name

BLOCKS = ["W", "X_s", "X_v", "T", "A"]
SPLITS = ["dev", "test"]


@dataclass
class FileCheck:
    dataset: str
    path: Path
    ok: bool
    error: str = ""
    size_bytes: int = 0


def check_file(path: Path) -> FileCheck:
    """Abre el archivo y lee la última fila de cada bloque: detecta archivos truncados."""
    size = path.stat().st_size
    try:
        with h5py.File(path, "r") as h:
            for s in SPLITS:
                for b in BLOCKS + ["Y"]:
                    _ = h[f"{b}_{s}"][-1]
        return FileCheck(dataset_name(path), path, True, size_bytes=size)
    except Exception as e:  # el archivo DS08d de Kaggle viene truncado
        return FileCheck(dataset_name(path), path, False, f"{type(e).__name__}: {e}", size)


def variable_names(path: Path) -> dict[str, list[str]]:
    with h5py.File(path, "r") as h:
        return {b: [v.decode() if isinstance(v, bytes) else str(v) for v in np.array(h[f"{b}_var"]).ravel()]
                for b in BLOCKS}


def n_rows(path: Path, split: str) -> int:
    with h5py.File(path, "r") as h:
        return h[f"A_{split}"].shape[0]


def iter_chunks(path: Path, split: str, chunk_rows: int, units: set[int] | None = None) -> Iterator[pd.DataFrame]:
    """Devuelve DataFrames de a `chunk_rows` filas con todas las variables y el RUL.

    `units` permite quedarse sólo con algunos motores (modo rápido).
    """
    names = variable_names(path)
    with h5py.File(path, "r") as h:
        n = h[f"A_{split}"].shape[0]
        for start in range(0, n, chunk_rows):
            stop = min(start + chunk_rows, n)
            a = h[f"A_{split}"][start:stop]
            mask = None
            if units is not None:
                mask = np.isin(a[:, 0].astype(int), list(units))
                if not mask.any():
                    continue
            cols = {}
            for b in BLOCKS:
                arr = a if b == "A" else h[f"{b}_{split}"][start:stop]
                if mask is not None:
                    arr = arr[mask]
                for j, v in enumerate(names[b]):
                    cols[v] = arr[:, j]
            y = h[f"Y_{split}"][start:stop].ravel()
            cols["RUL"] = y[mask] if mask is not None else y
            df = pd.DataFrame(cols)
            for c in ("unit", "cycle", "Fc", "hs"):
                df[c] = df[c].astype(np.int16)
            yield df


def units_in(path: Path, split: str) -> list[int]:
    with h5py.File(path, "r") as h:
        return sorted(np.unique(h[f"A_{split}"][:, 0]).astype(int).tolist())
