"""Piezas compartidas por los modelos: carga de la tabla, guardado de predicciones y metadatos."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Paths

ID_COLS = ["uid", "ds", "role", "unit", "cycle", "Fc", "hs", "RUL", "eol"]


def load_table(paths: Paths) -> tuple[pd.DataFrame, list[str]]:
    t = pd.read_parquet(paths.processed / "model_table.parquet")
    feats = json.loads((paths.processed / "features.json").read_text())
    return t.sort_values(["uid", "cycle"]).reset_index(drop=True), feats


def save_predictions(df: pd.DataFrame, q: np.ndarray, quantiles: list[float], name: str, paths: Paths) -> Path:
    """q: matriz n × n_cuantiles. Se ordena por fila para evitar cuantiles cruzados."""
    q = np.sort(q, axis=1)
    out = df[ID_COLS].copy()
    for j, a in enumerate(quantiles):
        out[f"q{int(round(a * 100)):02d}"] = q[:, j]
    path = paths.outputs / f"preds_{name}.parquet"
    out.to_parquet(path)
    return path


def hardware() -> dict:
    info = {"python": platform.python_version(), "system": f"{platform.system()} {platform.release()}",
            "cpu": platform.processor() or platform.machine()}
    try:
        import torch
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        info["torch"] = torch.__version__
    except ImportError:
        info["gpu"] = None
    try:  # en Colab
        import google.colab  # noqa: F401
        info["where"] = "Google Colab"
    except ImportError:
        info["where"] = "computadora local"
    return info
