"""Pasada 1 sobre los HDF5: integridad, calidad, agregados por vuelo y muestras.

Salidas (en <NCMAPSS_DATA_DIR>/processed/<modo>/):
  files.parquet           integridad, tamaño y filas de cada archivo
  colstats.parquet        conteo, faltantes, infinitos, media, desvío, mín y máx por variable
  flights_raw.parquet     una fila por vuelo (motor × ciclo) con sus agregados
  seconds_sample.parquet  muestra aleatoria de segundos (para EDA y modelo sano)
  example_flight.parquet  un vuelo completo, segundo a segundo
"""
from __future__ import annotations

import logging
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import io
from .config import Paths, dataset_name

log = logging.getLogger(__name__)
KEY = ["unit", "cycle"]


def _partial_flight_aggs(df: pd.DataFrame, names: dict[str, list[str]]) -> pd.DataFrame:
    """Agregados combinables (sumas, máximos, etc.) de un bloque de filas."""
    g = df.groupby(KEY, sort=False)
    parts = [g.size().rename("n_sec"),
             g[["Fc", "RUL"]].first(),
             g["hs"].agg(["min", "max"]).add_prefix("hs_")]
    sq = df[names["X_s"]] ** 2
    sq.columns = [f"{c}__sq" for c in sq.columns]
    sums = pd.concat([df[KEY], df[names["W"] + names["X_s"] + names["X_v"]], sq], axis=1).groupby(KEY, sort=False).sum()
    parts.append(sums.rename(columns=lambda c: c if c.endswith("__sq") else f"{c}__sum"))
    parts.append(g[names["W"] + names["X_s"]].max().add_suffix("__max"))
    parts.append(g[names["W"]].min().add_suffix("__min"))
    parts.append(g[names["T"]].last())
    return pd.concat(parts, axis=1)


def _combine_flight_aggs(partials: list[pd.DataFrame], names: dict[str, list[str]]) -> pd.DataFrame:
    p = pd.concat(partials)
    g = p.groupby(level=[0, 1], sort=True)
    how = {}
    for c in p.columns:
        if c.endswith(("__sum", "__sq")) or c == "n_sec":
            how[c] = "sum"
        elif c.endswith("__max") or c == "hs_max":
            how[c] = "max"
        elif c.endswith("__min") or c == "hs_min":
            how[c] = "min"
        elif c in names["T"]:
            how[c] = "last"
        else:
            how[c] = "first"
    a = g.agg(how)
    out = pd.DataFrame(index=a.index)
    out["n_sec"] = a["n_sec"]
    out["Fc"] = a["Fc"]
    out["RUL"] = a["RUL"]
    out["hs"] = a["hs_max"]                                   # 1 = vuelo sano
    out["hs_mixed"] = (a["hs_min"] != a["hs_max"]).astype(int)  # control: hs debería ser constante en el vuelo
    for c in names["W"]:
        out[f"{c}_mean"] = a[f"{c}__sum"] / a["n_sec"]
        out[f"{c}_max"] = a[f"{c}__max"]
        out[f"{c}_min"] = a[f"{c}__min"]
    for c in names["X_s"]:
        m = a[f"{c}__sum"] / a["n_sec"]
        out[f"{c}_mean"] = m
        out[f"{c}_std"] = np.sqrt(np.maximum(a[f"{c}__sq"] / a["n_sec"] - m ** 2, 0))
        out[f"{c}_max"] = a[f"{c}__max"]
    for c in names["X_v"]:
        out[f"{c}_mean"] = a[f"{c}__sum"] / a["n_sec"]
    for c in names["T"]:
        out[c] = a[c]
    return out.reset_index()


def _colstats_update(acc: dict, df: pd.DataFrame, cols: list[str]) -> None:
    x = df[cols].to_numpy(dtype=np.float64)
    fin = np.isfinite(x)
    xz = np.where(fin, x, 0.0)
    upd = {
        "n": np.full(len(cols), len(x), dtype=np.int64),
        "nan": np.isnan(x).sum(0),
        "inf": np.isinf(x).sum(0),
        "sum": xz.sum(0),
        "sq": (xz ** 2).sum(0),
        "min": np.where(fin, x, np.inf).min(0),
        "max": np.where(fin, x, -np.inf).max(0),
    }
    if not acc:
        acc.update(upd)
        return
    for k in ("n", "nan", "inf", "sum", "sq"):
        acc[k] = acc[k] + upd[k]
    acc["min"] = np.minimum(acc["min"], upd["min"])
    acc["max"] = np.maximum(acc["max"], upd["max"])


def process_file(path: Path, chunk_rows: int, sample_frac: float, seed: int,
                 max_units: int | None, example: dict) -> dict:
    """Procesa un archivo completo (dev y test). Corre en un proceso aparte."""
    ds = dataset_name(path)
    names = io.variable_names(path)
    cols = names["W"] + names["X_s"] + names["X_v"] + names["T"] + ["RUL"]
    flights, samples, examples, stats = [], [], [], []
    rng = np.random.default_rng([seed, zlib.crc32(ds.encode())])
    for split in io.SPLITS:
        units = None
        if max_units:
            units = set(io.units_in(path, split)[:max_units])
        partials, acc = [], {}
        seen: dict[tuple[int, int], int] = {}    # segundos ya vistos de cada vuelo (para el reloj del vuelo)
        for df in io.iter_chunks(path, split, chunk_rows, units):
            key = df["unit"].astype(np.int64) * 100000 + df["cycle"]
            offset = key.map(seen).fillna(0).to_numpy(dtype=np.int64)
            df["t_sec"] = df.groupby(key.to_numpy(), sort=False).cumcount().to_numpy() + offset
            for k, n in key.value_counts().items():
                seen[k] = seen.get(k, 0) + int(n)
            partials.append(_partial_flight_aggs(df, names))
            _colstats_update(acc, df, cols)
            m = rng.random(len(df)) < sample_frac
            samples.append(df.loc[m].assign(ds=ds, split=split))
            if ds == example["dataset"] and split == "dev":
                ex = df[(df["unit"] == example["unit"]) & (df["cycle"] == example["cycle"])]
                if len(ex):
                    examples.append(ex.assign(ds=ds, split=split))
        fl = _combine_flight_aggs(partials, names)
        fl.insert(0, "split", split)
        fl.insert(0, "ds", ds)
        flights.append(fl)
        st = pd.DataFrame({"variable": cols, **{k: v for k, v in acc.items()}})
        st.insert(0, "split", split)
        st.insert(0, "ds", ds)
        stats.append(st)
    return {
        "flights": pd.concat(flights, ignore_index=True),
        "sample": pd.concat(samples, ignore_index=True),
        "example": pd.concat(examples, ignore_index=True) if examples else None,
        "stats": pd.concat(stats, ignore_index=True),
        "names": names,
    }


def run(cfg: dict, paths: Paths, mode: str, workers: int = 4) -> None:
    files = sorted(paths.raw.glob(cfg["data"]["file_glob"]))
    if not files:
        raise FileNotFoundError(f"No hay archivos {cfg['data']['file_glob']} en {paths.raw}. "
                                "Revisá NCMAPSS_DATA_DIR (ver README).")
    checks = [io.check_file(f) for f in files]
    for c in checks:
        if not c.ok:
            log.warning("Archivo dañado, se excluye: %s (%s)", c.path.name, c.error)
    good = [c.path for c in checks if c.ok]
    max_units = None
    frac = cfg["data"]["seconds_sample_frac"]
    if mode == "sample":
        good = [f for f in good if dataset_name(f) in cfg["sample"]["datasets"]]
        max_units = cfg["sample"]["max_units_per_split"]
        frac = cfg["sample"]["seconds_sample_frac"]

    finfo = pd.DataFrame([{"ds": c.dataset, "file": c.path.name, "size_gb": c.size_bytes / 1e9,
                           "ok": c.ok, "error": c.error,
                           "used": c.path in good,
                           "rows_dev": io.n_rows(c.path, "dev") if c.ok else np.nan,
                           "rows_test": io.n_rows(c.path, "test") if c.ok else np.nan} for c in checks])
    finfo.to_parquet(paths.processed / "files.parquet")

    example = cfg["sample"]["example_flight"] if mode == "sample" else cfg["data"]["example_flight"]
    args = (cfg["data"]["chunk_rows"], frac, cfg["seed"], max_units, example)
    results = []
    with ProcessPoolExecutor(max_workers=min(workers, len(good))) as ex:
        futs = {ex.submit(process_file, f, *args): f for f in good}
        for fut, f in futs.items():
            results.append(fut.result())
            log.info("  listo %s", f.name)

    flights = pd.concat([r["flights"] for r in results], ignore_index=True)
    flights["uid"] = flights["ds"] + "_u" + flights["unit"].astype(str).str.zfill(2)
    flights = flights.sort_values(["ds", "unit", "cycle"]).reset_index(drop=True)
    flights.to_parquet(paths.processed / "flights_raw.parquet")
    pd.concat([r["sample"] for r in results], ignore_index=True).to_parquet(paths.processed / "seconds_sample.parquet")
    pd.concat([r["stats"] for r in results], ignore_index=True).to_parquet(paths.processed / "colstats.parquet")
    exs = [r["example"] for r in results if r["example"] is not None]
    if exs:
        pd.concat(exs, ignore_index=True).to_parquet(paths.processed / "example_flight.parquet")
    pd.Series(results[0]["names"]).to_json(paths.processed / "variable_names.json")
    log.info("Pasada 1: %d vuelos de %d motores en %d archivos", len(flights), flights["uid"].nunique(), len(good))
