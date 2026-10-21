"""Modelo de "motor sano" y residuos.

Cada sensor depende sobre todo de cómo se vuela (altitud, Mach, potencia, temperatura
de entrada). Una red chica aprende esa relación usando sólo segundos de vuelos SANOS
(hs = 1) de los motores de entrenamiento. Luego, para cada segundo:

    residuo = valor medido − valor esperado para un motor sano

El residuo queda cerca de cero en un motor sano y se aleja a medida que se desgasta.
La pasada 2 sobre los .h5 calcula el residuo de todos los segundos y lo resume por vuelo.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import io
from .config import Paths, dataset_name
from .splits import make_splits

log = logging.getLogger(__name__)


def _names(paths: Paths) -> dict[str, list[str]]:
    return json.loads((paths.processed / "variable_names.json").read_text())


class HealthyMLP(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: list[int]):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _load_model(paths: Paths, device: str = "cpu"):
    ck = torch.load(paths.models / "health_mlp.pt", map_location=device, weights_only=False)
    m = HealthyMLP(len(ck["inputs"]), len(ck["outputs"]), ck["hidden"]).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


def predict(model, ck, w: np.ndarray, device: str = "cpu", batch: int = 1_000_000) -> np.ndarray:
    xm, xs, ym, ys = (np.asarray(ck[k]) for k in ("x_mean", "x_std", "y_mean", "y_std"))
    out = np.empty((len(w), len(ck["outputs"])), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(w), batch):
            x = torch.as_tensor((w[i:i + batch] - xm) / xs, dtype=torch.float32, device=device)
            out[i:i + batch] = model(x).cpu().numpy() * ys + ym
    return out


def train(cfg: dict, paths: Paths, mode: str) -> None:
    hc = cfg["health_model"]
    names = _names(paths)
    flights = pd.read_parquet(paths.processed / "flights_raw.parquet")
    splits = make_splits(flights, cfg)
    splits.to_parquet(paths.processed / "splits.parquet")

    s = pd.read_parquet(paths.processed / "seconds_sample.parquet")
    s["uid"] = s["ds"] + "_u" + s["unit"].astype(str).str.zfill(2)
    s = s.merge(splits[["uid", "role"]], on="uid")
    inputs, outputs = names["W"], names["X_s"] + names["X_v"]
    tr = s[(s["role"] == "train") & (s["hs"] == 1)]
    va = s[(s["role"] == "val") & (s["hs"] == 1)]
    if len(tr) > hc["max_train_rows"]:
        tr = tr.sample(hc["max_train_rows"], random_state=cfg["seed"])

    x, y = tr[inputs].to_numpy(), tr[outputs].to_numpy()
    ck = {"inputs": inputs, "outputs": outputs, "hidden": hc["hidden"],
          "x_mean": x.mean(0), "x_std": x.std(0) + 1e-9, "y_mean": y.mean(0), "y_std": y.std(0) + 1e-9}
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HealthyMLP(len(inputs), len(outputs), hc["hidden"]).to(device)
    X = torch.as_tensor((x - ck["x_mean"]) / ck["x_std"], dtype=torch.float32, device=device)
    Y = torch.as_tensor((y - ck["y_mean"]) / ck["y_std"], dtype=torch.float32, device=device)
    steps = hc["steps_sample"] if mode == "sample" else hc["steps"]
    opt = torch.optim.AdamW(model.parameters(), lr=hc["lr"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=hc["lr"], total_steps=steps)
    gen = torch.Generator(device=device).manual_seed(cfg["seed"])
    run = 0.0
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(X), (hc["batch_size"],), device=device, generator=gen)
        loss = nn.functional.mse_loss(model(X[idx]), Y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % (steps // 5) == 0:
            log.info("  modelo sano · paso %d/%d · MSE (estandarizado) %.5f", step, steps, run)
    ck["state"] = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(ck, paths.models / "health_mlp.pt")

    # Qué tan bien explica el vuelo a cada sensor, en motores sanos NO vistos (validación)
    model.eval()
    pred = predict(model, ck, va[inputs].to_numpy(), device)
    yv = va[outputs].to_numpy()
    r2 = 1 - ((yv - pred) ** 2).sum(0) / ((yv - yv.mean(0)) ** 2).sum(0)
    metrics = {"r2_val": dict(zip(outputs, r2.round(6).tolist())),
               "n_train_seconds": int(len(tr)), "n_val_seconds": int(len(va)),
               "train_units": int(tr["uid"].nunique()), "device": device}
    (paths.outputs / "health_model.json").write_text(json.dumps(metrics, indent=1))
    log.info("  R² mediano en validación (sensores físicos): %.4f", np.median(r2[: len(names["X_s"])]))


def _residual_file(path: Path, paths: Paths, chunk_rows: int, units_by_split: dict) -> pd.DataFrame:
    torch.set_num_threads(4)
    model, ck = _load_model(paths)
    ds = dataset_name(path)
    outs = ck["outputs"]
    frames = []
    for split in io.SPLITS:
        units = units_by_split.get(split)
        if not units:
            continue
        parts = []
        for df in io.iter_chunks(path, split, chunk_rows, set(units)):
            r = df[outs].to_numpy() - predict(model, ck, df[ck["inputs"]].to_numpy())
            rd = pd.DataFrame(r, columns=[f"res_{c}" for c in outs])
            rd["unit"], rd["cycle"] = df["unit"].to_numpy(), df["cycle"].to_numpy()
            sq = rd[[f"res_{c}" for c in outs]] ** 2
            sq.columns = [f"{c}__sq" for c in sq.columns]
            g = pd.concat([rd, sq], axis=1).groupby(["unit", "cycle"])
            agg = g.sum()
            agg["n"] = g.size()
            parts.append(agg)
        a = pd.concat(parts).groupby(level=[0, 1]).sum()
        res = pd.DataFrame(index=a.index)
        for c in outs:
            m = a[f"res_{c}"] / a["n"]
            res[f"res_{c}_mean"] = m
            res[f"res_{c}_std"] = np.sqrt(np.maximum(a[f"res_{c}__sq"] / a["n"] - m ** 2, 0))
        res = res.reset_index()
        res.insert(0, "split", split)
        res.insert(0, "ds", ds)
        frames.append(res)
    return pd.concat(frames, ignore_index=True)


def residuals(cfg: dict, paths: Paths, mode: str, workers: int = 4) -> None:
    flights = pd.read_parquet(paths.processed / "flights_raw.parquet")
    files = pd.read_parquet(paths.processed / "files.parquet")
    used = files[files["used"]]
    jobs = {}
    for _, f in used.iterrows():
        sub = flights[flights["ds"] == f["ds"]]
        jobs[paths.raw / f["file"]] = {sp: sorted(g["unit"].unique().tolist()) for sp, g in sub.groupby("split")}
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        futs = [ex.submit(_residual_file, p, paths, cfg["data"]["chunk_rows"], u) for p, u in jobs.items()]
        res = pd.concat([f.result() for f in futs], ignore_index=True)
    out = flights.merge(res, on=["ds", "split", "unit", "cycle"], how="left", validate="1:1")
    if out.filter(like="res_").isna().any().any():
        raise RuntimeError("Hay vuelos sin residuo: revisar la pasada 2")
    out.to_parquet(paths.processed / "flights.parquet")
    log.info("Pasada 2: residuos de %d vuelos", len(out))
