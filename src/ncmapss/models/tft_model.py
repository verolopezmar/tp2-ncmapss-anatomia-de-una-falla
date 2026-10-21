"""Modelo principal: Temporal Fusion Transformer (pytorch-forecasting) con salida por cuantiles.

Planteo sin fuga de información:
  - Cada muestra predice el RUL del vuelo t mirando los últimos `max_encoder_length` vuelos
    del mismo motor (encoder) y las variables del vuelo t (decoder, que ya ocurrió).
  - El RUL pasado NO es una entrada (en un motor real nunca se conoce): el target no está
    entre las variables del encoder.
  - El RUL se normaliza con una media y un desvío GLOBALES. Normalizar por motor revelaría
    su vida total.
  - Clase de vuelo (Fc) como variable estática del motor.
El TFT necesita al menos un vuelo de historia: no predice el vuelo 1 de cada motor.

Puede correr en la computadora local (GPU si hay) o en Google Colab (notebooks/colab_tft.ipynb),
leyendo sólo model_table.parquet y features.json.
"""
from __future__ import annotations

import json
import logging
import time
import warnings

import numpy as np
import pandas as pd

from ..config import Paths
from .common import hardware, load_table, save_predictions

log = logging.getLogger(__name__)


def _prepare(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["Fc_cat"] = t["Fc"].astype(str)
    t["cycle"] = t["cycle"].astype(int)
    t["RUL"] = t["RUL"].astype(float)
    return t


def run(cfg: dict, paths: Paths, mode: str) -> None:
    warnings.filterwarnings("ignore")
    import lightning.pytorch as pl
    import torch
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import NaNLabelEncoder, TorchNormalizer
    from pytorch_forecasting.metrics import QuantileLoss

    base = cfg["tft"]
    torch.set_float32_matmul_precision("high")
    t, feats = load_table(paths)
    t = _prepare(t)
    qs = base["quantiles"]
    accel = "gpu" if torch.cuda.is_available() else "cpu"
    search = base["search"][:1] if mode == "sample" else base["search"]

    def feature_set(name: str) -> list[str]:
        ctx = ["cycle", "hours", "alt_max", "Mach_mean", "TRA_mean", "T2_mean"]
        if name == "smooth":
            return ctx + [f for f in feats if f.startswith("d_") and (f.endswith("_slope") or "_m" in f)]
        if name == "delta":
            return ctx + [f for f in feats if f.startswith("d_") and f.count("_") == 1]
        return [f for f in feats if f != "Fc"]

    def train_one(tc: dict, k: int):
        pl.seed_everything(cfg["seed"], workers=True)
        reals = feature_set(tc["feature_set"])
        train_ds = TimeSeriesDataSet(
            t[t["role"] == "train"], time_idx="cycle", target="RUL", group_ids=["uid"],
            max_encoder_length=tc["max_encoder_length"], min_encoder_length=1,
            max_prediction_length=1, min_prediction_length=1,
            static_categoricals=["Fc_cat"], categorical_encoders={"Fc_cat": NaNLabelEncoder(add_nan=True)},
            time_varying_known_reals=reals, time_varying_unknown_reals=[],
            target_normalizer=TorchNormalizer(method="standard"), add_target_scales=False,
            add_encoder_length=True, add_relative_time_idx=False)
        assert "RUL" not in train_ds.reals, "el RUL no puede ser una entrada del modelo"
        val_ds = TimeSeriesDataSet.from_dataset(train_ds, t[t["role"] == "val"], stop_randomization=True)
        bs = tc["batch_size"]
        tr_dl = train_ds.to_dataloader(train=True, batch_size=bs, num_workers=0)
        va_dl = val_ds.to_dataloader(train=False, batch_size=bs * 4, num_workers=0)
        model = TemporalFusionTransformer.from_dataset(
            train_ds, hidden_size=tc["hidden_size"], attention_head_size=tc["attention_head_size"], dropout=tc["dropout"],
            hidden_continuous_size=tc["hidden_continuous_size"], learning_rate=tc["learning_rate"],
            loss=QuantileLoss(quantiles=qs), log_interval=-1, reduce_on_plateau_patience=4)
        ckpt = ModelCheckpoint(dirpath=paths.models, filename=f"tft-cfg{k}", monitor="val_loss", mode="min", save_top_k=1)
        trainer = pl.Trainer(
            max_epochs=tc["max_epochs_sample"] if mode == "sample" else tc["max_epochs"], accelerator=accel, devices=1,
            gradient_clip_val=0.1, callbacks=[EarlyStopping(monitor="val_loss", patience=tc["patience"], mode="min"), ckpt],
            logger=False, enable_progress_bar=False, enable_model_summary=False)
        t0 = time.time()
        trainer.fit(model, tr_dl, va_dl)
        return {"k": k, "tc": tc, "val_loss": float(ckpt.best_model_score), "ckpt": ckpt.best_model_path,
                "epochs": int(trainer.current_epoch), "secs": time.time() - t0, "train_ds": train_ds, "va_dl": va_dl}

    runs = []
    for k, over in enumerate(search):
        tc = {**base, **over}
        r = train_one(tc, k)
        runs.append(r)
        log.info("  TFT config %d %s · val_loss %.4f · %d épocas · %.0f s", k, over or "(base)", r["val_loss"], r["epochs"], r["secs"])
    best_run = min(runs, key=lambda r: r["val_loss"])
    tc, train_ds, va_dl = best_run["tc"], best_run["train_ds"], best_run["va_dl"]
    bs = tc["batch_size"]
    secs = sum(r["secs"] for r in runs)
    best = TemporalFusionTransformer.load_from_checkpoint(best_run["ckpt"])

    # Predicción para todos los vuelos con al menos un vuelo de historia, de todos los roles
    all_ds = TimeSeriesDataSet.from_dataset(train_ds, t, stop_randomization=True, predict=False)
    dl = all_ds.to_dataloader(train=False, batch_size=bs * 8, num_workers=0)
    pr = best.predict(dl, mode="quantiles", return_index=True, return_x=True,
                      trainer_kwargs={"accelerator": accel, "logger": False, "enable_progress_bar": False})
    q = pr.output.detach().cpu().numpy()[:, 0, :]
    pred = pr.index.assign(enc_len=pr.x["encoder_lengths"].cpu().numpy(), **{f"_q{j}": q[:, j] for j in range(len(qs))})
    # pytorch-forecasting puede generar varias ventanas para el mismo vuelo con historias de distinto
    # largo: se usa la de historia más larga (lo que haría el sistema en operación)
    pred = pred.sort_values("enc_len").drop_duplicates(["uid", "cycle"], keep="last")
    m = t.merge(pred, on=["uid", "cycle"], how="inner", validate="1:1")
    save_predictions(m, m[[f"_q{j}" for j in range(len(qs))]].to_numpy(), qs, "tft", paths)

    # Importancia de variables (selección de variables del TFT, promedio sobre validación)
    raw = best.predict(va_dl, mode="raw", return_x=True, trainer_kwargs={"accelerator": accel, "logger": False, "enable_progress_bar": False})
    interp = best.interpret_output(raw.output, reduction="sum")
    enc = interp["encoder_variables"].cpu().numpy()
    enc = enc / enc.sum()
    imp = dict(zip(best.encoder_variables, np.round(enc * 100, 4).tolist()))
    meta = {"model": "Temporal Fusion Transformer", "epochs": best_run["epochs"], "best_val_loss": best_run["val_loss"],
            "train_seconds": round(secs, 1), "quantiles": qs, "n_params": int(sum(p.numel() for p in best.parameters())),
            "importance": imp, "hardware": hardware(), "device": "GPU" if accel == "gpu" else "CPU",
            "config": {k: tc[k] for k in ("feature_set", "max_encoder_length", "hidden_size", "attention_head_size", "dropout", "learning_rate", "batch_size")},
            "search": [{"config": search[r["k"]] or "base", "val_loss": round(r["val_loss"], 4), "epochs": r["epochs"]} for r in runs],
            "chosen": best_run["k"]}
    (paths.outputs / "model_tft.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    log.info("  TFT elegido: config %d · %d épocas · búsqueda total %.0f s (%s) · %d parámetros",
             best_run["k"], meta["epochs"], secs, meta["device"], meta["n_params"])
