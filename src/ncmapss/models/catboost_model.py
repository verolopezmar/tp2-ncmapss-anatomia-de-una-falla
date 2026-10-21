"""Baseline: CatBoost con regresión multi-cuantil (5 %, 50 %, 95 %).

Cada fila es un vuelo y sus variables sólo miran hacia atrás (ver features.py), así que un
modelo tabular puede predecir el RUL vuelo a vuelo. Early stopping con los motores de validación.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
from catboost import CatBoostRegressor, Pool

from ..config import Paths
from .common import hardware, load_table, save_predictions

log = logging.getLogger(__name__)


def run(cfg: dict, paths: Paths, mode: str) -> None:
    cc = cfg["catboost"]
    t, feats = load_table(paths)
    tr, va = t[t["role"] == "train"], t[t["role"] == "val"]
    qs = cc["quantiles"]
    loss = "MultiQuantile:alpha=" + ",".join(str(a) for a in qs)
    model = CatBoostRegressor(
        loss_function=loss, iterations=cc["iterations_sample"] if mode == "sample" else cc["iterations"],
        learning_rate=cc["learning_rate"], depth=cc["depth"], random_seed=cfg["seed"],
        od_type="Iter", od_wait=cc["early_stopping_rounds"], use_best_model=True, verbose=False,
        allow_writing_files=False, thread_count=-1)
    t0 = time.time()
    model.fit(Pool(tr[feats], tr["RUL"]), eval_set=Pool(va[feats], va["RUL"]))
    secs = time.time() - t0
    q = model.predict(t[feats])
    save_predictions(t, np.asarray(q), qs, "catboost", paths)
    model.save_model(str(paths.models / "catboost.cbm"))
    imp = dict(zip(feats, model.get_feature_importance().round(4).tolist()))
    meta = {"model": "CatBoost", "best_iteration": int(model.get_best_iteration()), "train_seconds": round(secs, 1),
            "quantiles": qs, "n_features": len(feats), "importance": imp, "hardware": hardware(), "device": "CPU"}
    (paths.outputs / "model_catboost.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    log.info("  CatBoost: %d iteraciones (mejor %d) en %.1f s", model.tree_count_, meta["best_iteration"], secs)
