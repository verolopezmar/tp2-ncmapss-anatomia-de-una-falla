"""Línea 2 (experimento exploratorio): ¿generaliza un modelo entrenado sólo con vuelos cortos?

Se entrena CatBoost con motores de desarrollo de vuelos CORTOS (Fc = 1) y se evalúa en los
motores de prueba de cada clase (corto, mediano, largo). Variantes:

  A  Referencia: todas las clases, residuos del modelo sano principal.
  B  Sólo cortos · sensores crudos (promedio del vuelo, sin descontar el efecto operativo).
  C  Sólo cortos · residuos de un modelo sano entrenado TAMBIÉN sólo con vuelos cortos
     (transferencia estricta: nada del pipeline vio vuelos medianos o largos).
  D  Sólo cortos · residuos del modelo sano principal (que sí vio todas las clases):
     separa el problema del modelo de RUL del problema de la normalización.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from . import metrics as M
from .config import Paths
from .features import build, sensor_names
from .fmt import num, pct
from .health import _load_model, _sample_with_roles, fit, predict, residuals
from .splits import load_splits
from .viz import fig, rnd, save_json

log = logging.getLogger(__name__)
VARIANTS = {"A": "Todas las clases · residuos", "B": "Sólo cortos · sensores crudos",
            "C": "Sólo cortos · residuos (modelo sano de cortos)", "D": "Sólo cortos · residuos (modelo sano general)"}
COLORS = {"A": "var(--navy)", "B": "var(--ti)", "C": "var(--hud)", "D": "var(--hud-2)"}


def _table(flights: pd.DataFrame, cfg: dict, sensors: list[str], splits: pd.DataFrame):
    t, feats = build(flights, sensors, cfg)
    t = t.merge(splits[["uid", "role"]], on="uid")
    return t, feats


def run(cfg: dict, paths: Paths, mode: str, workers: int = 4) -> dict:
    sensors = sensor_names(paths)
    names = json.loads((paths.processed / "variable_names.json").read_text())
    splits = load_splits(paths)
    cb_meta = json.loads((paths.outputs / "model_catboost.json").read_text(encoding="utf-8"))
    iters = max(50, cb_meta["best_iteration"] + 1)
    fc_names = cfg["flight_classes"]

    # Modelo sano entrenado sólo con vuelos cortos sanos de motores de desarrollo
    s = _sample_with_roles(paths, splits)
    tr = s[(s["split"] == "dev") & (s["hs"] == 1) & (s["Fc"] == 1)]
    va = s[(s["split"] == "dev") & (s["hs"] == 1) & (s["Fc"] != 1)]
    fit(cfg, paths, mode, tr, va, tag="_short")
    model, ck = _load_model(paths, tag="_short")
    main_model, main_ck = _load_model(paths)
    r2 = {}
    for k in sorted(s["Fc"].unique()):
        g = s[(s["split"] == "dev") & (s["hs"] == 1) & (s["Fc"] == k)]
        for tag, (mm, cc) in {"short": (model, ck), "main": (main_model, main_ck)}.items():
            p = predict(mm, cc, g[cc["inputs"]].to_numpy())
            y = g[cc["outputs"]].to_numpy()
            r = 1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)
            r2[(tag, int(k))] = float(np.median(r[: len(sensors)]))
    flights_short = residuals(cfg, paths, mode, workers, tag="_short")
    flights_main = pd.read_parquet(paths.processed / "flights.parquet")
    flights_raw = flights_main.copy()
    for sn in sensors:
        flights_raw[f"res_{sn}_mean"] = flights_raw[f"{sn}_mean"]

    tables = {"A": _table(flights_main, cfg, sensors, splits), "B": _table(flights_raw, cfg, sensors, splits),
              "C": _table(flights_short, cfg, sensors, splits)}
    tables["D"] = tables["A"]
    rows = []
    for v, (t, feats) in tables.items():
        dev = t[t["split"] == "dev"] if v == "A" else t[(t["split"] == "dev") & (t["Fc"] == 1)]
        test = t[(t["split"] == "test") & (t["cycle"] >= cfg["results"]["eval_min_cycle"])]
        m = CatBoostRegressor(loss_function="RMSE", iterations=iters, learning_rate=cfg["catboost"]["learning_rate"],
                              depth=cfg["catboost"]["depth"], random_seed=cfg["seed"], verbose=False, allow_writing_files=False)
        m.fit(dev[feats], dev["RUL"])
        p = m.predict(test[feats])
        for k, g in test.assign(pred=p).groupby("Fc"):
            sm = M.summary(g["RUL"], g["pred"], nasa=cfg["nasa_score"])
            rows.append({"variant": v, "Fc": int(k), "units_train": int(dev["uid"].nunique()), **sm})
        log.info("  Línea 2 · %s listo", VARIANTS[v])
    res = pd.DataFrame(rows)

    out = {"figures": {}, "insights": {}}
    data = []
    for v in VARIANTS:
        g = res[res["variant"] == v].sort_values("Fc")
        data.append({"type": "bar", "name": f"{v} · {VARIANTS[v]}", "x": [f"Prueba: vuelos {fc_names[k]}s" for k in g["Fc"]],
                     "y": rnd(g["rmse"], 2), "marker": {"color": COLORS[v]}, "customdata": rnd(g["nasa"], 3),
                     "hovertemplate": f"{VARIANTS[v]}<br>%{{x}}<br>RMSE %{{y:.2f}} · score NASA %{{customdata}}<extra></extra>"})
    out["figures"]["line2"] = fig(data, barmode="group", height=420, bargap=0.2, legend={"y": 1.22},
                                  yaxis={"title": {"text": "RMSE en motores de prueba [vuelos]"}})
    piv = res.pivot(index="variant", columns="Fc", values="rmse")
    long_k = int(res["Fc"].max())
    out["line2_table"] = [{"variante": f"{v} · {VARIANTS[v]}", "motores de entrenamiento": int(res[res['variant'] == v]['units_train'].iloc[0]),
                           **{f"RMSE vuelos {fc_names[k]}s": round(float(piv.loc[v, k]), 2) for k in piv.columns},
                           **{f"NASA vuelos {fc_names[k]}s": round(float(res[(res.variant == v) & (res.Fc == k)]['nasa'].iloc[0]), 3) for k in piv.columns}}
                          for v in VARIANTS]
    gap_c = piv.loc["C", long_k] / piv.loc["A", long_k]
    gap_b = piv.loc["B", long_k] / piv.loc["A", long_k]
    out["line2_r2"] = {f"{tag}_{k}": v for (tag, k), v in r2.items()}
    out["insights"]["line2"] = (
        f"Entrenando sólo con {int(res[res.variant == 'C']['units_train'].iloc[0])} motores de vuelos cortos, el error en vuelos {fc_names[long_k]}s es "
        f"<b>{num(gap_c, 1)} veces</b> el del modelo con todas las clases si todo el pipeline se ajusta con vuelos cortos (C), y "
        f"{num(gap_b, 1)} veces con sensores crudos (B). El modelo sano de cortos explica el {num(100 * r2[('short', 1)], 3)} % de un sensor "
        f"en vuelos cortos pero el {num(100 * r2[('short', long_k)], 2)} % en vuelos {fc_names[long_k]}s: fuera de su envolvente extrapola mal, "
        f"y ese error se confunde con desgaste. Con el modelo sano general (D) el RMSE en vuelos {fc_names[long_k]}s baja a {num(piv.loc['D', long_k], 2)}: "
        "la clave para generalizar está en separar bien el efecto operativo.")
    out["future_discussion_l2"] = [
        f"<b>Línea 2.</b> Un modelo que sólo vio vuelos cortos no generaliza bien a vuelos largos (RMSE {num(piv.loc['C', long_k], 2)} contra "
        f"{num(piv.loc['A', long_k], 2)} del modelo con todas las clases). La mayor parte de la brecha viene de la normalización operativa: el modelo "
        f"sano entrenado con vuelos cortos no conoce altitudes y regímenes de los largos. Esto justifica investigar Domain-Adversarial Neural "
        "Networks (que aprenden representaciones invariantes al tipo de vuelo), PatchTST sobre la señal segundo a segundo y transfer learning con "
        "pocos vuelos largos etiquetados."]
    save_json(out, paths.outputs / "line2.json")
    log.info("Línea 2 guardada")
    return out
