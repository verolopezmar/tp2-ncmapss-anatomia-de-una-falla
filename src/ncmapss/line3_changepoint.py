"""Línea 3 (experimento exploratorio): detección temprana del inicio de la degradación.

Índice de salud (HI) por vuelo: distancia de Mahalanobis de los 14 residuos físicos (respecto de
la línea base de cada motor) a la nube de vuelos sanos de los motores de entrenamiento. Sobre el
HI se comparan tres detectores contra el inicio real de la degradación anormal (`hs` pasa a 0):

  CUSUM online      acumula los excesos del HI sobre lo normal y da la alarma al superar h.
                    Sólo usa vuelos pasados: es desplegable en operación.
  PELT offline      (ruptures) primer punto de cambio de la serie completa: mira el futuro,
                    sirve como referencia de lo que se podría lograr.
  Umbral sobre T48  media móvil del residuo de T48 por encima de z_alarm: la regla más simple.

Para compararlos en igualdad de condiciones, el umbral de cada detector se elige en los motores
de DESARROLLO como el más sensible que no supere una tasa fijada de falsas alarmas (alarma antes
del inicio real). Luego se evalúa en los motores de prueba: demora (vuelo de alarma − vuelo de
inicio real), vuelos de anticipación a la falla y fracción del desgaste total ya ocurrida.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import ruptures as rpt

from .config import Paths
from .features import sensor_names
from .fmt import num, pct
from .viz import fig, rnd, save_json

log = logging.getLogger(__name__)
METHODS = {"cusum": "CUSUM online", "pelt": "PELT offline", "t48": "Umbral sobre T48"}
COLORS = {"cusum": "var(--orange)", "pelt": "var(--hud)", "t48": "var(--ti)"}


def health_index(t: pd.DataFrame, sensors: list[str], ref_mask: pd.Series) -> np.ndarray:
    D = t[[f"d_{s}" for s in sensors]].to_numpy()
    R = D[ref_mask.to_numpy()]
    mu = R.mean(0)
    cov = np.cov(R, rowvar=False)
    cov += np.eye(len(sensors)) * 1e-6 * np.trace(cov) / len(sensors)
    inv = np.linalg.inv(cov)
    X = D - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", X, inv, X) / len(sensors))


def cusum_alarm(u: np.ndarray, k: float, h: float) -> tuple[int | None, np.ndarray]:
    s, S = 0.0, np.zeros(len(u))
    first = None
    for i, x in enumerate(u):
        s = max(0.0, s + x - k)
        S[i] = s
        if first is None and s > h:
            first = i
    return first, S


def run(cfg: dict, paths: Paths) -> dict:
    lc, e = cfg["line3"], cfg["eda"]
    sensors = sensor_names(paths)
    t = pd.read_parquet(paths.processed / "model_table.parquet").sort_values(["uid", "cycle"]).reset_index(drop=True)
    fl = pd.read_parquet(paths.processed / "flights.parquet")[["uid", "cycle"] + json.loads((paths.processed / "variable_names.json").read_text())["T"]]
    t = t.merge(fl, on=["uid", "cycle"])
    ref = (t["role"] == "train") & (t["hs"] == 1)
    t["HI"] = health_index(t, sensors, ref)
    mu, sd = t.loc[ref, "HI"].mean(), t.loc[ref, "HI"].std()
    t["u"] = (t["HI"] - mu) / sd
    sd48 = t.loc[ref, "d_T48"].std()
    t["z48"] = (t["d_T48"] / sd48).groupby(t["uid"]).transform(lambda x: x.rolling(e["edge_flights"], min_periods=1).mean())

    T = json.loads((paths.processed / "variable_names.json").read_text())["T"]
    units = {uid: g.sort_values("cycle") for uid, g in t.groupby("uid", sort=True)}
    onset = {u: int(g.loc[g["hs"] == 0, "cycle"].min()) for u, g in units.items()}

    def first_index(method: str, g: pd.DataFrame, thr: float):
        if method == "cusum":
            return cusum_alarm(g["u"].to_numpy(), lc["cusum_k"], thr)[0]
        if method == "t48":
            over = np.where(g["z48"].to_numpy() > thr)[0]
            return int(over[0]) if len(over) else None
        bk = rpt.Pelt(model="l2", min_size=3).fit(g["HI"].to_numpy()).predict(pen=thr)
        return bk[0] if len(bk) > 1 else None

    # Calibración del umbral en motores de desarrollo: el más sensible con falsas alarmas ≤ objetivo
    grids = {"cusum": lc["cusum_h_grid"], "t48": lc["t48_z_grid"], "pelt": lc["pelt_penalty_grid"]}
    dev = [u for u, g in units.items() if g["role"].iloc[0] != "test"]
    thr = {}
    for m, (a, b, n) in grids.items():
        for th in np.linspace(a, b, int(n)):
            fa = np.mean([(lambda i: i is not None and units[u]["cycle"].iloc[i] < onset[u])(first_index(m, units[u], th)) for u in dev])
            if fa <= lc["target_false_alarm"]:
                thr[m] = float(th)
                break
        else:
            thr[m] = float(b)
        log.info("  Línea 3 · umbral %s = %.2f (calibrado en %d motores de desarrollo)", m, thr[m], len(dev))

    rows, cus = [], {}
    for uid, g in units.items():
        cyc = g["cycle"].to_numpy()
        eol = int(cyc.max())
        cus[uid] = cusum_alarm(g["u"].to_numpy(), lc["cusum_k"], thr["cusum"])[1]
        loss = (g[T].iloc[0] - g[T]).to_numpy()
        main = int(np.argmax(loss[-1]))
        frac = loss[:, main] / max(loss[-1, main], 1e-12)
        for m in METHODS:
            i = first_index(m, g, thr[m])
            det = int(cyc[i]) if i is not None and i < len(cyc) else None
            rows.append({"uid": uid, "role": g["role"].iloc[0], "method": m, "onset": onset[uid], "eol": eol, "detect": det,
                         "delay": det - onset[uid] if det is not None else np.nan,
                         "lead": eol - det if det is not None else np.nan,
                         "wear_frac": float(frac[i]) if det is not None else np.nan})
    r = pd.DataFrame(rows)
    test = r[r["role"] == "test"]

    summ = []
    for m in METHODS:
        g = test[test["method"] == m]
        det = g["detect"].notna()
        summ.append({"método": METHODS[m], "motores": int(len(g)), "detectados [%]": round(100 * det.mean(), 1),
                     "alarma antes del inicio real [%]": round(100 * (g["delay"] < 0).mean(), 1),
                     "demora mediana [vuelos]": float(np.nanmedian(g["delay"])) if det.any() else None,
                     "anticipación mediana a la falla [vuelos]": float(np.nanmedian(g["lead"])) if det.any() else None,
                     "desgaste ocurrido al alarmar [%]": round(100 * float(np.nanmedian(g["wear_frac"])), 1) if det.any() else None})
    out = {"figures": {}, "insights": {}, "line3_table": summ, "thresholds": thr}

    data = []
    for m in METHODS:
        g = test[test["method"] == m]
        data.append({"type": "box", "name": METHODS[m], "y": rnd(g["delay"].dropna()), "boxpoints": "all", "jitter": 0.4, "pointpos": 0,
                     "marker": {"color": COLORS[m], "size": 6}, "line": {"color": COLORS[m]}, "fillcolor": "rgba(0,0,0,0)",
                     "hovertemplate": f"{METHODS[m]}<br>demora %{{y}} vuelos<extra></extra>"})
    out["figures"]["line3_delay"] = fig(
        data, height=420, showlegend=False, yaxis={"title": {"text": "Vuelo de alarma − vuelo de inicio real"}, "zeroline": True},
        shapes=[{"type": "line", "x0": 0, "x1": 1, "xref": "paper", "y0": 0, "y1": 0, "line": {"color": "var(--navy)", "dash": "dot"}}],
        annotations=[{"x": 1, "xref": "paper", "xanchor": "right", "y": 1.02, "yref": "paper", "yanchor": "bottom", "showarrow": False,
                      "text": "0 = inicio real de la degradación · arriba = alarma tardía", "font": {"size": 11, "color": "var(--navy)"}}])

    # Motor de ejemplo: el de demora CUSUM más cercana a la mediana
    gc = test[test["method"] == "cusum"].dropna(subset=["delay"])
    uid = gc.iloc[(gc["delay"] - gc["delay"].median()).abs().argsort().iloc[0]]["uid"]
    g = t[t["uid"] == uid].sort_values("cycle")
    rr = r[r["uid"] == uid].set_index("method")
    shapes = [{"type": "line", "x0": rr["onset"].iloc[0], "x1": rr["onset"].iloc[0], "y0": 0, "y1": 1, "yref": "paper",
               "line": {"color": "var(--navy)", "dash": "dot", "width": 1.5}}]
    ann = [{"x": rr["onset"].iloc[0], "y": 1, "yref": "paper", "xanchor": "right", "yanchor": "top", "showarrow": False,
            "text": "inicio real ", "font": {"size": 11, "color": "var(--navy)"}}]
    for m in METHODS:
        d = rr.loc[m, "detect"]
        if pd.notna(d):
            shapes.append({"type": "line", "x0": d, "x1": d, "y0": 0, "y1": 1, "yref": "paper", "line": {"color": COLORS[m], "width": 2}})
    data = [{"type": "scatter", "mode": "lines+markers", "name": "Índice de salud (HI)", "x": rnd(g["cycle"]), "y": rnd(g["HI"], 3),
             "line": {"color": "var(--orange)", "width": 2}, "marker": {"size": 4}, "hovertemplate": "vuelo %{x}<br>HI %{y:.2f}<extra></extra>"},
            {"type": "scatter", "mode": "lines", "name": "Estadístico CUSUM", "x": rnd(g["cycle"]), "y": rnd(cus[uid], 2), "yaxis": "y2",
             "line": {"color": "var(--hud)", "width": 1.6, "dash": "dot"}, "hovertemplate": "vuelo %{x}<br>CUSUM %{y:.1f}<extra></extra>"}]
    for m in METHODS:   # entradas de leyenda para las líneas de alarma
        data.append({"type": "scatter", "mode": "lines", "x": [None], "y": [None], "name": f"alarma {METHODS[m]}", "line": {"color": COLORS[m], "width": 2}})
    out["figures"]["line3_example"] = fig(
        data, height=440, shapes=shapes, annotations=ann, legend={"y": 1.24}, margin={"r": 70},
        xaxis={"title": {"text": f"Número de vuelo · {uid.replace('_u', ' · motor ')} (prueba)"}},
        yaxis={"title": {"text": "Índice de salud"}}, yaxis2={"overlaying": "y", "side": "right", "title": {"text": "CUSUM", "standoff": 12}, "showgrid": False, "automargin": True})

    S = {s["método"]: s for s in summ}
    c, p, th = S[METHODS["cusum"]], S[METHODS["pelt"]], S[METHODS["t48"]]
    out["insights"]["line3_delay"] = (
        f"Con cada umbral calibrado en motores de desarrollo para no superar {pct(lc['target_false_alarm'])} de falsas alarmas, en los {c['motores']} motores de prueba el CUSUM online da la alarma con una demora mediana de <b>{num(c['demora mediana [vuelos]'])} vuelos</b> "
        f"respecto del inicio real, cuando falta una mediana de {num(c['anticipación mediana a la falla [vuelos]'])} vuelos para la falla y sólo ocurrió el "
        f"<b>{num(c['desgaste ocurrido al alarmar [%]'])} %</b> del desgaste total. El umbral sobre T48 sola tarda {num(th['demora mediana [vuelos]'])} vuelos "
        f"({num(th['desgaste ocurrido al alarmar [%]'])} % del desgaste). PELT, que mira la serie completa, marca {num(p['demora mediana [vuelos]'])}. "
        f"Alarmas antes del inicio real en prueba: CUSUM {num(c['alarma antes del inicio real [%]'])} %, T48 {num(th['alarma antes del inicio real [%]'])} %, "
        f"PELT {num(p['alarma antes del inicio real [%]'])} %.")
    out["insights"]["line3_example"] = (
        "El índice de salud resume los 14 residuos en un número: plano mientras el motor está sano y creciente cuando empieza la degradación. "
        "El CUSUM acumula los excesos sobre lo normal y da la alarma (línea de color) cuando la acumulación supera el umbral.")
    out["future_discussion_l3"] = [
        f"<b>Línea 3.</b> Con un índice de salud multivariado y un CUSUM se detecta el inicio del desgaste cuando ocurrió apenas el "
        f"{num(c['desgaste ocurrido al alarmar [%]'])} % de la degradación total, sin usar el RUL. Los próximos pasos que propone la línea son "
        "aprender el índice de salud con Transformers auto-supervisados y Masked Autoencoders sobre la señal segundo a segundo (sin etiquetas) "
        "y combinarlo con change point detection bayesiano online para dar alarmas con una probabilidad asociada."]
    save_json(out, paths.outputs / "line3.json")
    log.info("Línea 3 guardada · CUSUM demora mediana %s vuelos", c["demora mediana [vuelos]"])
    return out
