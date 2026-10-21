"""Intervalos de incertidumbre para el RUL: conformal asimétrico y Adaptive Conformal Inference.

Para cada modelo (CatBoost, TFT) se comparan tres formas de dar un intervalo [L, U]:

1. Cuantiles del modelo:  L = q05, U = q95, sin corrección.
2. CQR asimétrico (Romano et al., 2019), calibrado en motores que ningún modelo vio:
       s_lo = q05 − y   (positivo si el RUL real quedó por debajo de L: SOBREESTIMACIÓN)
       s_hi = y − q95   (positivo si quedó por encima de U: subestimación)
       L = q05 − Q_{1−α_over}(s_lo)     U = q95 + Q_{1−α_under}(s_hi)
   con α_over < α_under: se tolera menos quedar por encima de la vida real.
3. CQR por tramos (Mondrian): lo mismo, pero con una corrección distinta para cada tramo de RUL
   PREDICHO (que se conoce al momento de predecir). Da cobertura pareja cerca y lejos de la falla.
4. ACI de flota (Gibbs & Candès, 2021) sobre el CQR por tramos: los niveles α se ajustan a medida
   que los motores de prueba fallan y se conoce su RUL real (retroalimentación demorada):
       α_{k+1} = α_k + γ (α* − tasa de error del motor k)
   α se acota a [1/(n+1), 1): con α ≤ 0 el intervalo sería infinito. El orden de llegada de los
   motores es desconocido: se promedia sobre varias permutaciones.

La cota inferior L es el "RUL conservador" para planificar inspecciones.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from . import metrics as M
from .config import Paths
from .fmt import num, pct
from .viz import fig, rnd, save_json

log = logging.getLogger(__name__)
MODELS = {"catboost": "CatBoost", "tft": "TFT"}
METHODS = {"raw": "Cuantiles del modelo", "cqr": "CQR asimétrico", "mcqr": "CQR por tramos", "aci": "ACI de flota (por tramos)"}
MODEL_COLOR = {"catboost": "var(--ti)", "tft": "var(--hud)"}
METHOD_COLOR = {"raw": "var(--ti)", "cqr": "var(--hud-2)", "mcqr": "var(--hud)", "aci": "var(--orange)"}


def conformal_q(scores: np.ndarray, alpha: float) -> float:
    """Cuantil conformal con corrección de muestra finita; α ≤ 0 => intervalo infinito."""
    n = len(scores)
    if alpha <= 0:
        return np.inf
    if alpha >= 1:
        return -np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else float(np.sort(scores)[k - 1])


class Calibrator:
    """Puntajes conformales de calibración, globales o por tramo de RUL predicho (Mondrian)."""

    def __init__(self, cal: pd.DataFrame, edges: list[float] | None):
        self.edges = edges
        self.s_lo, self.s_hi = {}, {}
        b = self._bin(cal)
        for k in np.unique(b):
            g = cal[b == k]
            self.s_lo[k] = (g["q05"] - g["RUL"]).to_numpy()
            self.s_hi[k] = (g["RUL"] - g["q95"]).to_numpy()

    def _bin(self, df: pd.DataFrame) -> np.ndarray:
        if not self.edges:
            return np.zeros(len(df), dtype=int)
        b = np.digitize(df["q50"].to_numpy(), self.edges[1:-1])
        return b

    def bounds(self, df: pd.DataFrame, a_over: float, a_under: float):
        lo, hi = np.empty(len(df)), np.empty(len(df))
        b = self._bin(df)
        for k in np.unique(b):
            m = b == k
            key = k if k in self.s_lo else min(self.s_lo, key=lambda j: abs(j - k))   # tramo sin calibración: el más cercano
            sl, sh = self.s_lo[key], self.s_hi[key]
            ao = min(max(a_over, 1 / (len(sl) + 1)), 0.999)
            au = min(max(a_under, 1 / (len(sh) + 1)), 0.999)
            lo[m] = df["q05"].to_numpy()[m] - conformal_q(sl, ao)
            hi[m] = df["q95"].to_numpy()[m] + conformal_q(sh, au)
        return np.maximum(lo, 0), hi


def aci_fleet(test: pd.DataFrame, calib: Calibrator, a_over, a_under, gamma, order, record=False):
    """Recorre los motores de prueba en `order`; actualiza α cuando cada motor termina."""
    lo = np.empty(len(test))
    hi = np.empty(len(test))
    ao, au = a_over, a_under
    trace = []
    pos = test.groupby("uid").indices
    for uid in order:
        ix = pos[uid]
        g = test.iloc[ix]
        L, U = calib.bounds(g, ao, au)
        lo[ix], hi[ix] = L, U
        y = g["RUL"].to_numpy()
        e_o, e_u = np.mean(y < L), np.mean(y > U)
        if record:
            trace.append({"uid": uid, "alpha_over": ao, "alpha_under": au, "miss_over": e_o, "miss_under": e_u})
        ao = float(np.clip(ao + gamma * (a_over - e_o), 0, 0.999))
        au = float(np.clip(au + gamma * (a_under - e_u), 0, 0.999))
    return lo, hi, trace


def _available(paths: Paths) -> dict[str, pd.DataFrame]:
    out = {}
    for k in MODELS:
        p = paths.outputs / f"preds_{k}.parquet"
        if p.exists():
            out[k] = pd.read_parquet(p)
        else:
            log.warning("  Falta preds_%s.parquet: se omite %s (¿se entrenó en Colab? ver README)", k, MODELS[k])
    if not out:
        raise FileNotFoundError("No hay predicciones: corré antes las etapas catboost y/o tft")
    return out


def run(cfg: dict, paths: Paths) -> dict:
    cc, rc, ns = cfg["conformal"], cfg["results"], cfg["nasa_score"]
    preds = _available(paths)
    # Conjunto común: mismos vuelos para todos los modelos
    key = ["uid", "cycle"]
    common = None
    for p in preds.values():
        k = p.loc[p["cycle"] >= rc["eval_min_cycle"], key]
        common = k if common is None else common.merge(k, on=key)
    preds = {m: p.merge(common, on=key).sort_values(key).reset_index(drop=True) for m, p in preds.items()}
    rng = np.random.default_rng(cfg["seed"])
    test_units = sorted(next(iter(preds.values())).query("role == 'test'")["uid"].unique())
    orders = [rng.permutation(test_units) for _ in range(cc["fleet_permutations"])]

    rows, intervals, traces = [], {}, {}
    for m, p in preds.items():
        cal, test = p[p["role"] == "calib"], p[p["role"] == "test"].reset_index(drop=True)
        glob, mond = Calibrator(cal, None), Calibrator(cal, rc["rul_buckets"])
        y, pt = test["RUL"], test["q50"]
        iv = {"raw": (test["q05"].clip(lower=0).to_numpy(), test["q95"].to_numpy())}
        iv["cqr"] = glob.bounds(test, cc["alpha_over"], cc["alpha_under"])
        iv["mcqr"] = mond.bounds(test, cc["alpha_over"], cc["alpha_under"])
        acc = []
        for i, order in enumerate(orders):
            lo, hi, tr = aci_fleet(test, mond, cc["alpha_over"], cc["alpha_under"], cc["aci_gamma"], order, record=i == 0)
            acc.append((lo, hi, M.interval_metrics(y, lo, hi)))
            if i == 0:
                traces[m] = tr
                iv["aci"] = (lo, hi)        # la primera permutación se usa para los gráficos
        aci_spread = pd.DataFrame([a[2] for a in acc])
        for meth, (lo, hi) in iv.items():
            s = M.summary(y, pt, lo, hi, ns)
            if meth == "aci":   # métricas de intervalo promediadas sobre las permutaciones
                for k in ("coverage", "width", "miss_over", "miss_under"):
                    s[k] = float(aci_spread[k].mean())
                s["coverage_sd"] = float(aci_spread["coverage"].std())
            rows.append({"model": m, "method": meth, **s})
        intervals[m] = test.assign(**{f"lo_{k}": v[0] for k, v in iv.items()}, **{f"hi_{k}": v[1] for k, v in iv.items()})
    met = pd.DataFrame(rows)
    out = build_outputs(cfg, paths, preds, intervals, met, traces)
    save_json(out, paths.outputs / "results.json")
    log.info("Resultados guardados: %s", paths.outputs / "results.json")
    return out


# ════════════════════════════════════════════════════════════════════════════
#  Figuras, tablas y lectura de resultados
# ════════════════════════════════════════════════════════════════════════════
def build_outputs(cfg, paths, preds, intervals, met, traces) -> dict:
    cc, rc = cfg["conformal"], cfg["results"]
    target = 1 - cc["alpha"]
    # Modelo que alimenta el sistema: el de menor score NASA en VALIDACIÓN (nunca se mira la prueba para elegir)
    val_score = {m: M.nasa_score(p.loc[p["role"] == "val", "RUL"], p.loc[p["role"] == "val", "q50"], **cfg["nasa_score"]) for m, p in preds.items()}
    main = min(val_score, key=val_score.get)
    out = {"figures": {}, "insights": {}, "tables": {}, "facts": {}, "main_model": main,
           "models": {m: MODELS[m] for m in preds}, "val_nasa": val_score}

    # Resumen de la partición
    p0 = next(iter(preds.values()))
    sp = p0.groupby("role").agg(motores=("uid", "nunique"), vuelos=("uid", "size"))
    role_name = {"train": "Entrenamiento", "val": "Validación (early stopping)", "calib": "Calibración conformal", "test": "Prueba (oficial NASA)"}
    out["split_summary"] = [{"rol": role_name[r], "motores": int(sp.loc[r, "motores"]), "vuelos evaluados": int(sp.loc[r, "vuelos"])}
                            for r in ["train", "val", "calib", "test"] if r in sp.index]

    # Tabla de métricas
    table = []
    for r in met.itertuples():
        table.append({"modelo": MODELS[r.model], "intervalo": METHODS[r.method], "RMSE": round(r.rmse, 2), "MAE": round(r.mae, 2),
                      "score NASA": round(r.nasa, 3), "% sobreestima (mediana)": round(100 * r.over_point, 1),
                      "cobertura [%]": round(100 * r.coverage, 1), "ancho medio [vuelos]": round(r.width, 1),
                      "% real < cota inferior": round(100 * r.miss_over, 1), "% real > cota superior": round(100 * r.miss_under, 1),
                      "score NASA de la cota inferior": round(r.nasa_lower, 3)})
    out["metrics_table"] = table
    out["metrics"] = met.to_dict("records")

    # Métricas por subconjunto (modelo principal, ACI)
    it = intervals[main]
    per_ds = []
    for ds, g in it.groupby("ds"):
        s = M.summary(g["RUL"], g["q50"], g["lo_aci"], g["hi_aci"], cfg["nasa_score"])
        per_ds.append([ds, num(s["rmse"], 2), num(s["nasa"], 3), num(100 * s["coverage"], 1), num(s["width"], 1), num(100 * s["miss_over"], 1)])
    out["tables"]["per_ds"] = {"columns": ["dataset", "RMSE", "score NASA", "cobertura ACI [%]", "ancho [vuelos]", "% real < cota inf."], "rows": per_ds}

    # Predicho vs real (prueba)
    data = []
    for m in preds:
        g = intervals[m]
        data.append({"type": "scattergl", "mode": "markers", "name": MODELS[m], "x": rnd(g["RUL"]), "y": rnd(g["q50"], 2),
                     "visible": True if m == main else "legendonly",
                     "marker": {"size": 4, "opacity": 0.45, "color": MODEL_COLOR[m]},
                     "hovertemplate": f"{MODELS[m]}<br>real %{{x}} · predicho %{{y:.1f}}<extra></extra>"})
    mx = float(p0["RUL"].max())
    data.append({"type": "scatter", "mode": "lines", "x": [0, mx], "y": [0, mx], "name": "predicción perfecta",
                 "line": {"color": "var(--navy)", "dash": "dot", "width": 1.4}, "hoverinfo": "skip"})
    out["figures"]["pred_vs_true"] = fig(data, height=460, xaxis={"title": {"text": "RUL real [vuelos]"}},
                                         yaxis={"title": {"text": "RUL predicho (mediana) [vuelos]"}},
                                         shapes=[{"type": "path", "path": f"M0,0 L{mx},{mx} L0,{mx} Z", "fillcolor": "rgba(214,40,40,0.05)",
                                                  "line": {"width": 0}, "layer": "below"}],
                                         annotations=[{"x": mx * 0.18, "y": mx * 0.9, "showarrow": False, "text": "zona de sobreestimación",
                                                       "font": {"color": "var(--red)", "size": 11}}])
    mm = met.set_index(["model", "method"])
    best_rmse = mm.xs("raw", level="method")["rmse"]
    out["insights"]["pred_vs_true"] = (
        "En motores de prueba nunca vistos: " + "; ".join(f"<b>{MODELS[m]}</b> RMSE {num(best_rmse[m], 2)} vuelos, MAE {num(mm.loc[(m, 'raw'), 'mae'], 2)}, "
                                                           f"score NASA {num(mm.loc[(m, 'raw'), 'nasa'], 3)}" for m in preds) +
        ". Los puntos por encima de la diagonal son sobreestimaciones: el modelo cree que al motor le queda más vida de la real.")

    # Cobertura y ancho por tramo de RUL (modelo principal)
    edges = rc["rul_buckets"]
    buckets = {k: M.by_bucket(it.assign(lo=it[f"lo_{k}"], hi=it[f"hi_{k}"], pred=it["q50"]), edges) for k in METHODS}
    data = []
    for k in METHODS:
        b = buckets[k]
        data.append({"type": "bar", "name": METHODS[k], "x": b["tramo"].tolist(), "y": rnd(b["coverage"] * 100, 2), "marker": {"color": METHOD_COLOR[k]},
                     "customdata": rnd(b["width"], 1), "hovertemplate": f"{METHODS[k]}<br>RUL %{{x}}<br>cobertura %{{y:.1f}} %<br>ancho %{{customdata}} vuelos<extra></extra>"})
    out["figures"]["coverage_by_rul"] = fig(
        data, barmode="group", height=420, bargap=0.25, xaxis={"title": {"text": "Tramo de RUL real [vuelos]"}},
        yaxis={"title": {"text": "Cobertura [%]"}, "range": [0, 105]},
        shapes=[{"type": "line", "x0": 0, "x1": 1, "xref": "paper", "y0": target * 100, "y1": target * 100,
                 "line": {"color": "var(--navy)", "dash": "dot", "width": 1.4}}],
        annotations=[{"x": 1, "xref": "paper", "xanchor": "right", "y": target * 100, "yanchor": "bottom", "showarrow": False,
                      "text": f"objetivo {pct(target)}", "font": {"size": 11, "color": "var(--navy)"}}])
    ba = buckets["aci"]
    out["tables"]["coverage_by_rul"] = {"columns": ["tramo RUL", "vuelos", "RMSE", "cobertura ACI [%]", "ancho [vuelos]", "% real < cota inf."],
                                        "rows": [[r.tramo, r.n, num(r.rmse, 2), num(100 * r.coverage, 1), num(r.width, 1), num(100 * r.miss_over, 1)] for r in ba.itertuples()]}
    raw_cov = buckets["raw"].set_index("tramo")["coverage"]
    worst = raw_cov.idxmin()
    out["insights"]["coverage_by_rul"] = (
        f"Con los cuantiles del modelo ({MODELS[main]}) la cobertura cae a {pct(raw_cov.min())} en el tramo {worst} de RUL. "
        f"La calibración conformal la lleva a {pct(mm.loc[(main, 'aci'), 'coverage'])} en total. El ancho crece con el RUL: "
        f"{num(ba['width'].iloc[0], 1)} vuelos cerca de la falla contra {num(ba['width'].iloc[-1], 1)} con mucha vida por delante. "
        "Es la incertidumbre honesta: lejos de la falla el motor todavía no muestra desgaste.")

    # Traza del ACI
    tr = pd.DataFrame(traces[main])
    tr["k"] = np.arange(1, len(tr) + 1)
    tr["cum_over"] = tr["miss_over"].expanding().mean()
    out["figures"]["aci_trace"] = fig(
        [{"type": "scatter", "mode": "lines+markers", "name": "α sobreestimación (ACI)", "x": rnd(tr["k"]), "y": rnd(tr["alpha_over"] * 100, 3),
          "line": {"color": "var(--orange)", "width": 2.2}, "customdata": tr["uid"].tolist(),
          "hovertemplate": "motor n° %{x} (%{customdata})<br>α = %{y:.2f} %<extra></extra>"},
         {"type": "scatter", "mode": "lines", "name": "tasa acumulada de sobreestimación", "x": rnd(tr["k"]), "y": rnd(tr["cum_over"] * 100, 3),
          "line": {"color": "var(--navy)", "width": 1.6, "dash": "dot"}, "hovertemplate": "tras %{x} motores: %{y:.2f} %<extra></extra>"}],
        height=400, xaxis={"title": {"text": "Motores de prueba que ya fallaron (orden de llegada simulado)"}},
        yaxis={"title": {"text": "%"}},
        shapes=[{"type": "line", "x0": 0, "x1": 1, "xref": "paper", "y0": cc["alpha_over"] * 100, "y1": cc["alpha_over"] * 100,
                 "line": {"color": "var(--red)", "dash": "dash", "width": 1}}])
    out["insights"]["aci_trace"] = (
        f"El nivel de tolerancia a sobreestimar arranca en {pct(cc['alpha_over'])} y se corrige cada vez que un motor falla: si en "
        f"ese motor la cota inferior quedó por encima del RUL real más veces de lo permitido, α baja y los intervalos se ensanchan. "
        f"Sobre {cc['fleet_permutations']} órdenes de llegada distintos, la cobertura del ACI es {pct(mm.loc[(main, 'aci'), 'coverage'], 1)} "
        f"± {num(100 * mm.loc[(main, 'aci'), 'coverage_sd'], 1)} puntos.")

    # Motores de ejemplo: trayectoria del RUL real, predicho e intervalo (selector)
    rng = np.random.default_rng(cfg["seed"])
    units = sorted(it["uid"].unique())
    ex = sorted(rng.choice(units, size=min(rc["n_example_engines"], len(units)), replace=False))
    data, vis = [], []
    for j, u in enumerate(ex):
        g = it[it["uid"] == u]
        v = j == 0
        idx0 = len(data)
        data += [{"type": "scatter", "mode": "lines", "x": rnd(g["cycle"]), "y": rnd(g["hi_aci"], 2), "line": {"width": 0}, "showlegend": False,
                  "hoverinfo": "skip", "visible": v},
                 {"type": "scatter", "mode": "lines", "x": rnd(g["cycle"]), "y": rnd(g["lo_aci"], 2), "fill": "tonexty",
                  "fillcolor": "rgba(0,119,182,0.15)", "line": {"width": 0}, "name": f"intervalo {pct(1 - cc['alpha'])} (ACI)", "visible": v,
                  "showlegend": v, "hovertemplate": "vuelo %{x}<br>cota inferior %{y:.1f}<extra></extra>"},
                 {"type": "scatter", "mode": "lines", "x": rnd(g["cycle"]), "y": rnd(g["q50"], 2), "name": f"predicción {MODELS[main]}",
                  "line": {"color": "var(--hud)", "width": 2.2}, "visible": v, "showlegend": v,
                  "hovertemplate": "vuelo %{x}<br>predicho %{y:.1f}<extra></extra>"},
                 {"type": "scatter", "mode": "lines", "x": rnd(g["cycle"]), "y": rnd(g["RUL"]), "name": "RUL real",
                  "line": {"color": "var(--orange)", "width": 2, "dash": "dash"}, "visible": v, "showlegend": v,
                  "hovertemplate": "vuelo %{x}<br>real %{y}<extra></extra>"}]
        vis.append(list(range(idx0, len(data))))
    buttons = []
    for j, u in enumerate(ex):
        v = [False] * len(data)
        for i in vis[j]:
            v[i] = True
        g = it[it["uid"] == u]
        buttons.append({"label": u.replace("_u", " · motor "), "method": "update",
                        "args": [{"visible": v, "showlegend": [i in vis[j] and data[i].get("name") is not None and i != vis[j][0] for i in range(len(data))]}]})
    out["figures"]["engine_examples"] = fig(
        data, height=460, xaxis={"title": {"text": "Número de vuelo"}}, yaxis={"title": {"text": "RUL [vuelos]"}, "rangemode": "tozero"},
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 1, "xanchor": "right", "y": 1.16, "yanchor": "top"}])
    out["insights"]["engine_examples"] = (
        "Cada motor de prueba vuelo a vuelo: al principio el intervalo es ancho (el motor está sano y no hay señal); cuando "
        "empieza la degradación, la predicción se engancha a la diagonal del RUL real y el intervalo se angosta. La cota "
        "inferior del intervalo es el valor a usar para programar la inspección.")

    # Error por tramo de RUL (todos los modelos)
    data = []
    for m in preds:
        b = M.by_bucket(intervals[m].assign(lo=intervals[m]["lo_aci"], hi=intervals[m]["hi_aci"], pred=intervals[m]["q50"]), edges)
        data.append({"type": "bar", "name": MODELS[m], "x": b["tramo"].tolist(), "y": rnd(b["rmse"], 2), "marker": {"color": MODEL_COLOR[m]},
                     "customdata": rnd(b["over_point"] * 100, 1),
                     "hovertemplate": f"{MODELS[m]}<br>RUL %{{x}}<br>RMSE %{{y:.2f}}<br>sobreestima %{{customdata}} %<extra></extra>"})
    out["figures"]["error_by_rul"] = fig(data, barmode="group", height=380, bargap=0.25,
                                         xaxis={"title": {"text": "Tramo de RUL real [vuelos]"}}, yaxis={"title": {"text": "RMSE [vuelos]"}})
    bm = M.by_bucket(it.assign(lo=it["lo_aci"], hi=it["hi_aci"], pred=it["q50"]), edges)
    out["insights"]["error_by_rul"] = (
        f"El error depende mucho de cuánto falta: RMSE {num(bm['rmse'].iloc[0], 1)} vuelos en el tramo {bm['tramo'].iloc[0]} contra "
        f"{num(bm['rmse'].iloc[-1], 1)} en {bm['tramo'].iloc[-1]} ({MODELS[main]}). Cerca de la falla, que es cuando más importa, "
        "las predicciones son precisas.")

    # Importancia de variables
    data = []
    for m in preds:
        mf = paths.outputs / f"model_{m}.json"
        if not mf.exists():
            continue
        imp = pd.Series(json.loads(mf.read_text(encoding="utf-8"))["importance"])
        imp = (imp / imp.sum() * 100).sort_values(ascending=False).head(15)[::-1]
        data.append({"type": "bar", "orientation": "h", "name": MODELS[m], "y": imp.index.tolist(), "x": rnd(imp.values, 2),
                     "marker": {"color": MODEL_COLOR[m]}, "visible": True if m == main else "legendonly",
                     "hovertemplate": f"{MODELS[m]} · %{{y}}<br>%{{x:.1f}} % de la importancia<extra></extra>"})
    out["figures"]["feature_importance"] = fig(data, height=520, margin={"l": 140}, bargap=0.2,
                                               xaxis={"title": {"text": "Importancia relativa [%]"}})
    tops = {}
    for d in data:
        tops[d["name"]] = d["y"][::-1][:3]
    out["insights"]["feature_importance"] = (
        "Variables más usadas: " + "; ".join(f"<b>{k}</b>: {', '.join(v)}" for k, v in tops.items()) +
        ". Las variables suavizadas (medias y pendientes móviles respecto de la línea base del motor) dominan, como "
        "anticipaba el EDA. <i>d_</i> = residuo menos línea base · <i>_m10</i> = media de 10 vuelos · <i>_slope</i> = pendiente.")

    # Lectura general
    r_main = mm.loc[(main, "aci")]
    r_cqr = mm.loc[(main, "cqr")]
    r_raw = mm.loc[(main, "raw")]
    other = [m for m in preds if m != main]
    comp = ""
    if other:
        o = other[0]
        better = "mejor" if mm.loc[(main, "raw"), "rmse"] < mm.loc[(o, "raw"), "rmse"] else "peor"
        comp = (f"{MODELS[main]} tiene un RMSE {better} que {MODELS[o]} ({num(r_raw['rmse'], 2)} contra {num(mm.loc[(o, 'raw'), 'rmse'], 2)} vuelos) "
                f"y un score NASA de {num(r_raw['nasa'], 3)} contra {num(mm.loc[(o, 'raw'), 'nasa'], 3)}. ")
    n_cal = int(p0.loc[p0["role"] == "calib", "uid"].nunique())
    below = r_main["coverage"] < target
    sel = (f"Para el sistema se usa <b>{MODELS[main]}</b>, elegido por su score NASA en los motores de validación "
           + "(" + ", ".join(f"{MODELS[m]} {num(v, 3)}" for m, v in val_score.items()) + "), sin mirar la prueba. ") if len(preds) > 1 else ""
    out["discussion"] = [
        comp + sel + f"Los cuantiles que da el modelo por sí solo cubren el {pct(r_raw['coverage'], 1)} de los casos (objetivo {pct(target)}) "
               + ("y no están calibrados: la incertidumbre del modelo es optimista." if r_raw["coverage"] < target else
                  "pero sin ninguna garantía: la cobertura depende de qué tan bien aprendió el modelo, no de una calibración."),
        f"Con CQR asimétrico global la cobertura es {pct(r_cqr['coverage'], 1)} y el RUL real queda por debajo de la cota inferior en el "
        f"{pct(r_cqr['miss_over'], 1)} de los vuelos (tolerancia fijada: {pct(cc['alpha_over'])}). Calibrando por tramos de RUL predicho, "
        f"{pct(mm.loc[(main, 'mcqr'), 'coverage'], 1)}. El ACI de flota, que además se reajusta cada vez que un motor falla, logra "
        f"{pct(r_main['coverage'], 1)} de cobertura con un ancho medio de {num(r_main['width'], 1)} vuelos y deja la sobreestimación en "
        f"{pct(r_main['miss_over'], 1)}.",
        (f"La cobertura queda {num(100 * (target - r_main['coverage']), 1)} puntos por debajo del objetivo. La garantía conformal supone datos "
         f"intercambiables, pero acá se calibra con sólo {n_cal} motores y los vuelos de un mismo motor están fuertemente correlacionados: el "
         "tamaño de muestra efectivo es chico y los motores de prueba no se comportan exactamente como los de calibración. El ACI corrige "
         "parte de esa diferencia sobre la marcha, sobre todo en la cola peligrosa (sobreestimación)." if below else
         f"La cobertura alcanza el objetivo aun con sólo {n_cal} motores de calibración."),
        f"La mediana sobreestima en el {pct(r_main['over_point'])} de los vuelos, pero la cota inferior sólo en el {pct(r_main['miss_over'], 1)}: "
        "planificar con la cota inferior convierte un riesgo frecuente en uno controlado, a cambio de inspeccionar algunos vuelos antes de lo necesario.",
    ]
    out["facts"] = {"main_model": MODELS[main], "rmse": num(r_raw["rmse"], 2), "mae": num(r_raw["mae"], 2), "nasa": num(r_raw["nasa"], 3),
                    "coverage_raw": pct(r_raw["coverage"], 1), "coverage_cqr": pct(r_cqr["coverage"], 1), "coverage_aci": pct(r_main["coverage"], 1),
                    "width_aci": num(r_main["width"], 1), "over_point": pct(r_main["over_point"]), "miss_over_aci": pct(r_main["miss_over"], 1),
                    "target": pct(target), "alpha_over": pct(cc["alpha_over"]), "alpha_under": pct(cc["alpha_under"])}
    out["model_meta"] = {m: {k: v for k, v in json.loads((paths.outputs / f"model_{m}.json").read_text(encoding="utf-8")).items() if k != "importance"}
                         for m in preds if (paths.outputs / f"model_{m}.json").exists()}
    return out
