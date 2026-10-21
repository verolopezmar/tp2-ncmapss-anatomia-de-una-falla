"""Análisis exploratorio (EDA) de N-CMAPSS.

Calcula, a partir de los datos procesados, todas las figuras, tablas, cifras y lecturas
del EDA y las guarda en outputs/<modo>/eda.json. El notebook 01_eda.ipynb y el informe
HTML leen ese archivo: ningún número está escrito a mano.

Secciones (mismo recorrido que compartir.html, ampliado):
  1 dataset    estructura, volumen, calidad, faltantes y verificaciones de consistencia
  2 vuelo      perfil de un vuelo, envolvente, clases de vuelo corto / mediano / largo
  3 contexto   cuánto del sensor explica el vuelo, correlaciones, residuos
  4 desgaste   qué componente se degrada en cada escenario y cómo evoluciona
  5 señales    qué sensores anticipan la falla, evolución en la vida, atípicos, PCA
  6 vida útil  duración de los motores y distribución del RUL
  7 decisiones conclusiones del EDA que justifican el modelado
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from . import metadata as md
from .config import Paths
from .fmt import millions, num, pct, signed
from .splits import load_splits
from .viz import (COMP_COLOR, DIVERGING, FC_COLOR, HEAT, RUL_SCALE, SPLIT_COLOR, axis_ref, fig, rnd,
                  save_json, subplot_axes)

log = logging.getLogger(__name__)
SPLIT_NAME = {"dev": "Desarrollo", "test": "Prueba"}


# ════════════════════════════════════════════════════════════════════════════
#  Carga y preparación
# ════════════════════════════════════════════════════════════════════════════
class Ctx:
    def __init__(self, cfg: dict, paths: Paths):
        self.cfg, self.paths = cfg, paths
        self.mode = paths.outputs.name
        self.e = cfg["eda"]
        self.names = json.loads((paths.processed / "variable_names.json").read_text())
        self.W, self.XS, self.XV, self.T = (self.names[k] for k in ("W", "X_s", "X_v", "T"))
        self.files = pd.read_parquet(paths.processed / "files.parquet")
        self.colstats = pd.read_parquet(paths.processed / "colstats.parquet")
        self.sample = pd.read_parquet(paths.processed / "seconds_sample.parquet")
        ex = paths.processed / "example_flight.parquet"
        self.example = pd.read_parquet(ex) if ex.exists() else None
        self.health = json.loads((paths.outputs / "health_model.json").read_text())
        self.frac = (cfg["sample"]["seconds_sample_frac"] if self.mode == "sample"
                     else cfg["data"]["seconds_sample_frac"])

        f = pd.read_parquet(paths.processed / "flights.parquet")
        f = f.merge(load_splits(paths)[["uid", "role"]], on="uid")
        f["eol"] = f.groupby("uid")["cycle"].transform("max")
        f["hours"] = f["n_sec"] / 3600
        f["life"] = f["cycle"] / f["eol"]
        # residuo estandarizado: en "desvíos de un motor sano" (referencia: vuelos sanos de train)
        ref = f[(f["role"] == "train") & (f["hs"] == 1)]
        for s in self.XS + self.XV:
            c = f"res_{s}_mean"
            f[f"z_{s}"] = (f[c] - ref[c].mean()) / ref[c].std()
        self.f = f.sort_values(["ds", "unit", "cycle"]).reset_index(drop=True)
        self.units = f.drop_duplicates("uid").set_index("uid")
        self.ds_list = sorted(self.files["ds"])
        self.ds_ok = sorted(self.f["ds"].unique())
        ex_u = cfg["sample"]["example_unit"] if self.mode == "sample" else self.e["example_unit"]
        uid = f"{ex_u['dataset']}_u{ex_u['unit']:02d}"
        self.example_uid = uid if uid in self.units.index else self.units.index[0]


def _corr_rho(a, b) -> float:
    return float(spearmanr(a, b).statistic)


# ════════════════════════════════════════════════════════════════════════════
#  1 · Dataset
# ════════════════════════════════════════════════════════════════════════════
def section_dataset(c: Ctx, out: dict) -> None:
    f, files = c.f, c.files
    rows = f.groupby(["ds", "split"])["n_sec"].sum().unstack(fill_value=0)
    bad = files[~files["ok"]]["ds"].tolist()
    order = c.ds_list
    data = [{"type": "bar", "orientation": "h", "name": SPLIT_NAME[sp], "y": order,
             "x": rnd([rows.loc[d, sp] / 1e6 if d in rows.index else 0 for d in order], 3),
             "marker": {"color": SPLIT_COLOR[sp]},
             "hovertemplate": f"%{{y}} · {SPLIT_NAME[sp].lower()}<br>%{{x:.2f}} millones de registros<extra></extra>"}
            for sp in ("dev", "test")]
    ann = [{"x": 0.05, "y": d, "xanchor": "left", "showarrow": False, "font": {"color": "var(--red)", "size": 11},
            "text": f"{d}: el archivo viene dañado y no se puede abrir"} for d in bad]
    out["figures"]["sizes"] = fig(data, barmode="stack", height=360, bargap=0.35, annotations=ann,
                                  legend={"traceorder": "normal"}, yaxis={"autorange": "reversed"},
                                  xaxis={"title": {"text": "Millones de registros (1 registro = 1 segundo de vuelo)"}})
    tot = f["n_sec"].sum()
    dev_share = rows["dev"].sum() / tot
    biggest = rows.sum(1).idxmax()
    out["insights"]["sizes"] = (
        f"<b>{num(tot / 1e6, 1)} millones de segundos</b> de vuelo (≈ {num(tot / 3600)} h); el "
        f"{pct(dev_share)} es de desarrollo y el subconjunto más grande es <b>{biggest}</b>."
        + (f" <b>{', '.join(bad)}</b> viene truncado en Kaggle y quedó afuera." if bad else ""))
    g = f.groupby(["ds", "split"])
    t = pd.DataFrame({"motores": g["uid"].nunique(), "vuelos": g.size(), "registros_1Hz": g["n_sec"].sum()}).reset_index()
    out["tables"]["sizes"] = {"columns": ["dataset", "partición", "motores", "vuelos", "registros (1 Hz)"],
                              "rows": [[r.ds, r.split, int(r.motores), num(r.vuelos), num(r.registros_1Hz)] for r in t.itertuples()]}

    # Calidad: variación relativa de cada variable (desvío / |media|), calculada con TODOS los segundos
    cs = c.colstats.groupby("variable")[["n", "nan", "inf", "sum", "sq", "min", "max"]].agg(
        {"n": "sum", "nan": "sum", "inf": "sum", "sum": "sum", "sq": "sum", "min": "min", "max": "max"})
    mean = cs["sum"] / cs["n"]
    std = np.sqrt(np.maximum(cs["sq"] / cs["n"] - mean ** 2, 0))
    cv = (std / mean.abs() * 100).drop(c.T + ["RUL"], errors="ignore")
    order_cv = cv.sort_values().index.tolist()
    groups = [("Condición de vuelo", c.W, "var(--ti)"), ("Sensor físico", c.XS, "var(--hud)"),
              ("Sensor virtual", c.XV, "var(--c-hpc)")]
    out["figures"]["quality"] = fig(
        [{"type": "bar", "name": n, "x": [v for v in order_cv if v in cols], "y": rnd([cv[v] for v in order_cv if v in cols], 3),
          "customdata": [md.DESCRIPTIONS.get(v, v) for v in order_cv if v in cols], "marker": {"color": col},
          "hovertemplate": "%{x} · %{customdata}<br>variación relativa: %{y:.2f} %<extra></extra>"} for n, cols, col in groups],
        height=360, bargap=0.3, yaxis={"type": "log", "title": {"text": "Variación relativa [%] (escala log)"}},
        xaxis={"categoryorder": "array", "categoryarray": order_cv, "tickangle": -45})
    low3 = order_cv[:3]
    out["insights"]["quality"] = (
        f"Las variables que menos varían (<b>{', '.join(low3)}</b>) no son inútiles: T48 varía sólo un "
        f"{num(cv['T48'], 1)} % y, como se ve más adelante, es uno de los mejores indicadores del desgaste. "
        f"La que más varía es <b>{order_cv[-1]}</b> ({num(cv[order_cv[-1]], 0)} %).")

    # Verificaciones de consistencia (faltantes, duplicados, reglas del dataset)
    fs = f.sort_values(["uid", "cycle"])
    gu = fs.groupby("uid")
    n_nan, n_inf = int(cs["nan"].sum()), int(cs["inf"].sum())
    consecutive = gu["cycle"].apply(lambda x: bool((np.diff(x.to_numpy()) == 1).all() and x.iloc[0] == 1)).all()
    rul_ok = float((fs["RUL"] == fs["eol"] - fs["cycle"]).mean())
    fc_const = bool((gu["Fc"].nunique() == 1).all())
    hs_mono = bool(gu["hs"].apply(lambda h: (np.diff(h.to_numpy()) <= 0).all()).all())
    hs_const = int(f["hs_mixed"].sum())
    phys = cs.loc[[v for v in c.XS + c.XV if v.startswith(("T", "P", "N", "W"))], "min"]
    neg = phys[phys <= 0].index.tolist()
    checks = [
        {"name": "Archivos íntegros", "ok": not bad,
         "detail": f"{int(files['ok'].sum())} de {len(files)} archivos se abren completos" + (f"; {', '.join(bad)} está truncado." if bad else ".")},
        {"name": "Valores faltantes", "ok": n_nan == 0, "detail": f"{num(n_nan)} nulos en {num(cs['n'].sum())} celdas revisadas (todas las variables, todos los segundos)."},
        {"name": "Valores infinitos", "ok": n_inf == 0, "detail": f"{num(n_inf)} valores infinitos."},
        {"name": "Vuelos sin saltos ni repetidos", "ok": bool(consecutive), "detail": "Cada motor tiene sus vuelos numerados 1, 2, 3… sin huecos ni duplicados."},
        {"name": "RUL coherente", "ok": rul_ok == 1.0, "detail": f"RUL = vida total − n° de vuelo en el {pct(rul_ok, 1)} de los vuelos."},
        {"name": "Clase de vuelo fija por motor", "ok": fc_const, "detail": "Cada motor vuela siempre la misma clase (corta, mediana o larga)."},
        {"name": "Estado de salud consistente", "ok": hs_mono and hs_const == 0,
         "detail": "hs es constante dentro de cada vuelo y, una vez degradado, el motor nunca vuelve a 'sano'."},
        {"name": "Rangos físicos", "ok": not neg,
         "detail": "Todas las temperaturas, presiones, giros y flujos son positivos." if not neg else f"Valores no positivos en {', '.join(neg)}."},
    ]
    out["checks"] = checks
    out["tables"]["missing"] = {"columns": ["variable", "registros", "nulos", "infinitos", "mínimo", "máximo"],
                                "rows": [[v, num(cs.loc[v, 'n']), int(cs.loc[v, 'nan']), int(cs.loc[v, 'inf']),
                                          num(cs.loc[v, 'min'], 3), num(cs.loc[v, 'max'], 3)] for v in c.W + c.XS + c.XV + c.T]}

    # Jerarquía, bloques de variables, registro de ejemplo
    per_ds_units = f.groupby("ds")["uid"].nunique()
    eol = c.units["eol"]
    out["hierarchy"] = [
        {"n": "1", "name": "Flota", "desc": "Todos los motores simulados por NASA"},
        {"n": str(len(c.ds_ok)), "name": "Subconjuntos", "desc": f"Escenarios de falla ({c.ds_ok[0]} … {c.ds_ok[-1]})"},
        {"n": num(f['uid'].nunique()), "name": "Motores", "desc": f"Entre {per_ds_units.min()} y {per_ds_units.max()} por subconjunto"},
        {"n": num(len(f)), "name": "Vuelos", "desc": f"Entre {eol.min()} y {eol.max()} por motor"},
        {"n": millions(tot), "name": "Segundos", "desc": f"Un vuelo dura {num(f['hours'].mean(), 1)} h en promedio"},
    ]
    blocks = {"W": c.W, "X_s": c.XS, "X_v": c.XV, "T": c.T, "Y": ["RUL"], "A": ["unit", "cycle", "Fc", "hs"]}
    out["var_groups"] = [{"key": k, "name": md.BLOCKS[k][0], "cols": v, "desc": md.BLOCKS[k][1]} for k, v in blocks.items()]
    out["descriptions"] = md.DESCRIPTIONS
    if c.example is not None:
        ex = c.example
        cruise = ex[ex["alt"] >= 0.95 * ex["alt"].max()]
        row = cruise.iloc[len(cruise) // 2]
        out["example_label"] = f"{ex['ds'].iloc[0]} · motor {int(row['unit'])} · vuelo {int(row['cycle'])} · un segundo en crucero"
        blk_of = {v: k for k, vs in blocks.items() for v in vs}
        out["example_row"] = [{"block": blk_of[v], "var": v, "desc": md.DESCRIPTIONS.get(v, ""),
                               "value": num(row[v], 0 if abs(row[v]) >= 100 else 4 if abs(row[v]) < 1 else 2)}
                              for v in blocks["A"] + c.W + ["RUL"] + c.XS + c.XV + c.T]


# ════════════════════════════════════════════════════════════════════════════
#  2 · Vuelo
# ════════════════════════════════════════════════════════════════════════════
def section_flight(c: Ctx, out: dict) -> None:
    f, s, e = c.f, c.sample, c.e
    step = e["plot_downsample_s"]
    if c.example is not None:
        ex = c.example.sort_values("t_sec").iloc[::step]
        mins = ex["t_sec"] / 60
        vars5 = ["alt", "Mach", "TRA", "T2", "T48"]
        lay = subplot_axes(5, vgap=0.04)
        n = lay.pop("_n")
        colors = ["var(--hud)", "var(--hud)", "var(--hud)", "var(--ti)", "var(--orange)"]
        data = []
        for i, v in enumerate(vars5, start=1):
            xr, yr = axis_ref(i)
            data.append({"type": "scatter", "mode": "lines", "x": rnd(mins, 2), "y": rnd(ex[v], 4), "xaxis": xr, "yaxis": yr,
                         "name": md.SHORT_LABEL[v], "showlegend": False, "line": {"color": colors[i - 1], "width": 1.6},
                         "hovertemplate": f"minuto %{{x:.0f}}<br>{md.SHORT_LABEL[v]}: %{{y:.2f}}<extra></extra>"})
            lay[f"yaxis{'' if i == 1 else i}"]["title"] = {"text": v}
        lay[f"xaxis{n}"]["title"] = {"text": "Minutos desde el inicio del registro"}
        out["figures"]["profile"] = fig(data, height=620, **lay)
        dur = c.example["t_sec"].max() / 60
        cv_tra = ex["TRA"].std() / ex["TRA"].mean()
        cv_alt = ex["alt"].std() / ex["alt"].mean()
        r_t48 = np.corrcoef(ex["TRA"], ex["T48"])[0, 1]
        out["insights"]["profile"] = (
            f"Vuelo de <b>{num(dur)} minutos</b>. T48 sigue a la potencia (r = {num(r_t48, 2)} con TRA): lo que más "
            f"mueve a los sensores es cómo se vuela. Los registros no incluyen pista ni rodaje: empiezan, en la mediana "
            f"de todos los vuelos, a {num(f['alt_min'].median())} ft.")
        # datos para el HUD de la portada y la trayectoria animada
        hud = c.example.sort_values("t_sec").iloc[::max(1, step // 2)]
        out["hero"] = {"alt": rnd(hud["alt"], 0), "mach": rnd(hud["Mach"], 3), "tra": rnd(hud["TRA"], 1),
                       "t48": rnd(hud["T48"], 0), "t": rnd(hud["t_sec"], 0)}
        top = ex["alt"] >= 0.95 * ex["alt"].max()
        t_top = ex.loc[top, "t_sec"]
        t0, t1, tend = t_top.min(), t_top.max(), ex["t_sec"].max()
        out["flight_path"] = {"label": out.get("example_label", "").split(" · un")[0], "t": rnd(ex["t_sec"], 0),
                              "alt": rnd(ex["alt"], 0),
                              "phases": [{"name": "ASCENSO", "t": t0 / 2}, {"name": "CRUCERO", "t": (t0 + t1) / 2},
                                         {"name": "DESCENSO", "t": (t1 + tend) / 2}]}

    # Envolvente: tiempo en cada celda altitud × Mach (muestra escalada a segundos totales)
    nb = e["envelope_bins"]
    H, xe, ye = np.histogram2d(s["Mach"], s["alt"], bins=nb)
    z = np.log10(np.where(H > 0, H, np.nan) / c.frac)
    out["figures"]["envelope"] = fig(
        [{"type": "heatmap", "x": rnd((xe[:-1] + xe[1:]) / 2, 3), "y": rnd((ye[:-1] + ye[1:]) / 2, 0), "z": np.round(z.T, 2).tolist(),
          "colorscale": [[0, "var(--surface-2)"], [1, "var(--hud)"]], "colorbar": {"thickness": 10, "title": {"text": "log₁₀ s"}},
          "hovertemplate": "Mach %{x:.2f}<br>%{y:,.0f} ft<br>log₁₀(segundos) = %{z:.2f}<extra></extra>"}],
        height=430, xaxis={"title": {"text": "Velocidad [Mach]"}}, yaxis={"title": {"text": "Altitud [ft]"}})
    hi = (s["alt"] > e["high_alt_ft"]).mean()
    r_am = np.corrcoef(s["alt"], s["Mach"])[0, 1]
    out["insights"]["envelope"] = (
        f"Altitud y velocidad crecen juntas (r = {num(r_am, 2)}). Sólo el <b>{pct(hi)}</b> del tiempo transcurre por "
        f"encima de {num(e['high_alt_ft'])} ft, y las condiciones son continuas: no hay unos pocos regímenes fijos que "
        f"permitan separar los datos fácilmente (a diferencia del C-MAPSS original).")

    # Clases de vuelo
    names = c.cfg["flight_classes"]
    fc = f.groupby("Fc").agg(vuelos=("uid", "size"), motores=("uid", "nunique"), h=("hours", "mean"),
                             alt=("alt_max", "mean"), mach=("Mach_max", "mean"))
    data = [{"type": "violin", "name": f"Clase {k} · {names[k]}", "y": rnd(f.loc[f["Fc"] == k, "hours"], 3),
             "box": {"visible": True}, "meanline": {"visible": True}, "points": False, "spanmode": "hard", "opacity": 0.55,
             "line": {"color": FC_COLOR[k]}, "fillcolor": FC_COLOR[k],
             "hovertemplate": f"Clase {k} · {names[k]}<br>%{{y:.2f}} h<extra></extra>"} for k in sorted(f["Fc"].unique())]
    out["figures"]["flight_class"] = fig(data, height=430, showlegend=False, yaxis={"title": {"text": "Duración del vuelo [h]"}})
    ks = fc.index.tolist()
    out["insights"]["flight_class"] = (
        "Duración media de " + ", ".join(f"<b>{num(fc.loc[k, 'h'], 1)} h</b> ({names[k]})" for k in ks) +
        f". Los vuelos largos además suben más ({num(fc['alt'].iloc[-1])} ft contra {num(fc['alt'].iloc[0])} ft de altitud máxima media). "
        "Cada motor hace siempre la misma clase de vuelo, así que la clase es una propiedad del motor y no del vuelo.")
    out["tables"]["flight_class"] = {"columns": ["clase", "motores", "vuelos", "duración media [h]", "altitud máx. media [ft]", "Mach máx. medio"],
                                     "rows": [[f"{k} · {names[k]}", int(r.motores), num(r.vuelos), num(r.h, 2), num(r.alt), num(r.mach, 3)]
                                              for k, r in fc.iterrows()]}

    # Forma típica de un vuelo de cada clase (tiempo normalizado 0–100 %)
    ss = s.merge(f[["ds", "split", "unit", "cycle", "n_sec"]], on=["ds", "split", "unit", "cycle"])
    ss["tau"] = ss["t_sec"] / ss["n_sec"]
    bins = np.linspace(0, 1, 51)
    ss["tb"] = pd.cut(ss["tau"], bins, labels=(bins[:-1] + bins[1:]) / 2, include_lowest=True).astype(float)
    lay = subplot_axes(2, vgap=0.08)
    lay.pop("_n")
    data = []
    for i, v in enumerate(["alt", "TRA"], start=1):
        xr, yr = axis_ref(i)
        for k in ks:
            q = ss[ss["Fc"] == k].groupby("tb")[v].median()
            data.append({"type": "scatter", "mode": "lines", "x": rnd(q.index * 100, 1), "y": rnd(q.values, 2),
                         "xaxis": xr, "yaxis": yr, "name": f"Clase {k} · {names[k]}", "legendgroup": str(k),
                         "showlegend": i == 1, "line": {"color": FC_COLOR[k], "width": 2.2},
                         "hovertemplate": f"{names[k]} · %{{x:.0f}} % del vuelo<br>{v}: %{{y:,.1f}}<extra></extra>"})
        lay[f"yaxis{'' if i == 1 else i}"]["title"] = {"text": md.SHORT_LABEL[v]}
    lay["xaxis2"]["title"] = {"text": "Avance del vuelo [%]"}
    out["figures"]["flight_shapes"] = fig(data, height=520, **lay)
    ss = ss.merge(f[["ds", "split", "unit", "cycle", "alt_max"]], on=["ds", "split", "unit", "cycle"])
    cruise = (ss["alt"] >= 0.9 * ss["alt_max"]).groupby(ss["Fc"]).mean()
    out["insights"]["flight_shapes"] = (
        "Las tres clases siguen la secuencia ascenso → crucero → descenso, pero con regímenes distintos: la fracción del "
        "vuelo cerca de la altitud máxima (≥ 90 %) es " + ", ".join(f"<b>{pct(cruise[k])}</b> en los {names[k]}s" for k in ks) +
        ". Esa diferencia de régimen operativo es la que la Línea 2 propone separar del desgaste.")


# ════════════════════════════════════════════════════════════════════════════
#  3 · Contexto: vuelo vs. desgaste
# ════════════════════════════════════════════════════════════════════════════
def section_context(c: Ctx, out: dict) -> None:
    f, s, e = c.f, c.sample, c.e
    pts = s[s["split"] == "dev"].sample(min(e["scatter_points"], len(s)), random_state=c.cfg["seed"])
    out["figures"]["context"] = fig(
        [{"type": "scattergl", "mode": "markers", "x": rnd(pts["TRA"], 2), "y": rnd(pts["T48"], 1),
          "marker": {"color": rnd(pts["alt"], 0), "size": 4, "opacity": 0.65, "colorscale": [[0, "var(--orange)"], [0.5, "var(--amber)"], [1, "var(--hud)"]],
                     "colorbar": {"thickness": 10, "title": {"text": "altitud [ft]"}}},
          "hovertemplate": "TRA %{x:.1f} %<br>T48 %{y:.0f} °R<br>altitud %{marker.color:,.0f} ft<extra></extra>"}],
        height=430, xaxis={"title": {"text": "Palanca de potencia, TRA [%]"}}, yaxis={"title": {"text": "Temperatura de turbina, T48 [°R]"}})
    r = np.corrcoef(s["TRA"], s["T48"])[0, 1]
    X = np.column_stack([np.ones(len(s)), s["TRA"], s["alt"]])
    beta = np.linalg.lstsq(X, s["T48"].to_numpy(), rcond=None)[0]
    out["insights"]["context"] = (
        f"T48 depende sobre todo de la potencia (<b>r = {num(r, 2)}</b>). A igual potencia, cada 10.000 ft de altura la "
        f"{'bajan' if beta[2] < 0 else 'suben'} unos <b>{num(abs(beta[2]) * 1e4)} °R</b>. Dos vuelos distintos no se pueden comparar directamente.")

    # Variación que NO explica el vuelo (modelo de motor sano, en motores de validación)
    r2 = pd.Series(c.health["r2_val"])
    une = ((1 - r2).clip(lower=1e-6) * 100).sort_values()
    data = []
    for n, cols, col in [("Sensor físico", c.XS, "var(--hud)"), ("Sensor virtual", c.XV, "var(--c-hpc)")]:
        idx = [v for v in une.index if v in cols]
        data.append({"type": "bar", "orientation": "h", "name": n, "y": idx, "x": rnd(une[idx], 4),
                     "customdata": [[round(r2[v] * 100, 3), md.DESCRIPTIONS.get(v, "")] for v in idx], "marker": {"color": col},
                     "hovertemplate": "%{y} · %{customdata[1]}<br>explicado por el vuelo: %{customdata[0]:.2f} %<br>sin explicar: %{x:.3f} %<extra></extra>"})
    out["figures"]["variance_explained"] = fig(
        data, height=580, bargap=0.25, yaxis={"categoryorder": "array", "categoryarray": une.index.tolist()},
        xaxis={"type": "log", "title": {"text": "Parte del sensor que NO explican las condiciones de vuelo [%] (escala log)"}})
    med_r2 = float(r2[c.XS].median())
    out["facts"]["med_r2"] = f"{num(med_r2 * 100, 3)} %"
    out["insights"]["variance_explained"] = (
        f"Con sólo las cuatro condiciones de vuelo se explica el <b>{num(med_r2 * 100, 3)} %</b> de un sensor físico típico "
        f"(mediana, en motores sanos que el modelo no vio). En T48 queda sin explicar un <b>{num(une['T48'], 3)} %</b>: "
        f"ahí adentro está el desgaste. Lo que más queda sin explicar son <b>{', '.join(une.index[-3:][::-1])}</b>.")

    # Correlaciones entre todas las variables (agrupadas por similitud)
    vars_ = c.W + c.XS + c.XV
    cm = s[vars_].corr().to_numpy()
    dist = squareform(1 - np.abs(cm), checks=False).clip(min=0)
    order = leaves_list(linkage(dist, "average"))
    names = [vars_[i] for i in order]
    cmo = cm[np.ix_(order, order)]
    out["figures"]["corr"] = fig(
        [{"type": "heatmap", "x": names, "y": names, "z": np.round(cmo, 3).tolist(), "zmin": -1, "zmax": 1, "xgap": 1, "ygap": 1,
          "colorscale": DIVERGING, "colorbar": {"thickness": 10, "title": {"text": "r"}},
          "hovertemplate": "%{y} y %{x}<br>correlación r = %{z:.2f}<extra></extra>"}],
        height=660, xaxis={"tickangle": -60, "showgrid": False}, yaxis={"autorange": "reversed", "showgrid": False})
    iu = np.triu_indices(len(vars_), 1)
    pairs = pd.DataFrame({"a": np.array(vars_)[iu[0]], "b": np.array(vars_)[iu[1]], "r": cm[iu]})
    strong = (pairs["r"].abs() > e["corr_strong"]).sum()
    top = pairs.reindex(pairs["r"].abs().sort_values(ascending=False).index)
    ex_pairs, used = [], set()
    for p in top.itertuples():
        if p.a in used or p.b in used:
            continue
        ex_pairs.append(f"{p.a}–{p.b} r = {num(p.r, 2)}")
        used |= {p.a, p.b}
        if len(ex_pairs) == 3:
            break
    out["insights"]["corr"] = (
        f"<b>{strong} de {len(pairs)}</b> pares tienen |r| > {num(e['corr_strong'], 2)}. Se forman bloques de variables "
        f"casi redundantes (por ejemplo {', '.join(ex_pairs)}). Hay mucha información repetida: conviene trabajar con pocos "
        "indicadores bien elegidos o con modelos que toleren la colinealidad.")
    out["facts"]["corr_strong"] = f"{strong} de {len(pairs)}"

    # Antes y después: el mismo motor con y sin efecto del vuelo
    u = f[f["uid"] == c.example_uid]
    onset = u.loc[u["hs"] == 0, "cycle"].min()
    lay = subplot_axes(2, vgap=0.12)
    lay.pop("_n")
    data = [{"type": "scatter", "mode": "lines+markers", "x": rnd(u["cycle"]), "y": rnd(u["T48_mean"], 2), "name": "Medido",
             "xaxis": "x", "yaxis": "y", "marker": {"size": 5}, "line": {"color": "var(--ti)", "width": 1.5},
             "hovertemplate": "vuelo %{x}<br>%{y:.1f} °R<extra></extra>"},
            {"type": "scatter", "mode": "lines+markers", "x": rnd(u["cycle"]), "y": rnd(u["z_T48"], 3), "name": "Residuo",
             "xaxis": "x2", "yaxis": "y2", "marker": {"size": 5}, "line": {"color": "var(--orange)", "width": 2},
             "hovertemplate": "vuelo %{x}<br>z = %{y:.2f}<extra></extra>"}]
    lay["xaxis2"]["title"] = {"text": "Número de vuelo"}
    lay["yaxis"]["title"] = {"text": "T48 medido [°R]"}
    lay["yaxis2"]["title"] = {"text": "Residuo (z)"}
    shapes = [{"type": "line", "x0": onset - 0.5, "x1": onset - 0.5, "y0": 0, "y1": 1, "yref": "paper",
               "line": {"color": "var(--orange)", "dash": "dot", "width": 1.5}}] if pd.notna(onset) else []
    ann = [{"x": onset - 0.5, "y": 0.47, "yref": "paper", "xanchor": "left", "showarrow": False,
            "text": " empieza la degradación anormal", "font": {"color": "var(--muted)", "size": 11}}] if pd.notna(onset) else []
    out["figures"]["normalization"] = fig(data, height=480, shapes=shapes, annotations=ann, showlegend=False, **lay)
    rho_raw, rho_res = _corr_rho(u["T48_mean"], u["RUL"]), _corr_rho(u["z_T48"], u["RUL"])
    ds_, un = c.example_uid.split("_u")
    out["insights"]["normalization"] = (
        f"Motor {int(un)} de {ds_}. Medida, T48 casi no se relaciona con la vida restante (ρ = <b>{num(rho_raw, 2)}</b>); sin el "
        f"efecto del vuelo aparece la curva de desgaste (ρ = <b>{num(rho_res, 2)}</b>): plana hasta el vuelo {int(onset) - 1} y "
        f"cada vez más empinada hasta la falla en el vuelo {int(u['cycle'].max())}.")
    out["facts"].update(norm_rho_raw=num(rho_raw, 2), norm_rho_res=num(rho_res, 2))

    # Perfiles de vuelo repetidos: NASA reutiliza vuelos reales grabados
    prof = (f["n_sec"].astype(str) + "_" + f["alt_max"].round(0).astype(str) + "_" + f["TRA_mean"].round(3).astype(str))
    f["profile"] = prof
    n_prof = prof.nunique()
    by_fc = f.groupby("Fc")["profile"].nunique()
    shared = (f.groupby("profile")["uid"].nunique() > 1).mean()
    h = f[f["hs"] == 1]
    eta = {}
    for s in c.XS:
        x = h[f"res_{s}_mean"] - h.groupby("uid")[f"res_{s}_mean"].transform("mean")
        eta[s] = float(x.groupby(h["profile"]).transform("mean").var() / x.var())
    eta = pd.Series(eta).sort_values()
    out["figures"]["profiles"] = fig(
        [{"type": "bar", "orientation": "h", "y": eta.index.tolist(), "x": rnd(eta.values * 100, 2), "marker": {"color": "var(--hud)"},
          "customdata": [md.DESCRIPTIONS[s] for s in eta.index],
          "hovertemplate": "%{y} · %{customdata}<br>explicado por el perfil: %{x:.1f} %<extra></extra>"}],
        height=440, bargap=0.25, xaxis={"title": {"text": "Variación del residuo entre vuelos sanos explicada por el perfil de vuelo [%]"}, "range": [0, 100]})
    names = c.cfg["flight_classes"]
    out["insights"]["profiles"] = (
        f"Los {num(len(f))} vuelos usan sólo <b>{n_prof} perfiles distintos</b> (grabaciones reales que el simulador reutiliza): "
        + ", ".join(f"{by_fc[k]} en vuelos {names[k]}s" for k in by_fc.index) +
        f". El {pct(shared)} de los perfiles aparece en más de un motor. En la etapa sana, el perfil explica el "
        f"<b>{pct(eta['T48'])}</b> de la variación del residuo de T48 entre vuelos (y más del 90 % en {', '.join(eta[eta > 0.9].index)}): "
        "el residuo conserva un sesgo propio de cada perfil. Por eso el modelo promedia varios vuelos (medias móviles) "
        "y recibe descriptores del vuelo.")
    out["facts"].update(n_profiles=str(n_prof), profile_eta_t48=pct(eta["T48"]))

    # Señal frente a ruido (tarjetas)
    within = f["T48_std"].median()
    k = e["edge_flights"]
    g = f.sort_values("cycle").groupby("uid")["res_T48_mean"]
    drift = (g.apply(lambda x: x.iloc[-k:].mean() - x.iloc[:k].mean())).median()
    out["facts"].update(t48_within=f"±{num(within)} °R", t48_drift=f"{signed(drift)} °R")
    out["signal_noise"] = [
        {"k": "EXPLICADO POR EL VUELO", "value": f"{num(med_r2 * 100, 3)} %", "hot": False,
         "text": "de la variación de un sensor físico típico se explica sólo con altitud, velocidad, potencia y temperatura de entrada (mediana)."},
        {"k": "T48 · VARIACIÓN EN UN VUELO", "value": f"±{num(within)} °R", "hot": False,
         "text": "es el desvío estándar típico de la temperatura de turbina dentro de un mismo vuelo."},
        {"k": "T48 · AUMENTO EN TODA LA VIDA", "value": f"{signed(drift)} °R", "hot": True,
         "text": f"sube (mediana entre motores) desde los primeros {k} vuelos hasta los últimos {k}, en iguales condiciones de "
                 f"vuelo. Es cerca del {num(drift / within * 100)} % de la variación dentro de un vuelo."},
    ]


# ════════════════════════════════════════════════════════════════════════════
#  4 · Desgaste
# ════════════════════════════════════════════════════════════════════════════
def _loss_table(c: Ctx) -> pd.DataFrame:
    last = c.f.sort_values("cycle").groupby("uid")[c.T].last()
    loss = -last * 100
    loss["ds"] = c.units.loc[loss.index, "ds"]
    return loss.groupby("ds").mean()


def section_wear(c: Ctx, out: dict) -> None:
    f, e = c.f, c.e
    loss = _loss_table(c)
    short = [md.short_param(p) for p in c.T]
    z = loss[c.T].to_numpy()
    out["figures"]["health_heat"] = fig(
        [{"type": "heatmap", "x": short, "y": loss.index.tolist(), "z": np.round(z, 3).tolist(),
          "text": [[num(v, 2) if v > e["component_loss_threshold_pct"] else "" for v in row] for row in z], "texttemplate": "%{text}",
          "textfont": {"size": 11}, "xgap": 2, "ygap": 2, "colorscale": HEAT, "colorbar": {"thickness": 10, "title": {"text": "% de pérdida"}},
          "hovertemplate": "%{y} · %{x}<br>pérdida media en el último vuelo: %{z:.2f} %<extra></extra>"}],
        height=430, yaxis={"autorange": "reversed", "showgrid": False}, xaxis={"showgrid": False, "tickangle": -30})
    comps = {}
    for ds, row in loss.iterrows():
        hit = [p for p in c.T if row[p] > e["component_loss_threshold_pct"]]
        comps[ds] = [cp for cp in md.COMPONENTS if any(md.component_of(p) == cp for p in hit)]
    mx = loss[c.T].stack().idxmax()
    all5 = [d for d, cs in comps.items() if len(cs) == len(md.COMPONENTS)]
    out["insights"]["health_heat"] = (
        "Cada subconjunto daña componentes distintos: " + "; ".join(f"<b>{d}</b>: {' + '.join(cs)}" for d, cs in comps.items()) +
        f". La mayor pérdida es la de {md.short_param(mx[1])} en {mx[0]} ({num(loss.loc[mx], 1)} %)" +
        (f"; {' y '.join(all5)} desgastan los cinco componentes a la vez." if all5 else "."))
    out["tables"]["health_heat"] = {"columns": ["dataset"] + short,
                                    "rows": [[d] + [num(v, 2) for v in r] for d, r in zip(loss.index, z)]}
    n_combos = len({tuple(v) for v in comps.values()})
    out["facts"]["n_combos"] = str(n_combos)

    # Evolución del parámetro más afectado de cada subconjunto (selector por subconjunto)
    data, vis_by_ds = [], {}
    for ds in loss.index:
        p = loss.loc[ds, c.T].idxmax()
        vis_by_ds[ds] = []
        for uid, u in f[f["ds"] == ds].groupby("uid"):
            u = u.sort_values("cycle")
            on = u.loc[u["hs"] == 0, "cycle"].min()
            for phase, part, col, w in [("sano", u[u["cycle"] <= on] if pd.notna(on) else u, "var(--hud)", 1.6),
                                        ("degradado", u[u["cycle"] >= on - 1] if pd.notna(on) else u.iloc[:0], "var(--orange)", 2)]:
                if part.empty:
                    continue
                vis_by_ds[ds].append(len(data))
                data.append({"type": "scatter", "mode": "lines", "x": rnd(part["cycle"]), "y": rnd(part[p] * 100, 4),
                             "name": f"Fase {phase}", "legendgroup": phase, "showlegend": False, "visible": ds == loss.index[0],
                             "line": {"color": col, "width": w},
                             "hovertemplate": f"{ds} · motor {int(u['unit'].iloc[0])}<br>vuelo %{{x}}<br>{md.short_param(p)}: %{{y:.2f}} %<extra>{phase}</extra>"})
    for ds in loss.index:     # una sola entrada de leyenda por fase en cada vista
        seen = set()
        for i in vis_by_ds[ds]:
            data[i]["_ds"] = ds
    first_ds = loss.index[0]
    seen = set()
    for i in vis_by_ds[first_ds]:
        if data[i]["legendgroup"] not in seen:
            data[i]["showlegend"] = True
            seen.add(data[i]["legendgroup"])
    for d in data:
        d.pop("_ds", None)
    buttons = []
    for ds in loss.index:
        vis = [False] * len(data)
        for i in vis_by_ds[ds]:
            vis[i] = True
        p = md.short_param(loss.loc[ds, c.T].idxmax())
        buttons.append({"label": ds, "method": "update", "args": [{"visible": vis}, {"yaxis.title.text": f"{p} [% respecto de nuevo]"}]})
    out["figures"]["health_traj"] = fig(
        data, height=560, xaxis={"title": {"text": "Número de vuelo"}},
        yaxis={"title": {"text": f"{md.short_param(loss.loc[first_ds, c.T].idxmax())} [% respecto de nuevo]"}},
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 1, "xanchor": "right", "y": 1.13, "yanchor": "top"}])
    spread = {}
    for ds in loss.index:
        p = loss.loc[ds, c.T].idxmax()
        v0 = -f[(f["ds"] == ds) & (f["cycle"] == 1)][p] * 100
        spread[ds] = (v0.max() - v0.min()) / loss.loc[ds, p]
    out["insights"]["health_traj"] = (
        "Dos etapas en todos los motores: una caída lenta y casi recta (azul) y, desde un vuelo distinto para cada uno, "
        "una caída cada vez más pronunciada (naranja). Además, cada motor arranca con una salud inicial diferente: la "
        f"diferencia entre motores en el primer vuelo llega al {pct(max(spread.values()))} de la pérdida total del componente.")

    # Etapas de vida: vuelos sanos vs degradados por motor
    st = f.groupby("uid").agg(sano=("hs", "sum"), total=("cycle", "max"), ds=("ds", "first"), unit=("unit", "first"), split=("split", "first"))
    st["degr"] = st["total"] - st["sano"]
    st["frac"] = st["sano"] / st["total"]
    st = st.sort_values(["ds", "unit"])
    lab = [f"{r.ds} · {'D' if r.split == 'dev' else 'P'} · m{r.unit}" for r in st.itertuples()]
    out["figures"]["hs"] = fig(
        [{"type": "bar", "orientation": "h", "y": lab, "x": rnd(st["sano"]), "name": "Vuelos en estado sano", "marker": {"color": "var(--hud)"},
          "hovertemplate": "%{y}<br>sano: %{x} vuelos<extra></extra>"},
         {"type": "bar", "orientation": "h", "y": lab, "x": rnd(st["degr"]), "name": "Vuelos con degradación anormal", "marker": {"color": "var(--orange)"},
          "hovertemplate": "%{y}<br>degradado: %{x} vuelos<extra></extra>"}],
        barmode="stack", height=max(500, 11 * len(st)), bargap=0.25, margin={"l": 120}, legend={"traceorder": "normal", "y": 1.04},
        yaxis={"tickfont": {"size": 9}, "autorange": "reversed"}, xaxis={"title": {"text": "Vuelos hasta la falla"}})
    onset = st["sano"] + 1
    out["facts"]["frac_sano"] = pct(st["frac"].mean())
    out["insights"]["hs"] = (
        f"Un motor está sano, en promedio, sólo el <b>{pct(st['frac'].mean())}</b> de su vida ({pct(st['frac'].min())} a "
        f"{pct(st['frac'].max())}); la degradación anormal empieza entre el vuelo {onset.min()} y el {onset.max()}. Durante la "
        "etapa sana el RUL es casi imposible de estimar con precisión: ahí deberían esperarse los intervalos más anchos.")
    c.stage = st

    # Tarjetas por subconjunto (componentes, pérdidas y resumen)
    c.comps, c.loss = comps, loss


# ════════════════════════════════════════════════════════════════════════════
#  5 · Señales
# ════════════════════════════════════════════════════════════════════════════
def section_signals(c: Ctx, out: dict) -> None:
    f, e = c.f, c.e
    sens = c.XS + c.XV
    rho = pd.DataFrame({ds: {s: _corr_rho(g[f"z_{s}"], g["RUL"]) for s in sens} for ds, g in f.groupby("ds")})
    # ρ con la CERCANÍA a la falla (−RUL), como en la Parte I: rojo = sube al acercarse la falla
    near = -rho
    ylab = c.XS + [f"{s}*" for s in c.XV]
    out["figures"]["rul_corr"] = fig(
        [{"type": "heatmap", "x": near.columns.tolist(), "y": ylab, "z": np.round(near.loc[sens].to_numpy(), 3).tolist(),
          "zmin": -1, "zmax": 1, "xgap": 2, "ygap": 1, "colorscale": DIVERGING, "colorbar": {"thickness": 10, "title": {"text": "ρ"}},
          "customdata": [[md.DESCRIPTIONS.get(s, "")] * near.shape[1] for s in sens],
          "hovertemplate": "%{y} · %{customdata}<br>%{x}: ρ = %{z:.2f}<extra></extra>"}],
        height=700, yaxis={"autorange": "reversed", "showgrid": False}, xaxis={"showgrid": False, "side": "top"})
    strength = rho.loc[c.XS].abs().mean(1).sort_values(ascending=False)
    top4 = strength.index[:4].tolist()
    weak = strength.index[-2:].tolist()
    flips = [s for s in c.XS if (near.loc[s] > 0.3).any() and (near.loc[s] < -0.3).any()]
    fan_ds = [d for d, cs in c.comps.items() if cs == ["fan"]]
    fan_txt = ""
    if fan_ds:
        d = fan_ds[0]
        fan_top = near[d].abs().sort_values(ascending=False).index[:3]
        fan_txt = f" En {d} (sólo fan) reaccionan <b>{', '.join(fan_top)}</b>."
    out["insights"]["rul_corr"] = (
        f"Entre los sensores físicos, los que mejor siguen el desgaste en promedio son <b>{', '.join(top4)}</b>." + fan_txt +
        (f" {', '.join(flips)} cambian de signo según el escenario (suben en unos y bajan en otros)." if flips else "") +
        f" {' y '.join(weak)} casi no cambian (|ρ| medio ≤ {num(strength[weak].max(), 2)}).")
    c.top4, c.rho = top4, rho

    # Cuenta regresiva: residuo de los 4 mejores sensores en los últimos vuelos
    w = e["trend_window"]
    last = f[f["RUL"] <= w]
    lay = subplot_axes(2, 2, vgap=0.14, hgap=0.08)
    lay.pop("_n")
    data, ann, alarm = [], [], {}
    for i, s in enumerate(top4, start=1):
        xr, yr = axis_ref(i)
        sign = np.sign(near.loc[s].mean()) or 1
        q = last.groupby("RUL")[f"z_{s}"].quantile([0.25, 0.5, 0.75]).unstack() * sign
        data += [{"type": "scatter", "mode": "lines", "x": rnd(q.index), "y": rnd(q[0.25], 3), "xaxis": xr, "yaxis": yr,
                  "line": {"width": 0}, "hoverinfo": "skip", "showlegend": False},
                 {"type": "scatter", "mode": "lines", "x": rnd(q.index), "y": rnd(q[0.75], 3), "xaxis": xr, "yaxis": yr,
                  "fill": "tonexty", "fillcolor": "rgba(243,114,44,0.16)", "line": {"width": 0}, "hoverinfo": "skip", "showlegend": False},
                 {"type": "scatter", "mode": "lines", "x": rnd(q.index), "y": rnd(q[0.5], 3), "xaxis": xr, "yaxis": yr,
                  "line": {"color": "var(--orange)", "width": 2.2}, "showlegend": False,
                  "hovertemplate": f"faltan %{{x}} vuelos<br>{s}: mediana z = %{{y:.2f}}<extra></extra>"}]
        lay[f"xaxis{'' if i == 1 else i}"]["autorange"] = "reversed"
        if i > 2:
            lay[f"xaxis{i}"]["title"] = {"text": "Vuelos que faltan para la falla"}
        if i in (1, 3):
            lay[f"yaxis{'' if i == 1 else i}"]["title"] = {"text": "residuo (z)"}
        dom_x, dom_y = lay[f"xaxis{'' if i == 1 else i}"]["domain"], lay[f"yaxis{'' if i == 1 else i}"]["domain"]
        ann.append({"x": dom_x[0], "y": dom_y[1] + 0.01, "xref": "paper", "yref": "paper", "xanchor": "left", "yanchor": "bottom",
                    "showarrow": False, "font": {"size": 12}, "text": f"<b>{s}</b> · {md.DESCRIPTIONS[s]}" + (" (invertido)" if sign < 0 else "")})
        med = q[0.5].sort_index()
        above = med > e["z_alarm"]
        ok = above.cumprod().astype(bool)          # tramo continuo por encima del umbral, desde la falla hacia atrás
        alarm[s] = (int(ok[ok].index.max()) if ok.any() else 0, float(med.iloc[0]))
    out["figures"]["sensor_trends"] = fig(data, height=560, annotations=ann, **lay)
    s0 = top4[0]
    out["insights"]["sensor_trends"] = (
        f"Cerca de cero con mucha vida por delante y cada vez más lejos hacia el final. La mediana de {s0} supera "
        f"{num(e['z_alarm'])} desvíos cuando faltan <b>{alarm[s0][0]} vuelos</b> y llega a z ≈ {num(alarm[s0][1], 1)} en la falla. "
        "La banda es ancha porque cada motor parte de una salud distinta: por eso el modelado usa también el residuo "
        "respecto de los primeros vuelos de cada motor.")
    out["facts"]["alarm_flights"] = str(alarm[s0][0])

    # Evolución a lo largo de la vida (fracción de vida), por sensor y escenario
    nb = e["life_bins"]
    fb = f.assign(lb=np.minimum((f["life"] * nb).astype(int), nb - 1) / nb + 0.5 / nb)
    data, vis = [], {}
    ds_colors = ["var(--c-fan)", "var(--c-lpc)", "var(--c-hpc)", "var(--navy)", "var(--amber)", "var(--orange)",
                 "var(--c-hpt)", "var(--ti)", "var(--hud-2)", "var(--muted)"]
    for s in c.XS:
        vis[s] = []
        for j, (ds, g) in enumerate(fb.groupby("ds")):
            m = g.groupby("lb")[f"z_{s}"].median()
            vis[s].append(len(data))
            data.append({"type": "scatter", "mode": "lines+markers", "x": rnd(m.index * 100, 1), "y": rnd(m.values, 3), "name": ds,
                         "visible": s == top4[0], "line": {"color": ds_colors[j % len(ds_colors)], "width": 2}, "marker": {"size": 4},
                         "hovertemplate": f"{ds} · {s}<br>%{{x:.0f}} % de la vida<br>mediana z = %{{y:.2f}}<extra></extra>"})
    buttons = []
    for s in [top4[0]] + [x for x in c.XS if x != top4[0]]:
        v = [False] * len(data)
        for i in vis[s]:
            v[i] = True
        buttons.append({"label": s, "method": "update", "args": [{"visible": v}, {"yaxis.title.text": f"{s} · residuo (z), mediana"}]})
    out["figures"]["life_evolution"] = fig(
        data, height=500, xaxis={"title": {"text": "Avance de la vida del motor [%]"}},
        yaxis={"title": {"text": f"{top4[0]} · residuo (z), mediana"}},
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 1, "xanchor": "right", "y": 1.15, "yanchor": "top"}])
    early = fb[fb["life"] <= 0.5]
    late = fb[fb["life"] >= 0.9]
    rise = (late[f"z_{top4[0]}"].abs().median() / max(early[f"z_{top4[0]}"].abs().median(), 1e-9))
    out["insights"]["life_evolution"] = (
        f"Con la vida normalizada (0 % = primer vuelo, 100 % = falla) todos los escenarios muestran la misma forma: "
        f"meseta durante la primera mitad y aceleración al final. En {top4[0]}, el residuo típico del último 10 % de la vida "
        f"es <b>{num(rise, 1)} veces</b> el de la primera mitad. La forma es común, pero la magnitud y el sensor que reacciona "
        "dependen del escenario.")

    # Atípicos: ¿los valores extremos son errores o desgaste? Se miden contra los vuelos sanos de referencia.
    zt = e["outlier_z"]
    rows = []
    for s in c.XS:
        far = f[f"z_{s}"].abs() > zt
        rows.append((s, far[f["hs"] == 1].mean(), far[f["hs"] == 0].mean(), int(far.sum())))
    ot = pd.DataFrame(rows, columns=["s", "sano", "degr", "n"]).sort_values("degr", ascending=False)
    out["figures"]["outliers"] = fig(
        [{"type": "bar", "name": "Vuelos sanos", "x": ot["s"].tolist(), "y": rnd(ot["sano"] * 100, 3), "marker": {"color": "var(--hud)"},
          "hovertemplate": "%{x}<br>vuelos sanos extremos: %{y:.2f} %<extra></extra>"},
         {"type": "bar", "name": "Vuelos degradados", "x": ot["s"].tolist(), "y": rnd(ot["degr"] * 100, 3), "marker": {"color": "var(--orange)"},
          "hovertemplate": "%{x}<br>vuelos degradados extremos: %{y:.2f} %<extra></extra>"}],
        barmode="group", height=380, bargap=0.25, yaxis={"title": {"text": f"Vuelos con |z| > {num(zt)} [%]"}})
    far_any = (f[[f"z_{s}" for s in c.XS]].abs() > zt).any(axis=1)
    share_degr = (far_any & (f["hs"] == 0)).sum() / max(far_any.sum(), 1)
    healthy_far = far_any[f["hs"] == 1].mean()
    out["insights"]["outliers"] = (
        f"Un vuelo es extremo si algún residuo se aleja más de {num(zt)} desvíos de lo normal en un motor sano. Son el "
        f"{pct(far_any.mean())} de los vuelos, pero el <b>{pct(share_degr)}</b> está en la etapa degradada; entre los vuelos sanos "
        f"sólo el {pct(healthy_far, 1)} es extremo. No son errores de medición sino desgaste (y en la etapa sana, el efecto del "
        "perfil de vuelo). Por eso <b>no se elimina ningún vuelo</b>: los extremos son justamente la señal a predecir.")
    out["facts"].update(outlier_share_degr=pct(share_degr), outlier_healthy=pct(healthy_far, 1))
    out["tables"]["outliers"] = {"columns": ["sensor", "% sanos extremos", "% degradados extremos", "vuelos extremos"],
                                 "rows": [[r.s, num(r.sano * 100, 2), num(r.degr * 100, 2), int(r.n)] for r in ot.itertuples()]}

    # PCA de los 14 residuos físicos
    Z = f[[f"z_{s}" for s in c.XS]].to_numpy()
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
    Zs = (Z - mu) / sd
    U, S, Vt = np.linalg.svd(Zs, full_matrices=False)
    ev = S ** 2 / (S ** 2).sum()
    pc = Zs @ Vt[:2].T
    f2 = f.assign(pc1=pc[:, 0], pc2=pc[:, 1])
    healthy_c = f2.loc[f2["hs"] == 1, ["pc1", "pc2"]].mean()
    data, dss = [], sorted(f2["ds"].unique())
    for ds in dss:
        g = f2[f2["ds"] == ds]
        data.append({"type": "scattergl", "mode": "markers", "name": ds, "x": rnd(g["pc1"], 3), "y": rnd(g["pc2"], 3),
                     "marker": {"color": rnd(g["RUL"]), "coloraxis": "coloraxis", "size": 4, "opacity": 0.7},
                     "customdata": [[ds, int(a), int(b), int(r)] for a, b, r in zip(g["unit"], g["cycle"], g["RUL"])],
                     "hovertemplate": "%{customdata[0]} · motor %{customdata[1]} · vuelo %{customdata[2]}<br>faltan %{customdata[3]} vuelos<extra></extra>"})
    ends = f2[f2["RUL"] == 0].groupby("ds")[["pc1", "pc2"]].mean()
    data.append({"type": "scatter", "mode": "markers+text", "name": "Final de cada escenario", "x": rnd(ends["pc1"], 3), "y": rnd(ends["pc2"], 3),
                 "text": ends.index.tolist(), "textposition": "top center", "marker": {"symbol": "x", "size": 12, "color": "var(--navy)"},
                 "hovertemplate": "%{text}: posición media en la falla<extra></extra>"})
    buttons = [{"label": "Todos", "method": "restyle", "args": [{"visible": [True] * len(data)}]}]
    for i, ds in enumerate(dss):
        v = [False] * len(data)
        v[i] = v[-1] = True
        buttons.append({"label": ds, "method": "restyle", "args": [{"visible": v}]})
    out["figures"]["pca"] = fig(
        data, height=580, showlegend=False,
        coloraxis={"colorscale": RUL_SCALE, "colorbar": {"title": {"text": "RUL"}, "thickness": 10}},
        xaxis={"title": {"text": f"Componente principal 1 ({pct(ev[0])} de la variación)"}},
        yaxis={"title": {"text": f"Componente principal 2 ({pct(ev[1])})"}},
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 1, "xanchor": "right", "y": 1.12, "yanchor": "top"}])
    dist = np.hypot(f2["pc1"] - healthy_c["pc1"], f2["pc2"] - healthy_c["pc2"])
    rho_d = _corr_rho(dist, f2["RUL"])
    vec = ends.sub(healthy_c, axis=1)
    ang = np.degrees(np.arctan2(vec["pc2"], vec["pc1"]))
    spread = (ang.max() - ang.min())
    out["insights"]["pca"] = (
        f"Los vuelos sanos quedan juntos en una zona chica y cada escenario se aleja en su propia dirección (las direcciones "
        f"de falla abarcan {num(spread)}° alrededor de la zona sana). La distancia a la zona sana tiene ρ = <b>{num(rho_d, 2)}</b> "
        "con el RUL: sirve como indicador de salud general (lo usa la Línea 3).")
    out["facts"]["pca_rho"] = num(rho_d, 2)


# ════════════════════════════════════════════════════════════════════════════
#  6 · Vida útil
# ════════════════════════════════════════════════════════════════════════════
def section_life(c: Ctx, out: dict) -> None:
    f = c.f
    u = c.units.reset_index()
    data = [{"type": "box", "name": SPLIT_NAME[sp], "x": u.loc[u["split"] == sp, "ds"].tolist(), "y": rnd(u.loc[u["split"] == sp, "eol"]),
             "customdata": rnd(u.loc[u["split"] == sp, "unit"]), "boxpoints": "all", "jitter": 0.4, "pointpos": 0,
             "fillcolor": "rgba(0,0,0,0)", "line": {"width": 1.4}, "marker": {"color": SPLIT_COLOR[sp]},
             "hovertemplate": f"%{{x}} · motor %{{customdata}}<br>vida total: %{{y}} vuelos<extra>{SPLIT_NAME[sp]}</extra>"} for sp in ("dev", "test")]
    out["figures"]["lifespan"] = fig(data, height=430, boxmode="group", yaxis={"title": {"text": "Vida total [vuelos]"}})
    life = u.groupby("ds")["eol"].agg(["mean", "min", "max"])
    short2 = life["mean"].nsmallest(2)
    out["insights"]["lifespan"] = (
        f"Vidas de <b>{u['eol'].min()}</b> a <b>{u['eol'].max()}</b> vuelos (media {num(u['eol'].mean())}). Las más cortas son "
        f"<b>{', '.join(short2.index)}</b> ({' y '.join(num(v) for v in short2.values)} vuelos de media). Como cada motor dura "
        "distinto, al evaluar hay que separar motores completos: si un motor aparece en entrenamiento y prueba, el modelo "
        "\"conoce\" su vida.")
    out["tables"]["lifespan"] = {"columns": ["dataset", "media", "mínimo", "máximo"],
                                 "rows": [[d, num(r["mean"], 1), int(r["min"]), int(r["max"])] for d, r in life.iterrows()]}
    out["facts"].update(eol_min=str(u["eol"].min()), eol_max=str(u["eol"].max()))

    hrs = f.groupby("uid").agg(h=("hours", "sum"), eol=("eol", "first"), Fc=("Fc", "first"), ds=("ds", "first"), unit=("unit", "first"))
    names = c.cfg["flight_classes"]
    data = [{"type": "scatter", "mode": "markers", "name": f"Clase {k} · {names[k]}", "x": rnd(g["h"], 1), "y": rnd(g["eol"]),
             "customdata": [[d, int(n)] for d, n in zip(g["ds"], g["unit"])],
             "marker": {"color": FC_COLOR[k], "size": 10, "line": {"color": "var(--surface)", "width": 2}},
             "hovertemplate": f"%{{customdata[0]}} · motor %{{customdata[1]}}<br>%{{x:.0f}} h · %{{y}} vuelos<extra>Clase {k} · {names[k]}</extra>"}
            for k, g in hrs.groupby("Fc")]
    out["figures"]["life_vs_fc"] = fig(data, height=430, xaxis={"title": {"text": "Horas de vuelo acumuladas hasta la falla"}},
                                       yaxis={"title": {"text": "Vida total [vuelos]"}})
    by = hrs.groupby("Fc")[["eol", "h"]].mean()
    k0, k1 = by.index.min(), by.index.max()
    out["insights"]["life_vs_fc"] = (
        f"Vuelos {names[k0]}s: <b>{num(by.loc[k0, 'eol'])} vuelos</b> de vida media; {names[k1]}s: <b>{num(by.loc[k1, 'eol'])}</b>. "
        f"En horas pasa lo contrario ({num(by.loc[k0, 'h'])} h contra {num(by.loc[k1, 'h'])} h). Es una asociación (la clase "
        "se mezcla con el escenario), pero la clase de vuelo pesa en el RUL y se incluye como variable del modelo.")
    out["facts"].update(life_short=num(by.loc[k0, "eol"]), life_long=num(by.loc[k1, "eol"]))

    data = [{"type": "histogram", "name": SPLIT_NAME[sp], "x": rnd(f.loc[f["split"] == sp, "RUL"]), "xbins": {"size": 2}, "opacity": 0.75,
             "marker": {"color": SPLIT_COLOR[sp]}, "hovertemplate": f"RUL %{{x}}<br>%{{y}} vuelos<extra>{SPLIT_NAME[sp]}</extra>"} for sp in ("dev", "test")]
    out["figures"]["rul_dist"] = fig(data, barmode="overlay", height=340, bargap=0.05,
                                     xaxis={"title": {"text": "RUL [vuelos que faltan]"}}, yaxis={"title": {"text": "Cantidad de vuelos"}})
    out["insights"]["rul_dist"] = (
        f"Cada motor aporta un vuelo por cada valor de RUL, así que el histograma es plano hasta <b>{u['eol'].min() - 1}</b> y "
        f"luego baja (máx. {f['RUL'].max()}). Mediana: {num(f['RUL'].median())} vuelos. Hay pocos ejemplos de motores con mucha "
        "vida por delante: es donde el modelo tendrá menos datos y donde la incertidumbre debería ser mayor.")
    out["facts"]["rul_median"] = num(f["RUL"].median())


# ════════════════════════════════════════════════════════════════════════════
#  Resumen: KPIs, tarjetas por subconjunto, conclusiones y decisiones
# ════════════════════════════════════════════════════════════════════════════
def summary(c: Ctx, out: dict) -> None:
    f, files = c.f, c.files
    bad = files[~files["ok"]]["ds"].tolist()
    tot = f["n_sec"].sum()
    out["kpis"] = [
        {"code": "REG", "value": round(tot / 1e6, 1), "dec": 1, "unit": "M", "label": "registros, uno por segundo de vuelo"},
        {"code": "MOT", "value": int(f["uid"].nunique()), "label": "motores simulados hasta la falla"},
        {"code": "VUE", "value": int(len(f)), "label": "vuelos completos"},
        {"code": "HRS", "value": int(round(tot / 3600)), "unit": "h", "label": "horas de vuelo"},
        {"code": "SEN", "value": len(c.XS), "label": f"sensores físicos (+{len(c.XV)} virtuales)"},
        {"code": "ESC", "value": len(c.ds_ok), "label": "escenarios de falla analizados"},
    ]
    out["facts"].update(
        rows_total_fmt=f"{num(tot / 1e6, 1)} millones de", n_datasets_ok=str(len(c.ds_ok)), n_units=str(f["uid"].nunique()),
        n_flights=num(len(f)), hours=num(tot / 3600), bad_ds=", ".join(bad) if bad else "ninguno",
        flight_mean_h=num(f["hours"].mean(), 1))

    rows_ds = f.groupby("ds")["n_sec"].sum()
    st = c.stage
    names = c.cfg["flight_classes"]
    cards = []
    for ds in c.ds_list:
        if ds not in c.ds_ok:
            r = files.set_index("ds").loc[ds]
            cards.append({"name": ds, "ok": False, "units": 0, "flights": 0, "rows_fmt": millions((r["rows_dev"] or 0) + (r["rows_test"] or 0)) if r["ok"] else "—",
                          "eol": "—", "components": [], "loss": {}, "bullets": []})
            continue
        g = f[f["ds"] == ds]
        uu = c.units[c.units["ds"] == ds]
        lrow = c.loss.loc[ds, c.T].sort_values(ascending=False)
        lrow = lrow[lrow > c.e["component_loss_threshold_pct"]]
        best = c.rho.loc[c.XS, ds].abs().idxmax()
        fcs = uu["Fc"].value_counts().sort_index()
        cards.append({
            "name": ds, "ok": True, "units": int(len(uu)), "flights": int(len(g)), "rows_fmt": millions(rows_ds[ds]),
            "eol": num(uu["eol"].mean()), "components": c.comps[ds],
            "loss": {md.short_param(p): round(float(v), 2) for p, v in lrow.items()},
            "bullets": [f"Vida media: <b>{num(uu['eol'].mean())}</b> vuelos (entre {uu['eol'].min()} y {uu['eol'].max()}).",
                        f"En estado sano: <b>{pct(st[st['ds'] == ds]['frac'].mean())}</b> de la vida, en promedio.",
                        f"Mayor pérdida: <b>{md.short_param(lrow.index[0])}</b> ({num(lrow.iloc[0], 2)} %).",
                        f"Sensor físico que más cambia: <b>{best}</b> (|ρ| = {num(abs(c.rho.loc[best, ds]), 2)}).",
                        "Clases de vuelo: " + ", ".join(f"{n} {names[k]}" for k, n in fcs.items()) + "."],
        })
    out["datasets"] = cards

    F = out["facts"]
    out["takeaways"] = [
        {"title": "Datos limpios, con un archivo dañado" if bad else "Datos limpios",
         "text": f"{F['rows_total_fmt']} registros sin nulos ni infinitos y con todas las verificaciones de consistencia en orden."
                 + (f" El archivo de {F['bad_ds']} viene truncado en Kaggle y quedó afuera." if bad else "")},
        {"title": "Los sensores reflejan sobre todo cómo se vuela",
         "text": f"Las cuatro condiciones de vuelo explican el {F['med_r2']} de un sensor físico típico. En T48, el desgaste de toda "
                 f"la vida suma {F['t48_drift']}, contra variaciones de {F['t48_within']} dentro de un mismo vuelo."},
        {"title": "Hay que descontar el efecto del vuelo",
         "text": f"Con un modelo de motor sano se obtienen residuos donde el desgaste se ve claro: en el motor de ejemplo, la "
                 f"correlación de T48 con el RUL pasa de {F['norm_rho_raw']} a {F['norm_rho_res']}."},
        {"title": "Cada escenario de falla es distinto",
         "text": f"Hay {F['n_combos']} combinaciones distintas de componentes dañados en {F['n_datasets_ok']} subconjuntos, y cada "
                 "una cambia qué sensores reaccionan."},
        {"title": "El desgaste tiene dos etapas",
         "text": f"Primero lento y casi lineal, después cada vez más rápido. Los motores están sanos sólo el {F['frac_sano']} de "
                 "su vida, en promedio."},
        {"title": "El tipo de vuelo influye en la vida",
         "text": f"Los motores de vuelos largos duran {F['life_long']} vuelos en promedio y los de vuelos cortos, {F['life_short']}."},
    ]
    out["decisions"] = [
        {"finding": f"Las condiciones de vuelo explican el {F['med_r2']} de cada sensor.",
         "decision": "Trabajar con residuos (medido − motor sano) en lugar de los sensores crudos."},
        {"finding": f"Hay sólo {F['n_profiles']} perfiles de vuelo reutilizados y el perfil explica el {F['profile_eta_t48']} del ruido del residuo de T48.",
         "decision": "Suavizar con medias y pendientes móviles de varios vuelos e incluir descriptores del vuelo (duración, altitud, potencia)."},
        {"finding": "Cada motor arranca con una salud inicial distinta.",
         "decision": "Restar a cada motor su propia línea base (primeros vuelos) y usar medias y pendientes móviles."},
        {"finding": f"Vidas de {F['eol_min']} a {F['eol_max']} vuelos; los vuelos de un motor están correlacionados.",
         "decision": "Partir train / calibración / test por motor completo, nunca por vuelo."},
        {"finding": f"El {F['frac_sano']} de la vida es etapa sana, casi sin señal.",
         "decision": "Esperar más incertidumbre lejos de la falla: intervalos que se adapten y métricas por tramo de RUL."},
        {"finding": "El score de NASA castiga más sobreestimar la vida.",
         "decision": "Intervalos conformales asimétricos: menos tolerancia a quedar por encima del RUL real."},
        {"finding": f"Los vuelos largos viven menos vuelos ({F['life_long']} vs {F['life_short']}).",
         "decision": "Incluir la clase de vuelo y la duración como variables; evaluar la transferencia entre clases (Línea 2)."},
        {"finding": f"El {F['outlier_share_degr']} de los vuelos extremos está en la etapa degradada; entre los sanos sólo el {F['outlier_healthy']}.",
         "decision": "No eliminar vuelos: los extremos son la señal que hay que predecir."},
        {"finding": "Los sensores virtuales y los parámetros de salud no existen en un avión real.",
         "decision": "Usarlos sólo para entender el EDA; el modelo usa condiciones de vuelo y los 14 sensores físicos."},
    ]
    out["generated"] = datetime.now().strftime("%d/%m/%Y %H:%M")
    out["mode"] = c.mode


def run(cfg: dict, paths: Paths) -> dict:
    c = Ctx(cfg, paths)
    out = {"figures": {}, "insights": {}, "tables": {}, "facts": {}}
    for step in (section_dataset, section_flight, section_context, section_wear, section_signals, section_life, summary):
        step(c, out)
        log.info("  EDA · %s", step.__name__)
    save_json(out, paths.outputs / "eda.json")
    log.info("EDA: %d figuras, %d cifras → %s", len(out["figures"]), len(out["facts"]), paths.outputs / "eda.json")
    return out
