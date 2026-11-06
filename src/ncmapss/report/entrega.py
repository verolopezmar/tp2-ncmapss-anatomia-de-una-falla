"""Arma reports/entrega_modelo.html: el modelo final, sus resultados y sus límites, en una página.

Todas las cifras y figuras salen de outputs/full/cvplus.json y cvplus_intervals.parquet. Funciona sin conexión.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly

from ..config import Paths
from ..fmt import num

log = logging.getLogger(__name__)
INK, MUTED, GRID = "#0b1a2c", "#6a7e97", "#e8eef5"
BLUE, ORANGE, GRAY = "#0077b6", "#d9601a", "#7a4fb5"
METHODS = {"base_tft": "TFT · 11 motores", "base_catboost": "CatBoost · 11 motores",
           "cvplus": "CatBoost CV+ · 60 motores", "cvplus_aci": "CatBoost CV+ y ACI"}


def _ci(v: list[float], d: int = 1, scale: float = 1) -> str:
    return f"{num(v[0] * scale, d)} – {num(v[1] * scale, d)}"


def _layout(**kw) -> dict:
    base = {"paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)", "margin": {"l": 56, "r": 18, "t": 12, "b": 48},
            "font": {"family": "Inter, Segoe UI, system-ui, sans-serif", "size": 13, "color": INK},
            "hoverlabel": {"bgcolor": "#ffffff", "font": {"color": INK}}, "legend": {"orientation": "h", "y": 1.13, "x": 0},
            "separators": ",."}
    return {**base, **kw}


def _axis(title: str, **kw) -> dict:
    return {"title": {"text": title}, "gridcolor": GRID, "linecolor": "#c9d4e1", "zeroline": False, "ticks": "", **kw}


def _ci_fig(R: dict, key: str, metric: str, ref: float, ref_label: str, title: str, rng: list[float]) -> dict:
    """Punto e intervalo de confianza por método, contra el valor de referencia."""
    names = list(METHODS)
    x = [100 * R["test"][m][key] for m in names]
    lo = [100 * R["ci"][m][metric][0] for m in names]
    hi = [100 * R["ci"][m][metric][1] for m in names]
    data = [{"type": "scatter", "mode": "markers", "y": [METHODS[m] for m in names], "x": x, "showlegend": False,
             "marker": {"size": 11, "color": BLUE, "line": {"color": "#ffffff", "width": 2}},
             "error_x": {"type": "data", "symmetric": False, "array": [h - v for h, v in zip(hi, x)],
                         "arrayminus": [v - l for v, l in zip(x, lo)], "color": BLUE, "thickness": 2, "width": 5},
             "customdata": [[a, b] for a, b in zip(lo, hi)],
             "hovertemplate": "%{y}<br>%{x:.1f} % · IC 95 % %{customdata[0]:.1f} – %{customdata[1]:.1f} %<extra></extra>"}]
    return {"data": data, "layout": _layout(
        height=260, margin={"l": 190, "r": 18, "t": 26, "b": 48}, xaxis=_axis(title, range=rng),
        yaxis={"autorange": "reversed", "gridcolor": GRID, "ticks": ""},
        shapes=[{"type": "line", "x0": 100 * ref, "x1": 100 * ref, "y0": 0, "y1": 1, "yref": "paper",
                 "line": {"color": INK, "dash": "dot", "width": 1.4}}],
        annotations=[{"x": 100 * ref, "y": 1, "yref": "paper", "yanchor": "bottom", "showarrow": False, "text": ref_label,
                      "font": {"size": 12, "color": INK}}])}


def _policy_fig(R: dict) -> dict:
    ins = R["inspection"]
    tb = ins["by_tau"][str(ins["lead_flights"])]
    data = []
    for name, color, label in (("cota inferior", BLUE, "Cota inferior CV+"), ("mediana", ORANGE, "Mediana")):
        c = sorted(ins["curves"][name], key=lambda r: r["waste"])
        data.append({"type": "scatter", "mode": "lines", "name": label, "x": [r["waste"] for r in c], "y": [100 * r["late"] for r in c],
                     "customdata": [r["H"] for r in c], "line": {"color": color, "width": 2, "shape": "hv"},
                     "hovertemplate": label + "<br>umbral %{customdata} · %{x:.1f} vuelos · %{y:.1f} % tardías<extra></extra>"})
    fx = tb["vida fija por clase"]
    pts = [("Vida fija por clase", GRAY, fx, "diamond"), ("Política elegida", BLUE, tb["cota inferior"], "circle")]
    for label, color, r, sym in pts:
        data.append({"type": "scatter", "mode": "markers", "name": label, "x": [r["waste"]], "y": [100 * r["late"]],
                     "marker": {"size": 13, "color": color, "symbol": sym, "line": {"color": "#ffffff", "width": 2}},
                     "hovertemplate": label + "<br>%{x:.1f} vuelos · %{y:.1f} % tardías<extra></extra>"})
    return {"data": data, "layout": _layout(height=380, xaxis=_axis("Vida útil desperdiciada por motor [vuelos]", range=[0, 20]),
                                            yaxis=_axis("Inspecciones tardías [% de motores]", range=[-1, 45]))}


def _engines(paths: Paths) -> dict:
    it = pd.read_parquet(paths.outputs / "cvplus_intervals.parquet").sort_values(["uid", "cycle"])
    return {u: {c: g[c].round(2).tolist() for c in ("cycle", "RUL", "q50", "lo", "hi")} for u, g in it.groupby("uid")}


def _row(R: dict, m: str) -> str:
    t, c = R["test"][m], R["ci"][m]
    cells = [METHODS[m], num(t["rmse"], 2), num(t["nasa"], 3), f"{num(100 * t['coverage'], 1)} %", _ci(c["cov"], 1, 100),
             f"{num(100 * t['miss_over'], 1)} %", _ci(c["over"], 1, 100), num(t["width"], 1)]
    cls = ' class="sel"' if m == "cvplus_aci" else ""
    return f"<tr{cls}>" + "".join(f"<td>{v}</td>" for v in cells) + "</tr>"


def _policy_row(name: str, r: dict) -> str:
    h = "—" if r.get("H") is None else f"{num(r['H'])} vuelos"
    return (f"<tr><td>{name}</td><td>{h}</td><td>{r['late_engines']} de 39</td><td>{num(r['waste'], 1)}</td>"
            f"<td>{_ci(r['ci']['waste'])}</td></tr>")


def run(cfg: dict, paths: Paths) -> Path:
    R = json.loads((paths.outputs / "cvplus.json").read_text(encoding="utf-8"))
    v1 = json.loads((paths.outputs / "cvplus_v1_early_stopping.json").read_text(encoding="utf-8"))
    res = json.loads((paths.outputs / "results.json").read_text(encoding="utf-8"))
    ins = R["inspection"]
    tau = ins["lead_flights"]
    pol = ins["by_tau"][str(tau)]
    low, med, fx = pol["cota inferior"], pol["mediana"], pol["vida fija por clase"]
    gap = pol["cota inferior - vida fija"]["waste"]
    t, a, b, ft = R["test"]["cvplus"], R["test"]["cvplus_aci"], R["test"]["base_catboost"], R["test"]["base_tft"]
    dev = R["development"][R["chosen"]]
    d_cb, d_tft = R["ci"]["cvplus - base_catboost"], R["ci"]["base_tft - base_catboost"]
    saving = 100 * (1 - low["waste"] / fx["waste"])
    ds7 = R["per_ds"]["DS07"]
    b_aci = next(r for r in res["metrics_table"] if r["modelo"] == "CatBoost" and r["intervalo"].startswith("ACI"))
    limits = [
        f"<b>Se usó la prueba dos veces.</b> La primera versión fijaba las iteraciones por grupo con early stopping (entre "
        f"{min(f['iteraciones'] for f in v1['fit']['folds'])} y {max(f['iteraciones'] for f in v1['fit']['folds'])}) y dio score NASA "
        f"{num(v1['test']['cvplus']['nasa'], 3)}. Después de verlo se pasó a {R['fit']['iteraciones']} iteraciones comunes, elegidas sólo con "
        f"pinball en desarrollo: {num(t['nasa'], 3)}. La regla no usa la prueba, pero esa decisión de rediseño sí vino después de mirarla.",
        f"<b>La cobertura supera el objetivo.</b> En desarrollo CV+ da {num(100 * dev['coverage'], 1)} %, justo lo pedido; en prueba, "
        f"{num(100 * t['coverage'], 1)} %. Los motores de prueba resultan más fáciles (RMSE {num(t['rmse'], 1)} contra {num(dev['rmse'], 1)}): "
        "los intervalos salen más anchos de lo necesario. ACI corrige parte de eso.",
        f"<b>El costo es ancho.</b> Con ACI, el intervalo pasa de {num(b_aci['ancho medio [vuelos]'], 1)} a {num(a['width'], 1)} vuelos. La mejora es de sobreestimación y "
        "cobertura; no de precisión.",
        f"<b>DS07 sigue siendo el punto débil:</b> cobertura {num(100 * ds7['coverage'], 1)} %, la más baja por subconjunto.",
        f"<b>Son 39 motores.</b> La tasa de tardías no se puede separar: cada motor vale 2,6 puntos, y el umbral de {tau} vuelos es un supuesto "
        "nuestro, no un dato de mantenimiento real. Los datos son simulados (C-MAPSS con perfiles de vuelo reales).",
    ]
    figs = {"cov": _ci_fig(R, "coverage", "cov", R["target"], "objetivo 90 %", "Cobertura en prueba [%]", [70, 100]),
            "over": _ci_fig(R, "miss_over", "over", R["alpha_over"], "tolerancia 3 %", "Sobreestimación en prueba [%]", [0, 12]),
            "policy": _policy_fig(R)}
    payload = json.dumps({"figs": figs, "engines": _engines(paths)}, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    plotly_js = (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_text(encoding="utf-8")
    rows = "".join(_row(R, m) for m in METHODS)
    pol_rows = _policy_row("Cota inferior CV+", low) + _policy_row("Mediana", med) + _policy_row("Vida fija por clase", fx)
    team = " · ".join(cfg["report"]["team"])
    html = f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Modelo final RUL</title><style>
:root{{--bg:#f4f7fb;--surface:#fff;--line:#e1e8f0;--ink:#0b1a2c;--ink2:#3a4d66;--muted:#6a7e97;--blue:#0077b6;--navy:#0a2540}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 Inter,"Segoe UI",system-ui,sans-serif}}
main{{max-width:1040px;margin:auto;padding:40px 24px 64px}}h1{{font-size:clamp(30px,5vw,44px);line-height:1.1;letter-spacing:-.02em;color:var(--navy);margin:6px 0 10px}}
h2{{font-size:22px;letter-spacing:-.01em;color:var(--navy);margin:0 0 8px}}.eyebrow{{font-size:13px;color:var(--blue);font-weight:600}}
.lead{{font-size:18px;color:var(--ink2);max-width:760px;margin:0}}.meta{{color:var(--muted);font-size:13.5px;margin-top:10px}}
section{{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:24px;margin-top:18px}}section>p{{color:var(--ink2);margin:0 0 12px;max-width:820px}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:24px}}.kpi{{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:16px}}
.kpi b{{display:block;font-size:30px;line-height:1.15;color:var(--navy);font-variant-numeric:tabular-nums}}.kpi span{{font-size:13.5px;color:var(--ink2);display:block;margin-top:4px}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.tw{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}}
th{{font-size:12.5px;color:var(--muted);font-weight:600;text-align:right;padding:8px 10px;border-bottom:1px solid #c9d4e1;vertical-align:bottom}}
td{{padding:9px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}
tr.sel td{{background:#e8f3f9;font-weight:600}}label{{font-size:14px;color:var(--ink2)}}select{{font:inherit;padding:6px 8px;border:1px solid #c9d4e1;border-radius:8px;background:#fff;margin-left:6px}}
ol{{margin:0;padding-left:20px;color:var(--ink2)}}li{{margin:8px 0}}code{{background:#eef3f8;padding:2px 6px;border-radius:6px;font-size:14px}}
@media(max-width:760px){{.kpis{{grid-template-columns:1fr 1fr}}.two{{grid-template-columns:1fr}}main{{padding:24px 16px 48px}}section{{padding:18px}}}}
</style></head><body><main>
<div class="eyebrow">Proyecto Prioridad 2 · Anatomía de una Falla</div>
<h1>Cuántos vuelos le quedan a un motor, y cuándo inspeccionarlo</h1>
<p class="lead">Un solo modelo: CatBoost multicuantil con intervalos CV+. Predice la vida útil remanente con datos NASA N-CMAPSS y evita sobreestimarla.</p>
<div class="meta">{team} · Generado desde los datos el {datetime.now().strftime('%d/%m/%Y %H:%M')}</div>
<div class="kpis">
<div class="kpi"><b>{num(t['rmse'], 1)}</b><span>vuelos de error típico (RMSE) en 39 motores nunca vistos</span></div>
<div class="kpi"><b>{num(100 * a['miss_over'], 1)} %</b><span>de sobreestimación, antes {num(b_aci['% real < cota inferior'], 1)} % con ACI; tolerancia 3 %</span></div>
<div class="kpi"><b>{num(100 * a['coverage'], 1)} %</b><span>de cobertura del intervalo, antes {num(b_aci['cobertura [%]'], 1)} % con ACI; objetivo 90 %</span></div>
<div class="kpi"><b>{num(saving, 0)} %</b><span>menos vida desperdiciada que inspeccionar en un vuelo fijo</span></div></div>
<section><h2>Qué hicimos</h2><ol>
<li><b>Elegimos CatBoost.</b> Le gana al TFT en validación ({num(res['val_nasa']['catboost'], 3)} contra {num(res['val_nasa']['tft'], 3)}) y en prueba: el TFT tiene RMSE entre {num(d_tft['rmse']['dif_ic95'][0], 2)} y {num(d_tft['rmse']['dif_ic95'][1], 2)} vuelos peor, IC 95 % remuestreando motores.</li>
<li><b>Calibramos con los 60 motores de desarrollo.</b> Antes eran 11. Dividimos en {cfg['cvplus']['folds']} grupos y, para cada uno, reentrenamos modelo sano, residuos y CatBoost sin esos motores.</li>
<li><b>Fijamos las reglas en desarrollo</b> y evaluamos después en los 39 motores de prueba oficiales.</li>
<li><b>Convertimos la predicción en una decisión:</b> cuándo ordenar la inspección.</li></ol></section>
<section><h2>Resultados en motores de prueba</h2><p>La sobreestimación baja de forma clara y el error puntual no cambia de forma significativa: diferencia de RMSE entre {num(d_cb['rmse']['dif_ic95'][0], 2)} y {num(d_cb['rmse']['dif_ic95'][1], 2)} vuelos, y de score NASA entre {num(d_cb['nasa']['dif_ic95'][0], 3)} y {num(d_cb['nasa']['dif_ic95'][1], 3)}.</p>
<div class="tw"><table><thead><tr><th>Método</th><th>RMSE</th><th>Score NASA</th><th>Cobertura</th><th>IC 95 %</th><th>Sobreestima</th><th>IC 95 %</th><th>Ancho</th></tr></thead><tbody>{rows}</tbody></table></div>
<div class="two" style="margin-top:14px"><div id="f-cov"></div><div id="f-over"></div></div></section>
<section><h2>El modelo, vuelo a vuelo</h2><p>Cada predicción usa sólo la historia hasta ese vuelo. El intervalo se angosta cuando aparece el desgaste y la cota inferior es la que dispara la inspección.</p>
<label>Motor de prueba<select id="sel"></select></label><div id="f-eng"></div></section>
<section><h2>De la predicción a la inspección</h2><p>Se ordena inspeccionar en el primer vuelo con cota ≤ un umbral. Suponemos que ejecutarla lleva {tau} vuelos: es tardía si quedan menos. El umbral se eligió en desarrollo para no superar {num(100 * ins['max_late_rate'])} % de tardías. Frente a un vuelo fijo por clase, se ahorran entre {num(-gap[1], 1)} y {num(-gap[0], 1)} vuelos por motor (IC 95 %).</p>
<div id="f-pol"></div><div class="tw"><table><thead><tr><th>Política</th><th>Umbral</th><th>Tardías</th><th>Vida desperdiciada</th><th>IC 95 %</th></tr></thead><tbody>{pol_rows}</tbody></table></div></section>
<section><h2>Límites que hay que decir</h2><ol>{''.join(f'<li>{x}</li>' for x in limits)}</ol></section>
<section><h2>Para reproducirlo</h2><p>Todo se corrió por etapas en esta computadora sobre el dataset completo, con el mismo comando: <code>python scripts/run_all.py --workers 1</code>. Demo desde el archivo crudo: <code>python scripts/predecir_motor.py --motor DS02_u11 --cada 10 --verificar</code>.</p></section>
</main><script>{plotly_js}</script><script id="d" type="application/json">{payload}</script><script>
const D=JSON.parse(document.getElementById('d').textContent),C={{displayModeBar:false,responsive:true}};
Plotly.newPlot('f-cov',D.figs.cov.data,D.figs.cov.layout,C);Plotly.newPlot('f-over',D.figs.over.data,D.figs.over.layout,C);Plotly.newPlot('f-pol',D.figs.policy.data,D.figs.policy.layout,C);
const L={{paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',height:380,margin:{{l:56,r:18,t:12,b:48}},separators:',.',font:{{family:'Inter,Segoe UI,system-ui,sans-serif',size:13,color:'{INK}'}},legend:{{orientation:'h',y:1.13,x:0}},hovermode:'x unified',hoverlabel:{{bgcolor:'#fff'}},xaxis:{{title:{{text:'Número de vuelo'}},gridcolor:'{GRID}',zeroline:false}},yaxis:{{title:{{text:'Vuelos restantes'}},gridcolor:'{GRID}',zeroline:false,rangemode:'tozero'}}}};
function eng(u){{const e=D.engines[u];Plotly.react('f-eng',[
{{x:e.cycle,y:e.hi,mode:'lines',line:{{width:0}},showlegend:false,hoverinfo:'skip'}},
{{x:e.cycle,y:e.lo,mode:'lines',fill:'tonexty',fillcolor:'rgba(0,119,182,.14)',line:{{color:'{BLUE}',width:1}},name:'Cota inferior e intervalo',hovertemplate:'cota inferior %{{y:.1f}}<extra></extra>'}},
{{x:e.cycle,y:e.q50,mode:'lines',line:{{color:'{BLUE}',width:2}},name:'Predicción',hovertemplate:'predicción %{{y:.1f}}<extra></extra>'}},
{{x:e.cycle,y:e.RUL,mode:'lines',line:{{color:'{ORANGE}',width:2,dash:'dash'}},name:'RUL real',hovertemplate:'real %{{y}}<extra></extra>'}}],L,C)}}
const s=document.getElementById('sel');Object.keys(D.engines).forEach(u=>s.add(new Option(u.replace('_u',' · motor '),u)));s.value='DS02_u11';s.onchange=()=>eng(s.value);eng(s.value);
</script></body></html>"""
    out = paths.reports / "entrega_modelo.html"
    out.write_text(html, encoding="utf-8")
    log.info("Entrega: %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return out
