"""Arma reports/informe_final.html con los resultados reales del pipeline.

Lee outputs/<modo>/{eda,results,line2,line3}.json, junta figuras, cifras y lecturas, redacta
las conclusiones a partir de esos números y las inyecta en la plantilla Jinja2, que reutiliza
la hoja de estilos y las ilustraciones de compartir.html (Parte I). Plotly va embebido para
que el informe funcione sin conexión.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import plotly
from jinja2 import Environment, FileSystemLoader

from ..config import Paths
from ..fmt import num

log = logging.getLogger(__name__)
HERE = Path(__file__).parent
ASSETS = HERE / "assets"


def _load(paths: Paths, name: str) -> dict:
    p = paths.outputs / f"{name}.json"
    if not p.exists():
        log.warning("  Falta %s: esa sección del informe quedará vacía", p.name)
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _num(s: str) -> float:
    """'6,50' -> 6.5 ; '88,1 %' -> 88.1"""
    return float(str(s).replace("%", "").replace(".", "").replace(",", ".").strip())


def assemble(cfg: dict, paths: Paths, mode: str) -> dict:
    eda, res, l2, l3 = (_load(paths, n) for n in ("eda", "results", "line2", "line3"))
    R = {k: v for k, v in eda.items() if k not in ("figures", "insights", "tables", "facts")}
    R["figures"], R["insights"], R["tables"] = {}, {}, {}
    for d in (eda, res, l2, l3):
        for k in ("figures", "insights", "tables"):
            R[k].update(d.get(k, {}))
    facts = dict(eda.get("facts", {}))
    facts.update({f"r_{k}": v for k, v in res.get("facts", {}).items()})
    R["facts"] = facts
    for k in ("split_summary", "metrics_table", "model_meta", "discussion", "main_model"):
        R[k] = res.get(k)
    R["line2_table"] = l2.get("line2_table")
    R["line3_table"] = l3.get("line3_table")
    R["future_discussion"] = l2.get("future_discussion_l2", []) + l3.get("future_discussion_l3", [])
    tft = (res.get("model_meta") or {}).get("tft", {})
    if tft.get("search"):
        R["tft_search"] = [{"configuración": (", ".join(f"{k} = {v}" for k, v in s["config"].items()) if isinstance(s["config"], dict) else "base"),
                            "pérdida en validación": s["val_loss"], "épocas": s["epochs"], "elegida": "✓" if i == tft.get("chosen") else ""}
                           for i, s in enumerate(tft["search"])]

    # KPIs de portada: el dataset y el resultado principal
    kpis = [k for k in eda.get("kpis", []) if k["code"] in ("REG", "MOT", "VUE")]
    if res:
        rf = res["facts"]
        kpis += [
            {"code": "RMSE", "value": _num(rf["rmse"]), "dec": 1, "unit": "vuelos", "label": f"error típico del {rf['main_model']} en motores nunca vistos"},
            {"code": "COB", "value": _num(rf["coverage_aci"]), "dec": 1, "unit": "%", "label": f"de los vuelos con el RUL real dentro del intervalo (objetivo {rf['target']})"},
            {"code": "SOB", "value": _num(rf["miss_over_aci"]), "dec": 1, "unit": "%", "label": "de los vuelos con la cota inferior por encima del RUL real"},
        ]
    R["kpis"] = kpis

    # Conclusiones finales, redactadas con los números de cada etapa
    concl = []
    if eda:
        F = eda["facts"]
        concl.append({"title": "El desgaste existe, pero hay que destaparlo",
                      "text": f"Las condiciones de vuelo explican el {F['med_r2']} de cada sensor. Descontándolas con un modelo de motor sano, "
                              f"la correlación de T48 con el RUL pasa de {F['norm_rho_raw']} a {F['norm_rho_res']}."})
    if res:
        mt = {(r["modelo"], r["intervalo"]): r for r in res["metrics_table"]}
        models = list(dict.fromkeys(r["modelo"] for r in res["metrics_table"]))
        raw = {m: mt[(m, "Cuantiles del modelo")] for m in models}
        best = min(models, key=lambda m: raw[m]["RMSE"])
        txt = " · ".join(f"{m}: RMSE {num(raw[m]['RMSE'], 2)}, NASA {num(raw[m]['score NASA'], 3)}" for m in models)
        concl.append({"title": f"{best} predice mejor el RUL puntual" if len(models) > 1 else "Predicción del RUL",
                      "text": f"En los motores de prueba oficiales: {txt}. Cerca de la falla el error es de pocos vuelos."})
        rf = res["facts"]
        concl.append({"title": "La incertidumbre hay que calibrarla",
                      "text": f"Los cuantiles del modelo cubren el {rf['coverage_raw']} de los casos; con conformal adaptativo, el {rf['coverage_aci']} "
                              f"(objetivo {rf['target']}), con un ancho medio de {rf['width_aci']} vuelos."})
        concl.append({"title": "Sobreestimar deja de ser lo común",
                      "text": f"La mediana sobreestima en el {rf['over_point']} de los vuelos; la cota inferior del intervalo, sólo en el {rf['miss_over_aci']}. "
                              "Es el valor recomendable para programar inspecciones."})
    if l2.get("line2_table"):
        t = {r["variante"][0]: r for r in l2["line2_table"]}
        key = [k for k in t["A"] if k.startswith("RMSE")][-1]
        concl.append({"title": "Generalizar entre rutas no es gratis",
                      "text": f"Entrenando sólo con vuelos cortos, el RMSE en {key.replace('RMSE ', '')} pasa de {num(t['A'][key], 2)} a "
                              f"{num(t['C'][key], 2)} vuelos. Separar el efecto operativo es la clave (Línea 2)."})
    if l3.get("line3_table"):
        c = next(r for r in l3["line3_table"] if "CUSUM" in r["método"])
        concl.append({"title": "El desgaste se puede ver temprano",
                      "text": f"Un CUSUM sobre un índice de salud multivariado da la alarma {num(c['demora mediana [vuelos]'])} vuelos después del inicio real "
                              f"(mediana), con sólo el {num(c['desgaste ocurrido al alarmar [%]'], 1)} % del desgaste ocurrido (Línea 3)."})
    R["conclusions"] = concl
    R["generated"] = datetime.now().strftime("%d/%m/%Y %H:%M")
    R["mode"] = mode
    return R


def run(cfg: dict, paths: Paths, mode: str) -> Path:
    R = assemble(cfg, paths, mode)
    env = Environment(loader=FileSystemLoader(HERE / "templates"), autoescape=False)
    tpl = env.get_template("informe.html.j2")
    plotly_js = (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_text(encoding="utf-8")
    html = tpl.render(
        mode=mode, team=cfg["report"]["team"],
        plotly_js=plotly_js,
        report_json=json.dumps(R, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"),
        css=(ASSETS / "anatomia.css").read_text(encoding="utf-8") + "\n" + (ASSETS / "anatomia_p2.css").read_text(encoding="utf-8"),
        svg_defs=(ASSETS / "svg_defs.html").read_text(encoding="utf-8"),
        hero_engine=(ASSETS / "hero_engine.svg").read_text(encoding="utf-8"),
        app_js=(ASSETS / "anatomia.js").read_text(encoding="utf-8"),
    )
    out = paths.reports / ("informe_final.html" if mode == "full" else "informe_final_sample.html")
    out.write_text(html, encoding="utf-8")
    log.info("Informe: %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return out
