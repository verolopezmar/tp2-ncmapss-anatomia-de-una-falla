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
    R["models_chosen"] = (res.get("model_meta") or {}).get(res.get("main_model"), {}).get("model")
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
            {"code": "RMSE", "value": _num(rf["rmse"]), "dec": 1, "unit": "vuelos", "label": f"error típico ({rf['main_model']}) en motores nunca vistos"},
            {"code": "COB", "value": _num(rf["coverage_aci"]), "dec": 1, "unit": "%", "label": f"de los vuelos con el RUL real dentro del intervalo (objetivo {rf['target']})"},
            {"code": "SOB", "value": _num(rf["miss_over_aci"]), "dec": 1, "unit": "%", "label": "de los vuelos con la cota inferior por encima del RUL real"},
        ]
    R["kpis"] = kpis

    # Conclusiones finales, redactadas con los números de cada etapa
    concl = []
    if eda:
        F = eda["facts"]
        concl.append({"key": "residuos", "title": "El desgaste existe, pero hay que destaparlo",
                      "text": f"Las condiciones de vuelo explican el {F['med_r2']} de cada sensor. Descontándolas con un modelo de motor sano, "
                              f"la correlación de T48 con el RUL pasa de {F['norm_rho_raw']} a {F['norm_rho_res']}."})
    if res:
        mt = {(r["modelo"], r["intervalo"]): r for r in res["metrics_table"]}
        models = list(dict.fromkeys(r["modelo"] for r in res["metrics_table"]))
        raw = {m: mt[(m, "Cuantiles del modelo")] for m in models}
        best = min(models, key=lambda m: raw[m]["RMSE"])
        txt = " · ".join(f"{m}: RMSE {num(raw[m]['RMSE'], 2)}, NASA {num(raw[m]['score NASA'], 3)}" for m in models)
        concl.append({"key": "modelo", "title": f"{best} predice mejor el RUL puntual" if len(models) > 1 else "Predicción del RUL",
                      "text": f"En los motores de prueba oficiales: {txt}. Cerca de la falla el error es de pocos vuelos."})
        rf = res["facts"]
        concl.append({"key": "calibrar", "title": "La incertidumbre hay que calibrarla",
                      "text": f"Los cuantiles del modelo cubren el {rf['coverage_raw']} de los casos; con conformal adaptativo, el {rf['coverage_aci']} "
                              f"(objetivo {rf['target']}), con un ancho medio de {rf['width_aci']} vuelos."})
        concl.append({"key": "cota", "title": "Sobreestimar deja de ser lo común",
                      "text": f"La mediana sobreestima en el {rf['over_point']} de los vuelos; la cota inferior del intervalo, sólo en el {rf['miss_over_aci']}. "
                              "Es el valor recomendable para programar inspecciones."})
    if l2.get("line2_table"):
        t = {r["variante"][0]: r for r in l2["line2_table"]}
        key = [k for k in t["A"] if k.startswith("RMSE")][-1]
        dom, dat = t["C"][key] - t["E"][key], t["E"][key] - t["A"][key]
        concl.append({"key": "rutas", "title": "Generalizar entre rutas no es gratis",
                      "text": f"Entrenando sólo con vuelos cortos, el RMSE en {key.replace('RMSE ', '')} pasa de {num(t['A'][key], 2)} a "
                              f"{num(t['C'][key], 2)} vuelos; con la misma cantidad de motores de todas las clases sería {num(t['E'][key], 2)}. "
                              + ("La mayor parte de la brecha es por no haber visto ese tipo de vuelo (Línea 2)." if dom > dat else
                                 "La mayor parte de la brecha es por tener menos motores (Línea 2).")})
    if l3.get("line3_table"):
        c = next(r for r in l3["line3_table"] if "CUSUM" in r["método"])
        concl.append({"key": "alerta", "title": "El desgaste se puede ver temprano",
                      "text": f"Un CUSUM sobre un índice de salud multivariado da la alarma {num(c['demora mediana [vuelos]'])} vuelos después del inicio real "
                              f"(mediana), con sólo el {num(c['desgaste ocurrido al alarmar [%]'], 1)} % del desgaste ocurrido (Línea 3)."})
    R["conclusions"] = concl
    R["closing"] = [c for c in concl if c["key"] in ("residuos", "cota", "rutas", "alerta")]
    R["generated"] = datetime.now().strftime("%d/%m/%Y %H:%M")
    R["mode"] = mode
    return R


def presentation_facts(R: dict, cfg: dict) -> dict:
    """Cifras extra para la versión corta (presentación), calculadas desde los resultados."""
    f = {}
    mt = R.get("metrics_table") or []
    raw = {r["modelo"]: r for r in mt if r["intervalo"] == "Cuantiles del modelo"}
    for m, k in (("CatBoost", "cb"), ("TFT", "tft")):
        if m in raw:
            f[f"p_rmse_{k}"] = num(raw[m]["RMSE"], 2)
            f[f"p_nasa_{k}"] = num(raw[m]["score NASA"], 3)
    split = {r["rol"].split(" ")[0]: r for r in R.get("split_summary") or []}
    if split:
        f["p_train_units"] = str(split["Entrenamiento"]["motores"])
        f["p_train_flights"] = num(split["Entrenamiento"]["vuelos evaluados"])
        f["p_calib_units"] = str(split["Calibración"]["motores"])
    tft = (R.get("model_meta") or {}).get("tft")
    if tft:
        f["p_tft_params"] = num(tft["n_params"])
    eb = R["figures"].get("error_by_rul")
    if eb:   # error en el tramo más cercano a la falla, por modelo
        for tr in eb["data"]:
            k = "cb" if tr["name"] == "CatBoost" else "tft"
            f[f"p_rmse_near_{k}"] = num(tr["y"][0], 1)
        f["p_near_bucket"] = eb["data"][0]["x"][0]
    if R.get("line2_table"):
        t = {r["variante"][0]: r for r in R["line2_table"]}
        key = [k for k in t["A"] if k.startswith("RMSE")][-1]
        for v in "ABCDE":
            f[f"p_l2_{v}"] = num(t[v][key], 1)
        f["p_l2_class"] = key.replace("RMSE vuelos ", "")
        f["p_l2_units"] = str(t["C"]["motores de entrenamiento"])
        f["p_l2_units_all"] = str(t["A"]["motores de entrenamiento"])
        f["p_l2_data"] = num(t["E"][key] - t["A"][key], 1)
        f["p_l2_domain"] = num(t["C"][key] - t["E"][key], 1)
    if R.get("line3_table"):
        t = {r["método"]: r for r in R["line3_table"]}
        c, th, p = (next(v for k, v in t.items() if s in k) for s in ("CUSUM", "T48", "PELT"))
        f.update(p_l3_delay=num(c["demora mediana [vuelos]"]), p_l3_wear=num(c["desgaste ocurrido al alarmar [%]"], 1),
                 p_l3_lead=num(c["anticipación mediana a la falla [vuelos]"]), p_l3_fa=num(c["alarma antes del inicio real [%]"], 0),
                 p_l3_det=num(c["detectados [%]"], 0), p_l3_units=str(c["motores"]),
                 p_l3_t48_delay=num(th["demora mediana [vuelos]"]), p_l3_t48_fa=num(th["alarma antes del inicio real [%]"], 0),
                 p_l3_t48_det=num(th["detectados [%]"], 0), p_l3_pelt_fa=num(p["alarma antes del inicio real [%]"], 0),
                 p_l3_target_fa=f"{num(100 * cfg['line3']['target_false_alarm'])} %")
    return f


def _f(x) -> float:
    try:
        return _num(x)
    except (TypeError, ValueError):
        return float("nan")


def presentation_slides(R: dict) -> dict:
    """Puntos cortos (tipo diapositiva) de la versión para presentar, redactados desde los números."""
    F = R["facts"]
    S: dict = {}
    first = F["top_sensors"].split(",")[0]
    S["eda"] = [
        {"k": "Hallazgo 1", "title": "El vuelo tapa el desgaste",
         "text": f"T48 varía {F['t48_within']} dentro de un vuelo; el desgaste de toda la vida suma {F['t48_drift']}. "
                 f"Un modelo de motor sano explica el {F['med_r2']} de cada sensor → trabajamos con <b>residuos</b>."},
        {"k": "Hallazgo 2", "title": "Pocos sensores anticipan la falla",
         "text": f"Los residuos de <b>{F['top_sensors']}</b> son los que más siguen al desgaste. La mediana de {first} "
                 f"se sale de lo normal a {F['alarm_flights']} vuelos de la falla."},
        {"k": "Hallazgo 3", "title": "Cada motor dura distinto",
         "text": f"Entre {F['eol_min']} y {F['eol_max']} vuelos, y sólo el {F['frac_sano']} de la vida es sana "
                 "→ partición por motor completo y más incertidumbre lejos de la falla."},
        {"k": "Hallazgo 4", "title": "Vuelos cortos, medianos y largos",
         "text": f"Duran {F['fc_hours']}, usan sólo {F['n_profiles']} perfiles reales repetidos, y los motores de vuelos largos viven "
                 f"{F['life_long']} vuelos contra {F['life_short']} de los cortos → la clase de vuelo es una variable (y la Línea 2)."},
    ]
    if "p_rmse_cb" in F and "p_rmse_tft" in F:
        cb_wins = _f(F["p_rmse_cb"]) < _f(F["p_rmse_tft"])
        near_tft = _f(F.get("p_rmse_near_tft")) < _f(F.get("p_rmse_near_cb"))
        S["model_title"] = "CatBoost le ganó al TFT" if cb_wins else "El TFT le ganó a CatBoost"
        S["why_model"] = ([
            f"Pocos datos para un modelo grande: {F['p_train_units']} motores ({F['p_train_flights']} vuelos) para un TFT de {F['p_tft_params']} parámetros.",
            "Las variables ya resumen la historia (medias y pendientes de los últimos vuelos): leer la secuencia agrega poco.",
            "CatBoost es robusto con datos tabulares chicos y casi no necesita ajuste.",
        ] if cb_wins else [
            "Leer la secuencia de vuelos aporta información que las variables resumidas no capturan.",
        ]) + ([f"Pero cerca de la falla (RUL {F['p_near_bucket']}) el TFT es mejor: RMSE {F['p_rmse_near_tft']} contra {F['p_rmse_near_cb']}."]
              if cb_wins and near_tft else [])
    if "r_coverage_aci" in F:
        below = _f(F["r_coverage_aci"]) < _f(F["r_target"])
        S["why_cov_title"] = ("¿Por qué no llega al " + F["r_target"] + "?") if below else "Cobertura alcanzada"
        S["why_cov"] = ([
            f"Se calibra con sólo {F['p_calib_units']} motores.",
            "Los vuelos de un mismo motor están muy correlacionados: la muestra efectiva es mucho más chica que la cantidad de vuelos.",
            "La garantía conformal supone datos intercambiables; los motores de prueba no se comportan igual que los de calibración.",
            "El ACI corrige sobre la marcha, sobre todo en la cola peligrosa: la sobreestimación.",
        ] if below else [f"Con {F['p_calib_units']} motores de calibración se alcanza el objetivo."])
    if "p_l2_A" in F:
        dom_bigger = _f(F["p_l2_domain"]) > _f(F["p_l2_data"])
        d_helps = _f(F["p_l2_D"]) < _f(F["p_l2_C"])
        S["l2"] = [
            f"Con los {F['p_l2_units_all']} motores de todas las clases: RMSE <b>{F['p_l2_A']}</b> en vuelos {F['p_l2_class']}.",
            f"Control, {F['p_l2_units']} motores de todas las clases: <b>{F['p_l2_E']}</b> → {F['p_l2_data']} vuelos se deben a tener menos datos.",
            f"{F['p_l2_units']} motores de vuelos <b>cortos</b>: <b>{F['p_l2_C']}</b> → {F['p_l2_domain']} vuelos más por no haber visto vuelos {F['p_l2_class']}.",
            ("Lo que más pesa es el <b>tipo de vuelo</b>, no la cantidad de datos." if dom_bigger
             else "Lo que más pesa es la <b>cantidad de motores</b>, no el tipo de vuelo."),
            (f"Un modelo sano que vio todas las clases {'ayuda' if d_helps else 'no ayuda'} ({F['p_l2_D']} contra {F['p_l2_C']}): "
             + ("separar el efecto operativo es parte de la solución." if d_helps
                else "descontar el efecto operativo no alcanza; cambia cuánto desgaste produce cada vuelo.")),
        ]
    if "p_l3_delay" in F:
        better_t48 = _f(F["p_l3_delay"]) <= _f(F["p_l3_t48_delay"]) and _f(F["p_l3_fa"]) <= _f(F["p_l3_t48_fa"])
        S["l3"] = [
            "Índice de salud: distancia de los 14 residuos a la nube de vuelos sanos (un número por vuelo).",
            f"CUSUM online: acumula excesos y avisa al superar un umbral calibrado en motores de desarrollo para ≤ {F['p_l3_target_fa']} de falsas alarmas.",
            f"En {F['p_l3_units']} motores de prueba detecta el {F['p_l3_det']} %, con {F['p_l3_fa']} % de alarmas antes del inicio real.",
            (f"Mejor que mirar sólo T48: {F['p_l3_t48_delay']} vuelos de demora, {F['p_l3_t48_fa']} % de falsas alarmas y {F['p_l3_t48_det']} % detectados."
             if better_t48 else f"Comparado con T48 sola: {F['p_l3_t48_delay']} vuelos de demora y {F['p_l3_t48_fa']} % de falsas alarmas."),
        ]
    S["l2_future"] = [
        {"k": "DANN", "title": "Domain-Adversarial NN",
         "text": "Predecir el RUL mientras un discriminador intenta adivinar la clase de vuelo: rasgos de desgaste que no delatan la ruta."},
        {"k": "PATCHTST", "title": "Señal segundo a segundo",
         "text": "Transformer sobre parches de la señal cruda de cada vuelo, en lugar de promedios: capta los transitorios de cada régimen."},
        {"k": "TRANSFER", "title": "Pocos motores nuevos",
         "text": "Preentrenar con vuelos cortos y ajustar con pocos motores de vuelos largos: ¿cuántos hacen falta?"},
    ]
    S["l3_future"] = [
        {"k": "SELF-SUPERVISED", "title": "Transformers sin etiquetas",
         "text": "Aprender qué es un vuelo normal con millones de segundos, sin usar el RUL ni el estado de salud."},
        {"k": "MAE", "title": "Masked Autoencoders",
         "text": "Ocultar partes de la señal y reconstruirlas: el error de reconstrucción es un índice de salud aprendido."},
        {"k": "CPD BAYESIANO", "title": "Alarmas con probabilidad",
         "text": "Detección de cambios online que dé la probabilidad de que el desgaste ya empezó, en vez de un sí o no."},
    ]
    S["limits"] = [
        "<b>Datos simulados</b>: C-MAPSS con vuelos reales; habría que validarlo con datos de operación.",
        f"<b>Pocos motores</b>: {F['n_units']} en total y {F.get('p_calib_units', '—')} para calibrar; la garantía conformal es aproximada.",
        f"<b>Un archivo dañado</b>: {F['bad_ds']} viene truncado en Kaggle y quedó afuera.",
    ]
    return S


def _render(template: str, R: dict, cfg: dict, mode: str) -> str:
    env = Environment(loader=FileSystemLoader(HERE / "templates"), autoescape=False)
    plotly_js = (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_text(encoding="utf-8")
    return env.get_template(template).render(
        mode=mode, team=cfg["report"]["team"], presenters=cfg["report"]["presenters"], lines=cfg["report"]["professor_lines"],
        S=presentation_slides(R), F=R["facts"],
        plotly_js=plotly_js,
        report_json=json.dumps(R, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"),
        css=(ASSETS / "anatomia.css").read_text(encoding="utf-8") + "\n" + (ASSETS / "anatomia_p2.css").read_text(encoding="utf-8"),
        svg_defs=(ASSETS / "svg_defs.html").read_text(encoding="utf-8"),
        hero_engine=(ASSETS / "hero_engine.svg").read_text(encoding="utf-8"),
        app_js=(ASSETS / "anatomia.js").read_text(encoding="utf-8"),
    )


def run(cfg: dict, paths: Paths, mode: str) -> list[Path]:
    """Genera la versión corta para presentar (informe_final) y la completa como anexo (informe_completo)."""
    R = assemble(cfg, paths, mode)
    R["facts"].update(presentation_facts(R, cfg))
    sfx = "" if mode == "full" else "_sample"
    outs = []
    for template, name in (("presentacion.html.j2", f"informe_final{sfx}.html"), ("informe.html.j2", f"informe_completo{sfx}.html")):
        out = paths.reports / name
        out.write_text(_render(template, R, cfg, mode), encoding="utf-8")
        log.info("Informe: %s (%.1f MB)", out, out.stat().st_size / 1e6)
        outs.append(out)
    return outs
