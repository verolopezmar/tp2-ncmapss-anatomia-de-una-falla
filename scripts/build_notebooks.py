"""Genera los notebooks del proyecto (texto + celdas) y opcionalmente los ejecuta.

    python scripts/build_notebooks.py            # escribe los .ipynb
    python scripts/build_notebooks.py --execute  # además los ejecuta (requiere haber corrido run_all.py)

Los notebooks no calculan nada a mano: leen lo que dejó el pipeline en outputs/ y lo muestran
con su explicación. Así el informe HTML y los notebooks muestran exactamente los mismos números.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

SETUP = """import sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import plotly.io as pio
pio.renderers.default = "notebook_connected"
from src.ncmapss import nb
MODE = "full"          # "sample" si corriste el pipeline con --sample"""


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def eda_notebook():
    c = [md("""
# Anatomía de una Falla · Parte II — 01 · Análisis exploratorio (EDA)

Este notebook recorre el dataset **NASA N-CMAPSS** antes de modelar. Cada sección explica **qué se mira y por qué**;
los números, tablas y gráficos se calculan desde los datos en `src/ncmapss/eda.py` (etapa `eda` de `scripts/run_all.py`)
y se leen de `outputs/<modo>/eda.json`. No hay valores escritos a mano: si cambian los datos o la configuración,
cambian las lecturas.

**Cómo regenerarlo:** `python scripts/run_all.py --only prepare health residuals features eda` (o `--sample` para la versión rápida).
"""), code(SETUP), code("""
# Opcional: recalcular el EDA desde los datos procesados (unos segundos)
# from src.ncmapss import eda; from src.ncmapss.config import load_config, get_paths
# eda.run(load_config(), get_paths(load_config(), MODE))
E = nb.load("eda", MODE)
print("EDA generado el", E["generated"], "· modo", E["mode"])
nb.facts()
"""),
         md("""
## 1 · Estructura del dataset

N-CMAPSS tiene **un registro por segundo de vuelo**. Cada archivo `N-CMAPSS_DSxx.h5` es un *escenario de falla*
(qué componentes se degradan) y trae dos particiones: **desarrollo** (para entrenar) y **prueba** (motores nuevos).
Las variables vienen en seis bloques:

| bloque | contenido | ¿existe en un avión real? |
|---|---|---|
| `W` | condiciones de vuelo: altitud, Mach, potencia (TRA), temperatura de entrada (T2) | sí |
| `X_s` | 14 sensores físicos: temperaturas, presiones, giros, combustible | sí |
| `X_v` | 14 sensores virtuales del simulador | no |
| `T` | 10 parámetros de salud (eficiencia y flujo de cada componente) | no |
| `Y` | RUL: vuelos que le quedan al motor | es lo que se predice |
| `A` | motor, n° de vuelo, clase de vuelo (Fc) y estado de salud (hs) | sí (salvo hs) |
"""), code("nb.table('sizes')"), code("nb.show('sizes')"),
         md("""
### Calidad de los datos: faltantes, infinitos y consistencia

Se revisan **todos los segundos** (no una muestra): nulos, infinitos, rangos físicos y reglas que el dataset debería
cumplir (RUL = vida total − n° de vuelo, vuelos numerados sin huecos, clase de vuelo fija por motor, estado de salud
que nunca vuelve a "sano"). También se chequea la integridad de cada archivo: uno de los `.h5` de Kaggle está truncado.
"""), code("nb.checks()"), code("nb.table('missing')"),
         md("La **variación relativa** (desvío / media) muestra cuánto se mueve cada variable. Que varíe poco no significa que no sirva."),
         code("nb.show('quality')"),
         md("""
## 2 · Perfiles de vuelo y condiciones operativas

Cada ciclo es un **vuelo completo** con condiciones que cambian segundo a segundo. Los motores se agrupan en tres
**clases de vuelo**: corto (Fc = 1), mediano (Fc = 2) y largo (Fc = 3). Miramos un vuelo, la envolvente de operación,
la duración de cada clase y la forma típica de cada perfil.
"""), code("nb.show('profile')"), code("nb.show('envelope')"), code("nb.show('flight_class')\nnb.table('flight_class')"),
         code("nb.show('flight_shapes')"),
         md("""
## 3 · Contexto: separar el vuelo del desgaste

La pregunta clave: ¿cuánto de lo que mide un sensor depende de **cómo se vuela** y cuánto del **desgaste**?
Se entrena un *modelo de motor sano* (red neuronal chica) que predice cada sensor desde las 4 condiciones de vuelo,
usando sólo vuelos sanos de los motores de entrenamiento. Lo medido menos lo esperado es el **residuo**.
"""), code("nb.show('context')"), code("nb.show('variance_explained')"), code("nb.show('corr')"),
         code("nb.show('normalization')"),
         md("""
### Los perfiles de vuelo se repiten

El simulador reutiliza un conjunto finito de **vuelos reales grabados**. Eso deja en el residuo un sesgo propio de
cada perfil, que conviene conocer antes de modelar (y que es central para la Línea 2: generalizar entre tipos de vuelo).
"""), code("nb.show('profiles')"),
         md("""
## 4 · Qué componentes se desgastan

Los parámetros de salud (`T`) no se miden en la realidad, pero en la simulación muestran **qué se daña** en cada
escenario y **cómo evoluciona** el daño. El estado `hs` marca cuándo empieza la degradación anormal.
"""), code("nb.show('health_heat')\nnb.table('health_heat')"), code("nb.show('health_traj')"), code("nb.show('hs')"),
         md("""
## 5 · Evolución de los sensores y sensores más informativos

Ya sin el efecto del vuelo: qué sensores cambian al acercarse la falla (correlación de Spearman con la cercanía a la
falla), cómo evolucionan a lo largo de la vida, si hay valores atípicos y cómo se ve el estado del motor resumido en dos ejes.
"""), code("nb.show('rul_corr')"), code("nb.show('sensor_trends')"), code("nb.show('life_evolution')"),
         code("nb.show('outliers')\nnb.table('outliers')"), code("nb.show('pca')"),
         md("## 6 · Vida útil y distribución del RUL"),
         code("nb.show('lifespan')\nnb.table('lifespan')"), code("nb.show('life_vs_fc')"), code("nb.show('rul_dist')"),
         md("""
## 7 · Conclusiones del EDA → decisiones de modelado

Cada decisión del modelado (notebooks `02_rul_incertidumbre.ipynb` y `04_modelo_final.ipynb`) responde a un hallazgo de este análisis:
"""), code("import pandas as pd\npd.set_option('display.max_colwidth', None)\nnb.decisions()"),
         code("pd.DataFrame(E['takeaways'])")]
    return c


def rul_notebook():
    return [md("""
# Anatomía de una Falla · Parte II — 02 · RUL con incertidumbre (Línea 1, etapa 1)

> **Etapa 1, reemplazada por el modelo final.** Acá se comparan CatBoost y el Temporal Fusion Transformer calibrados con
> sólo 11 motores. Se eligió CatBoost (mejor en validación y en prueba) y después se recalibró con los 60 motores de
> desarrollo mediante CV+: ese modelo final está en **`04_modelo_final.ipynb`**. Las cifras de esta etapa son la línea
> base contra la que se compara el modelo final.

**Pregunta:** ¿cuántos vuelos le quedan a un motor, y con qué margen de error, controlando especialmente el riesgo de
**sobreestimar** su vida útil?

**Modelos:** CatBoost (regresión por cuantiles) y Temporal Fusion Transformer, ambos con intervalos conformales (CQR y
**Adaptive Conformal Inference**) calibrados en 11 motores que ningún modelo vio.

Los resultados se generan con `python scripts/run_all.py --only catboost tft conformal` y se leen de `outputs/<modo>/`.
"""), code(SETUP), code("""
R = nb.load("results", MODE)
import pandas as pd
pd.set_option('display.max_colwidth', None)
pd.DataFrame(R["split_summary"])
"""), md("""
## Metodología

1. **Partición por motor** (sin fuga): test = motores de prueba oficiales; los de desarrollo se dividen en
   entrenamiento, validación (early stopping) y calibración (intervalos).
2. **Variables** (sólo hacia atrás): residuos del vuelo, residuo menos la línea base del motor, medias y pendientes
   móviles, número de vuelo, clase y descriptores del vuelo.
3. **Modelos** con salida por cuantiles (5 %, 50 %, 95 %).
4. **Conformal**: se corrigen los cuantiles con los errores en motores de calibración (CQR) y luego se adapta el nivel
   con **ACI** a medida que los motores de la flota fallan y se conoce su RUL real. La cola de **sobreestimación** recibe
   menos tolerancia que la de subestimación.
5. **Métricas**: RMSE, MAE, score asimétrico de NASA, cobertura y ancho de los intervalos y % de sobreestimación.
"""), code("pd.DataFrame(R['metrics_table'])"), code("nb.show('pred_vs_true', R)"), code("nb.show('coverage_by_rul', R)"),
        code("nb.show('aci_trace', R)"), code("nb.show('engine_examples', R)"), code("nb.show('error_by_rul', R)"),
        code("nb.show('feature_importance', R)"),
        md("## Lectura de los resultados"), code("for t in R['discussion']: nb.html('<p>'+t+'</p>')")]


def future_notebook():
    return [md("""
# Anatomía de una Falla · Parte II — 03 · Líneas 2 y 3 (experimentos exploratorios)

- **Línea 2 · Generalizar entre tipos de vuelo:** ¿un modelo entrenado sólo con vuelos cortos estima bien la degradación
  en vuelos medianos y largos? ¿Ayuda descontar el efecto operativo (residuos) frente a usar sensores crudos?
- **Línea 3 · Detección temprana:** ¿se puede detectar el inicio de la degradación (cambio de régimen) antes de que la
  vida útil caiga, sólo con los sensores? Se compara contra el inicio real (`hs`).
"""), code(SETUP), code("L2 = nb.load('line2', MODE)\nL3 = nb.load('line3', MODE)\nimport pandas as pd\npd.set_option('display.max_colwidth', None)"),
        md("## Línea 2 · De vuelos cortos a vuelos largos"), code("pd.DataFrame(L2['line2_table'])"),
        code("nb.show('line2', L2)"),
        md("## Línea 3 · Detección temprana con change points"), code("pd.DataFrame(L3['line3_table'])"),
        code("nb.show('line3_delay', L3)"), code("nb.show('line3_example', L3)"),
        md("## Lectura y trabajo futuro"), code("for t in L2['future_discussion_l2'] + L3['future_discussion_l3']: nb.html('<p>'+t+'</p>')")]


def final_notebook():
    return [md("""
# Anatomía de una Falla · Parte II — 04 · Modelo final (CatBoost con CV+, ACI de flota y política de inspección)

Este es el **modelo final** del proyecto y el que respalda el abstract del congreso (`paper_phm2027/main.tex`).
Reemplaza a la etapa 1 (notebook `02`, con 11 motores de calibración).

**Cómo regenerarlo:** `python scripts/run_all.py --workers 1 --only cv_fit cv_eval entrega`. Este notebook sólo lee
`outputs/full/cvplus*.json` y `cvplus_intervals.parquet`: no necesita los datos crudos.

## Metodología final

1. **Modelo de motor sano y residuos.** Una red chica predice cada sensor desde altitud, Mach, potencia y temperatura de
   entrada, entrenada sólo con vuelos sanos. Residuo = medido − esperado.
2. **Variables que sólo miran hacia atrás:** residuo menos la línea base del motor, medias y pendientes móviles (5 y 10
   vuelos), número de vuelo, clase y condiciones del vuelo.
3. **CatBoost multicuantil (5 / 50 / 95 %)** con validación cruzada **por motor**: los 60 motores de desarrollo se
   dividen en 5 grupos estratificados por subconjunto. Para cada grupo se reentrena **todo** (modelo sano, residuos y
   CatBoost) sin esos motores, y se predice sobre ellos. Así los 60 motores aportan errores fuera de muestra para
   calibrar (antes eran 11). Las iteraciones (200, comunes a los grupos) se eligen por pérdida pinball en desarrollo.
4. **Intervalos CV+ asimétricos.** De un 10 % de error permitido, 3 % para sobreestimar la vida (la cola peligrosa) y
   7 % para subestimarla.
5. **ACI de flota (variante online).** Los motores de prueba se recorren en un orden de llegada; cuando cada uno falla y
   se conoce su vida real, el nivel de cada cola se corrige según cuántas veces su historia quedó fuera del intervalo.
   Se promedia sobre 30 órdenes aleatorios.
6. **Política de inspección.** Se ordena inspeccionar en el primer vuelo en que la cota inferior CV+ (sin ACI) cae por
   debajo de un umbral H. Si ejecutarla lleva τ = 5 vuelos (supuesto), es tardía cuando quedan menos. H se elige en
   desarrollo para no superar 5 % de tardías.

Todas las reglas (método, iteraciones, umbral) se eligen con motores de desarrollo; los 39 motores de prueba oficiales
se usan para evaluar. Los intervalos de confianza remuestrean **motores** (5000 réplicas).
"""), code(SETUP), code("""
import json
import pandas as pd
import plotly.graph_objects as go
from src.ncmapss.report import entrega as E
pd.set_option('display.max_colwidth', None)
R = nb.load("cvplus", MODE)
DEC = nb.load("cvplus_decision", MODE)
FIT = nb.load("cvplus_fit", MODE)
print("Motores de desarrollo:", R["units_dev"], "· de prueba:", R["units_test"], "· método elegido:", R["chosen"])
"""), md("""
## 1 · Decisión en desarrollo

La regla se fijó antes de mirar la prueba: tolerancia a sobreestimar, cercanía a la cobertura objetivo y menor ancho.
CV+ global cumple la tolerancia de 3 % y queda en 90 % de cobertura; la variante por tramos se descarta.
"""), code("""
cols = {"coverage": "cobertura", "miss_over": "sobreestima", "miss_under": "subestima", "width": "ancho [vuelos]", "rmse": "RMSE"}
pd.DataFrame(DEC["desarrollo"]).T[list(cols)].rename(columns=cols).round(3)
"""), md("""
## 2 · Cuántas iteraciones

Pérdida pinball fuera de grupo (sólo motores de desarrollo) según el número de iteraciones de CatBoost.
"""), code("""
c = FIT["curva_pinball"]
fig = go.Figure(go.Scatter(x=c["iter"], y=c["loss"], mode="lines+markers"))
fig.add_vline(x=FIT["iteraciones"], line_dash="dot", annotation_text=f"elegidas: {FIT['iteraciones']}")
fig.update_layout(height=320, xaxis_title="Iteraciones", yaxis_title="Pinball fuera de grupo", template="simple_white")
fig.show()
"""), md("""
## 3 · Resultados en los 39 motores de prueba

- **TFT / CatBoost · 11 motores:** etapa 1 (notebook `02`), calibrados con 11 motores (CQR por tramos, sin ACI).
- **CatBoost CV+ · 60 motores:** el modelo final, sin actualización online.
- **CatBoost CV+ y ACI:** el modelo final con la actualización de flota.

Para comparar con la etapa 1 en igualdad de condiciones: sin ACI, la línea base es «CatBoost · 11 motores»; con ACI,
la línea base con ACI está en `results.json` (sobreestimación 4,7 %, cobertura 88,2 %, ancho 18,8 vuelos).
"""), code("""
rows = []
for m, name in E.METHODS.items():
    t, ci = R["test"][m], R["ci"][m]
    rows.append({"método": name, "RMSE": t["rmse"], "score NASA": t["nasa"],
                 "cobertura [%]": 100 * t["coverage"], "IC cobertura": f"{100*ci['cov'][0]:.1f} – {100*ci['cov'][1]:.1f}",
                 "sobreestima [%]": 100 * t["miss_over"], "IC sobreestima": f"{100*ci['over'][0]:.1f} – {100*ci['over'][1]:.1f}",
                 "ancho [vuelos]": t["width"]})
res = nb.load("results", MODE)
b = next(r for r in res["metrics_table"] if r["modelo"] == "CatBoost" and r["intervalo"].startswith("ACI"))
rows.insert(2, {"método": "CatBoost · 11 motores y ACI", "RMSE": b["RMSE"], "score NASA": b["score NASA"],
                "cobertura [%]": b["cobertura [%]"], "IC cobertura": "—", "sobreestima [%]": b["% real < cota inferior"],
                "IC sobreestima": "—", "ancho [vuelos]": b["ancho medio [vuelos]"]})
pd.DataFrame(rows).round(2)
"""), md("""
### Diferencias pareadas (IC 95 %, remuestreando motores)

Si el intervalo no cruza el cero, la diferencia es clara con 39 motores.
"""), code("""
diff = []
for k, v in R["ci"].items():
    if " - " in k:
        for metric, d in v.items():
            diff.append({"comparación": k, "métrica": metric, "IC 95 % de la diferencia": f"{d['dif_ic95'][0]:.3f} a {d['dif_ic95'][1]:.3f}"})
pd.DataFrame(diff)
"""), code("""
go.Figure(E._ci_fig(R, "coverage", "cov", R["target"], "objetivo 90 %", "Cobertura en prueba [%]", [70, 100])).show()
go.Figure(E._ci_fig(R, "miss_over", "over", R["alpha_over"], "tolerancia 3 %", "Sobreestimación en prueba [%]", [0, 12])).show()
"""), md("## 4 · Por subconjunto, clase de vuelo y tramo de vida remanente (CV+)"), code("""
cols = {"rmse": "RMSE", "coverage": "cobertura", "miss_over": "sobreestima", "width": "ancho", "n": "vuelos"}
display(pd.DataFrame(R["per_ds"]).T[list(cols)].rename(columns=cols).round(3))
display(pd.DataFrame(R["per_fc"]).T[list(cols)].rename(columns=cols).rename(index={"1": "corto", "2": "mediano", "3": "largo"}).round(3))
pd.DataFrame(R["buckets"]).set_index("tramo")[list(cols)].rename(columns=cols).round(3)
"""), md("""
## 5 · Motor a motor

Cada predicción usa sólo la historia hasta ese vuelo. Cambiá `MOTOR` para ver otro motor de prueba.
"""), code("""
it = pd.read_parquet(nb.output_path("cvplus_intervals.parquet", MODE))
MOTOR = "DS02_u11"
g = it[it.uid == MOTOR].sort_values("cycle")
fig = go.Figure([
    go.Scatter(x=g.cycle, y=g.hi, mode="lines", line_width=0, showlegend=False, hoverinfo="skip"),
    go.Scatter(x=g.cycle, y=g.lo, mode="lines", fill="tonexty", name="intervalo CV+ (cota inferior abajo)"),
    go.Scatter(x=g.cycle, y=g.q50, mode="lines", name="predicción (mediana)"),
    go.Scatter(x=g.cycle, y=g.RUL, mode="lines", line_dash="dash", name="RUL real")])
fig.update_layout(height=380, xaxis_title="Número de vuelo", yaxis_title="Vuelos restantes", template="simple_white")
fig.show()
"""), md("""
## 6 · De la predicción a la inspección

Vida útil desperdiciada = vuelos que quedaban cuando se inspeccionó, menos τ. Se compara contra una vida fija por clase
de vuelo, elegida también en desarrollo con la misma regla de tardías. La política con la mediana se muestra porque
rinde parecido: la ganancia frente a la vida fija viene sobre todo de tener un pronóstico.
"""), code("""
rows = []
for tau, pol in R["inspection"]["by_tau"].items():
    for name in ("cota inferior", "mediana", "vida fija por clase"):
        r = pol[name]
        rows.append({"τ [vuelos]": int(tau), "política": name, "umbral H": r.get("H"), "tardías": f"{r['late_engines']} de {R['units_test']}",
                     "vida desperdiciada": round(r["waste"], 2), "IC 95 %": f"{r['ci']['waste'][0]:.1f} – {r['ci']['waste'][1]:.1f}"})
display(pd.DataFrame(rows))
go.Figure(E._policy_fig(R)).show()
"""), md("## 7 · Límites que hay que decir"), code("""
v1 = nb.load("cvplus_v1_early_stopping", MODE)
t, a, dev = R["test"]["cvplus"], R["test"]["cvplus_aci"], R["development"][R["chosen"]]
for s in [
    f"<b>La prueba se miró dos veces.</b> La primera versión (early stopping por grupo) dio score NASA {v1['test']['cvplus']['nasa']:.3f}; "
    f"después se pasó a {FIT['iteraciones']} iteraciones comunes elegidas sólo con pinball en desarrollo ({t['nasa']:.3f}). La regla no usa la "
    "prueba, pero la decisión de rediseñar vino después de verla.",
    f"<b>Sobrecobertura en prueba.</b> CV+ da {100*dev['coverage']:.1f} % en desarrollo y {100*t['coverage']:.1f} % en prueba: los motores de "
    f"prueba son más fáciles (RMSE {t['rmse']:.1f} contra {dev['rmse']:.1f}). ACI la reduce a {100*a['coverage']:.1f} %, a costa de subir la "
    f"sobreestimación de {100*t['miss_over']:.1f} % a {100*a['miss_over']:.1f} %.",
    "<b>ACI usa motores de prueba.</b> Es una actualización online: incorpora la vida real de cada motor de prueba cuando falla. "
    "No es una evaluación offline pura.",
    f"<b>Son {R['units_test']} motores.</b> Cada inspección tardía vale {100/R['units_test']:.1f} puntos; τ = 5 es un supuesto, no un dato de "
    "mantenimiento real. Los motores y su degradación son simulados; los perfiles de vuelo son reales.",
]:
    nb.html("<p>" + s + "</p>")
""")]


def colab_notebook():
    return [md("""
# Anatomía de una Falla · Parte II — Entrenar el TFT en Google Colab

Para quien no tenga GPU NVIDIA. El TFT sólo necesita **dos archivos procesados** (unos pocos MB), no los 29 GB del dataset:

- `model_table.parquet` y `features.json`, que genera localmente
  `python scripts/run_all.py --only prepare health residuals features`
  en `<NCMAPSS_DATA_DIR>/processed/full/`.

**Pasos:** `Entorno de ejecución → Cambiar tipo de entorno → GPU (T4)`, correr las celdas en orden, subir los dos archivos
cuando se pidan y al final descargar `preds_tft.parquet` y `model_tft.json`. Copiarlos a `outputs/full/` del repo local y correr
`python scripts/run_all.py --only conformal cv_eval report entrega`. El informe indica que el TFT se entrenó en Colab.
"""), code("!nvidia-smi --query-gpu=name,memory.total --format=csv"),
        code("""
!git clone -q https://github.com/verolopezmar/tp2-ncmapss-anatomia-de-una-falla.git repo
%cd repo
!pip install -q pytorch-forecasting==1.8.0 lightning==2.6.6 catboost==1.2.10 python-dotenv pyyaml
"""), code("""
import os, pathlib
os.environ["NCMAPSS_DATA_DIR"] = "/content/ncmapss_data"
dest = pathlib.Path("/content/ncmapss_data/processed/full"); dest.mkdir(parents=True, exist_ok=True)
from google.colab import files
up = files.upload()                       # elegir model_table.parquet y features.json
for name, data in up.items():
    (dest / name).write_bytes(data)
print(sorted(p.name for p in dest.iterdir()))
"""), code("!python scripts/run_all.py --only tft"),
        code("""
from google.colab import files
for f in ["outputs/full/preds_tft.parquet", "outputs/full/model_tft.json"]:
    files.download(f)
""")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    NB.mkdir(exist_ok=True)
    books = {"01_eda.ipynb": eda_notebook(), "02_rul_incertidumbre.ipynb": rul_notebook(),
             "03_lineas_futuras.ipynb": future_notebook(), "04_modelo_final.ipynb": final_notebook(),
             "colab_tft.ipynb": colab_notebook()}
    for name, cells in books.items():
        n = nbf.v4.new_notebook()
        n.cells = cells
        n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbf.write(n, NB / name)
        print("escrito", name)
    if a.execute:
        from nbconvert.preprocessors import ExecutePreprocessor
        for name in [b for b in books if b != "colab_tft.ipynb"]:
            n = nbf.read(NB / name, as_version=4)
            ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(n, {"metadata": {"path": str(NB)}})
            nbf.write(n, NB / name)
            print("ejecutado", name)


if __name__ == "__main__":
    sys.exit(main())
