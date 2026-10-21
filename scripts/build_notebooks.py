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

Cada decisión del modelado (notebook `02_rul_incertidumbre.ipynb`) responde a un hallazgo de este análisis:
"""), code("import pandas as pd\npd.set_option('display.max_colwidth', None)\nnb.decisions()"),
         code("pd.DataFrame(E['takeaways'])")]
    return c


def rul_notebook():
    return [md("""
# Anatomía de una Falla · Parte II — 02 · RUL con incertidumbre (Línea 1)

**Pregunta:** ¿cuántos vuelos le quedan a un motor, y con qué margen de error, controlando especialmente el riesgo de
**sobreestimar** su vida útil?

**Modelos:** CatBoost (baseline, regresión por cuantiles) y Temporal Fusion Transformer (modelo principal), ambos con
intervalos de **Adaptive Conformal Prediction** calibrados en motores que ningún modelo vio.

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
"""), code(SETUP), code("R = nb.load('results', MODE)"),
        md("## Línea 2 · De vuelos cortos a vuelos largos"), code("import pandas as pd\npd.DataFrame(R['line2_table'])"),
        code("nb.show('line2', R)"),
        md("## Línea 3 · Detección temprana con change points"), code("pd.DataFrame(R['line3_table'])"),
        code("nb.show('line3_delay', R)"), code("nb.show('line3_example', R)"),
        md("## Lectura"), code("for t in R['future_discussion']: nb.html('<p>'+t+'</p>')")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    NB.mkdir(exist_ok=True)
    books = {"01_eda.ipynb": eda_notebook(), "02_rul_incertidumbre.ipynb": rul_notebook(), "03_lineas_futuras.ipynb": future_notebook()}
    for name, cells in books.items():
        n = nbf.v4.new_notebook()
        n.cells = cells
        n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbf.write(n, NB / name)
        print("escrito", name)
    if a.execute:
        from nbconvert.preprocessors import ExecutePreprocessor
        for name in books:
            n = nbf.read(NB / name, as_version=4)
            ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(n, {"metadata": {"path": str(NB)}})
            nbf.write(n, NB / name)
            print("ejecutado", name)


if __name__ == "__main__":
    sys.exit(main())
