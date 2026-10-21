# Anatomía de una Falla · Parte II

**Cuánto le queda y con qué certeza.** Laboratorio II · TP2.

Análisis exploratorio del dataset **NASA N-CMAPSS** (motores turbofan simulados hasta la falla con vuelos reales) y
predicción de la **vida útil remanente (RUL) con intervalos de incertidumbre** que controlan el riesgo de sobreestimarla.

- **Informe final:** [`reports/informe_final.html`](reports/informe_final.html). Descargalo y abrilo en el navegador: funciona sin conexión.
- **Parte I (presentada antes):** [`compartir.html`](compartir.html).

Integrantes: López Marsico Verónica · Acciardi Juan · Naón Pedro · Giannice Agustín.

---

## Qué hay en este trabajo

| Pedido de la devolución | Dónde se resuelve |
|---|---|
| Orden claro | Un pipeline por etapas (`scripts/run_all.py`), un módulo por etapa en `src/ncmapss/` y un informe con el mismo recorrido |
| EDA real | `src/ncmapss/eda.py` + `notebooks/01_eda.ipynb`: estructura, calidad (faltantes, infinitos, rangos, 8 verificaciones), perfiles de vuelo, evolución de sensores, correlaciones, componentes, RUL, atípicos y **decisiones de modelado** justificadas |
| Nada hardcodeado | Todos los números, tablas, gráficos y lecturas se calculan desde los datos; los parámetros están en `config.yaml`; el HTML lo arma un script |
| Línea 1 (principal) | CatBoost (baseline) + Temporal Fusion Transformer + Adaptive Conformal Prediction · `src/ncmapss/models/`, `src/ncmapss/uncertainty.py` |
| Líneas 2 y 3 | Trabajo futuro fundamentado, con un experimento simple de cada una · `line2_transfer.py`, `line3_changepoint.py` |

## Estructura

```
config.yaml              todos los parámetros del análisis (semilla, umbrales, modelos, conformal…)
.env.example             plantilla para indicar dónde están los datos (NCMAPSS_DATA_DIR)
requirements.txt
scripts/
  run_all.py             corre el pipeline completo o por etapas (--sample = modo rápido)
  build_notebooks.py     genera (y opcionalmente ejecuta) los notebooks
src/ncmapss/
  io.py · prepare.py     lectura por bloques de los HDF5, integridad, agregados por vuelo
  health.py              modelo de "motor sano" y residuos (medido − esperado)
  splits.py · features.py   partición por motor y variables que sólo miran hacia atrás
  eda.py                 análisis exploratorio completo → outputs/<modo>/eda.json
  models/                catboost_model.py · tft_model.py
  metrics.py · uncertainty.py   métricas, CQR asimétrico, CQR por tramos y ACI de flota
  line2_transfer.py · line3_changepoint.py   experimentos de las líneas 2 y 3
  report/                plantilla, estilos (los de compartir.html) y build.py del informe
notebooks/
  01_eda.ipynb · 02_rul_incertidumbre.ipynb · 03_lineas_futuras.ipynb · colab_tft.ipynb
outputs/full/            resultados en JSON (versionados, permiten regenerar el informe sin los datos)
reports/informe_final.html
```

## 1 · Instalación

Python 3.11–3.13.

```bash
git clone https://github.com/verolopezmar/tp2-ncmapss-anatomia-de-una-falla.git
cd tp2-ncmapss-anatomia-de-una-falla
python -m venv .venv
# Windows: .venv\Scripts\activate     ·     Linux/Mac: source .venv/bin/activate

# PyTorch: con GPU NVIDIA (CUDA 12.8, sirve para las RTX 50xx y anteriores)
pip install torch --index-url https://download.pytorch.org/whl/cu128
# …o sin GPU:  pip install torch

pip install -r requirements.txt
```

> **Tip OneDrive / Dropbox:** si el repo está en una carpeta sincronizada, conviene crear el entorno virtual **fuera**
> (por ejemplo `python -m venv C:\datos\venv-ncmapss`) para no sincronizar varios GB.

## 2 · Datos (no están en el repo)

El dataset pesa ~29 GB y GitHub no acepta archivos de más de 100 MB.

1. Descargarlo de Kaggle: <https://www.kaggle.com/datasets/bishals098/nasa-cmapss-2-engine-degradation>
   (botón *Download*, o con la CLI de Kaggle: `kaggle datasets download -d bishals098/nasa-cmapss-2-engine-degradation`).
2. Descomprimirlo en una carpeta **fuera del repo y fuera de OneDrive**, por ejemplo `C:\datos\ncmapss`. Tienen que quedar
   los archivos `N-CMAPSS_DS01-005.h5`, `N-CMAPSS_DS02-006.h5`, …, `N-CMAPSS_DS08d-010.h5`.
3. Copiar `.env.example` como `.env` y poner **tu** ruta:
   ```
   NCMAPSS_DATA_DIR=C:\datos\ncmapss
   ```
   Si no hay `.env`, se usa `data.default_dir` de `config.yaml`. Los datos procesados se guardan en `<NCMAPSS_DATA_DIR>/processed/`.

> **Nunca subas credenciales:** `kaggle.json`, `.env` y cualquier token están en `.gitignore`.
>
> **DS08d:** el archivo `N-CMAPSS_DS08d-010.h5` de Kaggle viene truncado y no se puede abrir. El pipeline lo detecta,
> lo excluye y lo informa (también en el informe). Se usan los otros 9 subconjuntos.

## 3 · Cómo correr todo

```bash
python scripts/run_all.py --sample     # modo rápido: 2 subconjuntos y 4 motores por partición, unos 2 minutos
python scripts/run_all.py              # versión completa
```

El modo rápido sirve para probar que todo funciona; sus resultados se guardan aparte (`outputs/sample/`,
`reports/informe_final_sample.html`) y no pisan los de la versión completa.

Etapas (se pueden correr sueltas con `--only` o saltear con `--skip`):

| etapa | qué hace | tiempo aprox.* |
|---|---|---|
| `prepare` | integridad, calidad y agregados por vuelo (pasada 1 sobre los .h5) | 40 s |
| `health` | modelo de motor sano (GPU si hay) | 20 s |
| `residuals` | residuos por vuelo (pasada 2 sobre los .h5) | 50 s |
| `features` | tabla de modelado y partición por motor | 2 s |
| `eda` | figuras, tablas y lecturas del EDA | 5 s |
| `catboost` | baseline con cuantiles | 20 s |
| `tft` | Temporal Fusion Transformer con búsqueda de 8 configuraciones | 15–20 min con GPU |
| `conformal` | intervalos CQR / ACI y métricas | 2 s |
| `line2` · `line3` | experimentos de las líneas 2 y 3 | 2 min · 2 s |
| `report` | arma `reports/informe_final.html` | 1 s |

\* Medido con un Intel Core Ultra 9 y una RTX 5070 Ti. Sin GPU el TFT tarda bastante más: ver Colab abajo.

Ejemplos:

```bash
python scripts/run_all.py --only eda report        # rehacer el EDA y el informe
python scripts/run_all.py --skip tft               # todo menos el TFT (p. ej. si se entrena en Colab)
python scripts/run_all.py --only report            # sólo regenerar el HTML con los resultados de outputs/full
```

### Notebooks

```bash
python scripts/build_notebooks.py --execute   # regenera y ejecuta 01, 02 y 03 con los resultados guardados
```

Los notebooks no calculan nada a mano: muestran las figuras, tablas y lecturas que deja el pipeline, con la explicación
de cada análisis.

### Entrenar el TFT en Google Colab (sin GPU local)

1. Localmente: `python scripts/run_all.py --skip tft` (genera todo lo demás y `model_table.parquet` + `features.json`
   en `<NCMAPSS_DATA_DIR>/processed/full/`).
2. Abrir `notebooks/colab_tft.ipynb` en Colab con entorno **GPU**, subir esos dos archivos y correr las celdas.
3. Descargar `preds_tft.parquet` y `model_tft.json`, copiarlos a `outputs/full/` y correr
   `python scripts/run_all.py --only conformal report`. El informe registra dónde se entrenó cada modelo.

## Metodología en breve

1. **Residuos:** una red chica (*modelo de motor sano*) predice cada sensor desde altitud, Mach, potencia y temperatura de
   entrada, entrenada sólo con vuelos sanos de motores de entrenamiento. Residuo = medido − esperado.
2. **Partición sin fuga:** test = motores de prueba oficiales de NASA; los de desarrollo se dividen **por motor** en
   entrenamiento, validación y calibración.
3. **Variables que sólo miran hacia atrás:** residuo menos la línea base de cada motor, medias y pendientes móviles,
   número de vuelo, clase de vuelo y condiciones del vuelo.
4. **Modelos:** CatBoost multi-cuantil y TFT (pytorch-forecasting) con cuantiles 5 / 50 / 95 %. El RUL pasado nunca es
   entrada y se normaliza de forma global (normalizar por motor revelaría su vida total).
5. **Incertidumbre:** CQR asimétrico (menos tolerancia a sobreestimar), CQR por tramos de RUL predicho y
   **Adaptive Conformal Inference** a nivel flota (α se ajusta cada vez que un motor falla y se conoce su RUL real).
6. **Métricas:** RMSE, MAE, score asimétrico de NASA, cobertura, ancho del intervalo y % de sobreestimación
   (de la mediana y de la cota inferior).

## Reproducibilidad

Semilla fija (`seed` en `config.yaml`), versiones fijadas en `requirements.txt`, partición determinística y muestras
reproducibles. Los resultados de `outputs/full/*.json` corresponden a la última corrida completa; el informe muestra la
fecha y el hardware de cada entrenamiento.

## Fuentes

- Arias Chao, Kulkarni, Goebel y Fink (2021). *Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics.* Data, 6(1), 5.
- Lim, Arık, Loeff y Pfister (2021). *Temporal Fusion Transformers for interpretable multi-horizon time series forecasting.*
- Romano, Patterson y Candès (2019). *Conformalized Quantile Regression.*
- Gibbs y Candès (2021). *Adaptive Conformal Inference Under Distribution Shift.*
- Prokhorenkova et al. (2018). *CatBoost: unbiased boosting with categorical features.*
