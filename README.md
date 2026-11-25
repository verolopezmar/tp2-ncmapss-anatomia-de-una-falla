# Anatomía de una Falla · Parte II

**Cuántos vuelos le quedan a un motor, con qué certeza y cuándo inspeccionarlo.** Laboratorio II · TP2 · Pontificia
Universidad Católica Argentina.

Predicción de la **vida útil remanente (RUL)** de motores turbofan del dataset **NASA N-CMAPSS** (motores simulados hasta
la falla bajo perfiles de vuelo reales), con **intervalos de incertidumbre que controlan el riesgo de sobreestimar la vida**
y una **política de inspección** derivada de esos intervalos. El trabajo dio origen a un abstract para la
*PHM Society Annual Conference 2027*.

---

## Empezá acá

| Si querés… | Abrí |
|---|---|
| Ver el **modelo final** y sus resultados en una página | [`reports/entrega_modelo.html`](reports/entrega_modelo.html) (descargalo y abrilo en el navegador; funciona sin conexión) |
| Recorrer el **modelo final** con código y tablas | [`notebooks/04_modelo_final.ipynb`](notebooks/04_modelo_final.ipynb) |
| Leer el **abstract para PHM 2027** | [`paper_phm2027/main.tex`](paper_phm2027/main.tex) (plantilla oficial de PHM Society) |
| Ver la **presentación** del TP (EDA, Línea 1, Líneas 2 y 3, cambios finales) | [`reports/informe_final.html`](reports/informe_final.html) |
| Ver el **EDA** completo | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) y el anexo [`reports/informe_completo.html`](reports/informe_completo.html) |
| Ver la **etapa 1** (CatBoost contra TFT con 11 motores de calibración) | [`notebooks/02_rul_incertidumbre.ipynb`](notebooks/02_rul_incertidumbre.ipynb) |
| Ver los **experimentos de las Líneas 2 y 3** | [`notebooks/03_lineas_futuras.ipynb`](notebooks/03_lineas_futuras.ipynb) |
| Ver la **Parte I** (presentada antes) | [`compartir.html`](compartir.html) |

> Las notebooks de GitHub muestran las tablas, pero las figuras interactivas (plotly) sólo se ven al abrirlas en Jupyter o
> VS Code con conexión. Los HTML de `reports/` traen todo incluido.

## Metodología final

```
vuelo (segundo a segundo) → modelo de motor sano → residuos → variables que sólo miran hacia atrás
   → CatBoost multicuantil (5 / 50 / 95 %) con validación cruzada por motor
   → intervalos CV+ asimétricos → [opcional] ACI de flota → política de inspección
```

1. **Residuos.** Una red chica (*modelo de motor sano*) predice cada sensor desde altitud, Mach, potencia y temperatura de
   entrada, entrenada sólo con vuelos sanos. Residuo = medido − esperado: separa el desgaste del efecto de cómo se vuela.
2. **Variables que sólo miran hacia atrás:** residuo menos la línea base de cada motor, medias y pendientes móviles (5 y
   10 vuelos), número de vuelo, clase de vuelo y condiciones del vuelo. El RUL pasado nunca es entrada.
3. **Partición sin fuga.** Prueba = los **39 motores de prueba oficiales** de NASA, intactos. Desarrollo = **60 motores**.
4. **CatBoost con validación cruzada por motor.** Los 60 motores de desarrollo se dividen en 5 grupos (estratificados por
   subconjunto). En cada grupo se reentrena **todo el pipeline** (modelo sano, residuos y CatBoost) sin esos motores y se
   predice sobre ellos. Así los 60 motores aportan errores fuera de muestra para calibrar, en lugar de 11 reservados. Las
   iteraciones (200, comunes) se eligen por pérdida pinball en desarrollo.
5. **Intervalos CV+ asimétricos.** Objetivo de 90 % de cobertura; del 10 % de error permitido, **3 %** para sobreestimar
   la vida (la cola peligrosa) y 7 % para subestimarla.
6. **ACI de flota (variante online).** Cuando cada motor falla y se conoce su vida real, el nivel de cada cola se corrige
   según cuántas veces su historia quedó fuera del intervalo (paso 0,05). Se promedia sobre 30 órdenes de llegada.
7. **Política de inspección.** Se ordena inspeccionar en el primer vuelo en que la cota inferior CV+ cae por debajo de un
   umbral H. Si ejecutarla lleva τ = 5 vuelos (supuesto nuestro), es tardía cuando quedan menos. H se elige en desarrollo
   para no superar 5 % de inspecciones tardías. Se compara contra una vida fija por clase de vuelo.

Método, iteraciones y umbral se eligen sólo con motores de desarrollo. Los intervalos de confianza remuestrean **motores**
(5000 réplicas). Código: [`src/ncmapss/cvplus.py`](src/ncmapss/cvplus.py). Parámetros: sección `cvplus` y `conformal` de
[`config.yaml`](config.yaml).

## Resultados finales (39 motores de prueba)

Fuente: [`outputs/full/cvplus.json`](outputs/full/cvplus.json) y [`outputs/full/results.json`](outputs/full/results.json),
corrida del 05/11/2026 (AMD Ryzen + RTX 5060, Python 3.13, torch 2.11).

| Método | RMSE [vuelos] | Cobertura | Cota inferior que sobreestima | Ancho medio [vuelos] |
|---|---|---|---|---|
| TFT · 11 motores de calibración (etapa 1) | 7,48 | 85,4 % | 6,6 % | 19,6 |
| CatBoost · 11 motores de calibración (etapa 1) | 6,51 | 87,1 % | 7,0 % | 18,2 |
| CatBoost · 11 motores + ACI de flota (etapa 1) | 6,51 | 88,2 % | 4,7 % | 18,8 |
| **CatBoost CV+ · 60 motores** (modelo final) | 6,61 | 95,8 % | 1,1 % | 25,8 |
| **CatBoost CV+ · 60 motores + ACI de flota** | 6,61 | 93,5 % | 1,9 % | 23,9 |

Cómo leerla (compará filas con el mismo tratamiento):

- **Sin ACI** (filas 2 y 4): la sobreestimación de la cota inferior baja de 7,0 % a 1,1 % (diferencia entre −9,1 y −3,0
  puntos, IC 95 %). El RMSE no cambia de forma significativa (diferencia entre −0,35 y +0,56 vuelos). La cobertura queda
  por encima del objetivo: los intervalos salen más anchos de lo necesario.
- **Con ACI** (filas 3 y 5): sobreestimación de 4,7 % a 1,9 % y cobertura de 88,2 % a 93,5 %, con intervalos 5 vuelos más anchos.
- **CatBoost contra TFT** (mismos motores, filas 1 y 2): el TFT tiene entre 0,36 y 1,56 vuelos más de RMSE (IC 95 %).
- **Política de inspección** (τ = 5): con la cota inferior CV+, 1 inspección tardía en 39 motores y 8,9 vuelos de vida
  desperdiciada por motor, contra 15,9 de la vida fija por clase (**44 % menos**, ahorro de 3,9 a 10,1 vuelos por motor,
  IC 95 %) con la misma cantidad de tardías. Una regla equivalente sobre la mediana rinde parecido (9,2 vuelos).

**Límites.** La prueba se miró dos veces (el cambio a 200 iteraciones comunes se decidió después de ver un primer
resultado; ver `outputs/full/cvplus_v1_early_stopping.json`). El ACI de flota usa la vida real de cada motor de prueba
cuando falla: es una evaluación online, no offline pura. Son 39 motores (cada tardía vale 2,6 puntos), τ = 5 es un
supuesto, DS07 es el subconjunto más débil (cobertura 82,1 %) y los motores y su degradación son simulados.

## Experimentos

| Experimento | Qué responde | Resultado principal | Dónde |
|---|---|---|---|
| Etapa 1 · CatBoost contra TFT | ¿Qué modelo usar? | CatBoost gana en validación (score NASA 0,992 contra 1,138) y en prueba | notebook 02 |
| Línea 2 · generalizar entre tipos de vuelo | ¿Un modelo entrenado con vuelos cortos sirve para vuelos largos? | No: RMSE 14,7 en vuelos largos contra 9,3 con la misma cantidad de motores de todas las clases | notebook 03 |
| Línea 3 · detección temprana | ¿Se detecta el inicio del desgaste sin usar el RUL? | Un CUSUM sobre un índice de salud alarma con 6 vuelos de demora mediana, con ~4 % del desgaste ocurrido (umbral calibrado para 10 % de falsas alarmas) | notebook 03 |
| Modelo final v1 (early stopping por grupo) | Primer intento de CV+ | Score NASA 0,719; reemplazado por 200 iteraciones comunes (0,704) | `outputs/full/cvplus_v1_early_stopping.json` |

## Cómo reproducir

### a) Sin los datos (sólo con lo versionado)

`outputs/full/` trae los JSON y los parquet livianos de la corrida final, así que se pueden regenerar informes y notebooks:

```bash
python scripts/run_all.py --only report entrega        # los tres HTML de reports/
python scripts/build_notebooks.py --execute            # regenera y ejecuta las notebooks 01–04
```

### b) Con los datos: todo el pipeline

```bash
python scripts/run_all.py --sample     # modo rápido: 2 subconjuntos y 4 motores por partición; no pisa la versión completa
python scripts/run_all.py --workers 1  # versión completa
```

| etapa | qué hace | tiempo aprox.* |
|---|---|---|
| `prepare` | integridad, calidad y agregados por vuelo (pasada 1 sobre los .h5) | 40 s |
| `health` | modelo de motor sano (GPU si hay) | 20 s |
| `residuals` | residuos por vuelo (pasada 2 sobre los .h5) | 50 s |
| `features` | tabla de modelado y partición por motor | 2 s |
| `eda` | figuras, tablas y lecturas del EDA | 5 s |
| `catboost` · `tft` | etapa 1: CatBoost y Temporal Fusion Transformer (búsqueda de 8 configuraciones) | 20 s · 15–20 min con GPU |
| `conformal` | etapa 1: intervalos CQR / ACI y métricas | 2 s |
| `line2` · `line3` | experimentos de las Líneas 2 y 3 | 2 min · 2 s |
| `cv_fit` · `cv_eval` | **modelo final**: CatBoost por grupos de motores, CV+, ACI de flota, bootstrap y política de inspección | 25 min · 10 s |
| `report` · `entrega` | arma los HTML de `reports/` | 1 s |

\* Tiempos medidos en una PC con GPU NVIDIA. Sin GPU el TFT tarda bastante más (ver Colab abajo). Con la misma semilla,
otra máquina puede dar cifras levemente distintas (por ejemplo, RMSE 6,50 contra 6,51).

Las etapas se pueden correr sueltas con `--only` o saltear con `--skip`. Con poca RAM (16 GB) usar `--workers 1`.

### c) Demo del modelo final sobre un motor

```bash
python scripts/predecir_motor.py --motor DS02_u11 --cada 10 --verificar
```

Parte del `.h5` crudo y corta la historia en cada vuelo pedido. Necesita los datos y los modelos de los 5 grupos
(`outputs/full/models/`, unos 54 MB, no versionados): se generan con `run_all.py --only cv_fit`.

### Entrenar el TFT en Google Colab (sin GPU local)

1. Localmente: `python scripts/run_all.py --skip tft`.
2. Abrir [`notebooks/colab_tft.ipynb`](notebooks/colab_tft.ipynb) en Colab con GPU, subir `model_table.parquet` y
   `features.json` (de `<NCMAPSS_DATA_DIR>/processed/full/`) y correr las celdas.
3. Copiar `preds_tft.parquet` y `model_tft.json` a `outputs/full/` y correr
   `python scripts/run_all.py --only conformal cv_eval report entrega`.

## Instalación

Python 3.12 o 3.13 (`numpy` y `scipy` fijados en `requirements.txt` no instalan en 3.11).

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

## Datos (no están en el repo)

El dataset pesa ~29 GB y GitHub no acepta archivos de más de 100 MB.

1. Descargarlo de Kaggle: <https://www.kaggle.com/datasets/bishals098/nasa-cmapss-2-engine-degradation>
   (o `kaggle datasets download -d bishals098/nasa-cmapss-2-engine-degradation`).
2. Descomprimirlo en una carpeta **fuera del repo y fuera de OneDrive**, por ejemplo `C:\datos\ncmapss`. Tienen que quedar
   los archivos `N-CMAPSS_DS01-005.h5`, `N-CMAPSS_DS02-006.h5`, …, `N-CMAPSS_DS08d-010.h5`.
3. Copiar `.env.example` como `.env` y poner **tu** ruta: `NCMAPSS_DATA_DIR=C:\datos\ncmapss`.
   Si no hay `.env`, se usa `data.default_dir` de `config.yaml`.

> **Nunca subas credenciales:** `kaggle.json`, `.env` y cualquier token están en `.gitignore`.
>
> **DS08d:** el archivo `N-CMAPSS_DS08d-010.h5` de Kaggle viene truncado y no se puede abrir. El pipeline lo detecta,
> lo excluye y lo informa. Se usan los otros 9 subconjuntos (60 motores de desarrollo y 39 de prueba).

## Estructura

```
config.yaml              todos los parámetros (semilla, umbrales, modelos, conformal, cvplus…)
requirements.txt · .env.example
scripts/
  run_all.py             pipeline completo o por etapas (--sample = modo rápido)
  build_notebooks.py     genera (y con --execute ejecuta) las notebooks
  predecir_motor.py      demo del modelo final sobre un motor
  compare_parte1.py      compara las cifras de la Parte I con el EDA actual (herramienta de la Parte I)
src/ncmapss/
  io.py · prepare.py     lectura por bloques de los HDF5, integridad, agregados por vuelo
  health.py              modelo de motor sano y residuos
  splits.py · features.py   partición por motor y variables que sólo miran hacia atrás
  eda.py                 análisis exploratorio → outputs/<modo>/eda.json
  models/                etapa 1: catboost_model.py · tft_model.py
  metrics.py · uncertainty.py   métricas; CQR asimétrico, CQR por tramos y ACI de flota (etapa 1)
  cvplus.py              MODELO FINAL: CV+ por motor, ACI de flota, bootstrap por motor y política de inspección
  line2_transfer.py · line3_changepoint.py   experimentos de las Líneas 2 y 3
  report/                build.py (informes), entrega.py (página del modelo final), plantillas y estilos
  nb.py                  ayudas de las notebooks (leen outputs/, no recalculan)
notebooks/               01_eda · 02_rul_incertidumbre (etapa 1) · 03_lineas_futuras · 04_modelo_final · colab_tft
outputs/full/            resultados de la corrida final (JSON y parquet livianos, versionados)
reports/                 entrega_modelo.html · informe_final.html · informe_completo.html
paper_phm2027/           abstract para PHM 2027 (main.tex + PHMSociety.cls) y estado del arte
compartir.html           Parte I
```

Nada está escrito a mano: los números, tablas y gráficos de informes y notebooks se calculan desde los datos y se leen
de `outputs/`. Semilla fija (`seed` en `config.yaml`), versiones fijadas y partición determinística.

## Autores

- **Trabajo práctico (Laboratorio II):** Verónica Ana López Marsico, Juan Ignacio Acciardi, Pedro Naón y Agustín Giannice.
- **Abstract PHM 2027:** Juan Ignacio Acciardi, Agustín Giannice, Verónica Ana López Marsico, Pedro Naón y
  Julio Alberto Paredes Rojas (docente).

## Fuentes

- Arias Chao, Kulkarni, Goebel y Fink (2021). *Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics.* Data, 6(1), 5.
- Prokhorenkova et al. (2018). *CatBoost: unbiased boosting with categorical features.*
- Lim, Arık, Loeff y Pfister (2021). *Temporal Fusion Transformers for interpretable multi-horizon time series forecasting.*
- Romano, Patterson y Candès (2019). *Conformalized Quantile Regression.*
- Barber, Candès, Ramdas y Tibshirani (2021). *Predictive inference with the jackknife+.* Annals of Statistics (base de CV+).
- Gibbs y Candès (2021). *Adaptive Conformal Inference Under Distribution Shift.*
- Page (1954). *Continuous inspection schemes* (CUSUM) · Killick, Fearnhead y Eckley (2012). *Optimal detection of changepoints with a linear computational cost* (PELT).
- Antecedentes del abstract: ver [`paper_phm2027/estado_del_arte.md`](paper_phm2027/estado_del_arte.md).
