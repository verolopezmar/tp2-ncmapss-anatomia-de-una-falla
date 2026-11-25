# Estado del arte y posicionamiento · abstract PHM 2027

Relevamiento hecho el 24/11/2026 para posicionar el abstract (`main.tex`) y como base del paper completo. Cada referencia
se verificó en una página real (editor, Crossref, Semantic Scholar, arXiv, Zenodo o el PDF). Donde sólo se leyó el
resumen o los metadatos, se indica. **Antes de citar en el paper, leer el texto completo de cada una.**

## El antecedente más cercano

**Robinson, C. D. (2026).** Remaining Useful Life Estimation for Aircraft Engines with Risk-Aware Prediction Intervals via
Conformalized Quantile Regression. *International Journal of Prognostics and Health Management*, 17(1).
<https://doi.org/10.36001/ijphm.2026.v17i1.4724> (PDF completo leído)

- Gradient boosting + **CQR asimétrica**: reparte α = 0,10 en 2,5 % / 7,5 % (cociente r = 3), casi igual a nuestro 3 / 7,
  justificado por evitar el mantenimiento tardío.
- C-MAPSS (FD001–FD004), no N-CMAPSS. Calibración con una sola partición 70/30 por motor; sin validación cruzada.
- Sin calibración adaptativa y sin política de decisión. Su trabajo futuro pide textualmente *"adaptive conformal methods
  that maintain coverage under distribution shift"* y *"maintenance optimization models that explicitly leverage the
  calibrated prediction intervals for decision support"*.

**Consecuencia:** la asimetría **no** es la contribución. El aporte es la combinación que propone el profesor: calibración
por validación cruzada por motor con reentrenamiento completo, actualización adaptativa de flota y decisión de inspección,
sobre N-CMAPSS. Es justamente lo que Robinson deja abierto.

## Trabajos relevantes

### Predicción conformal e incertidumbre en vida útil de turbofanes

| Trabajo | Dataset | Qué hace | Diferencia con el nuestro |
|---|---|---|---|
| Javanmardi y Hüllermeier (2023), IJPHM 14(2). <https://doi.org/10.36001/ijphm.2023.v14i2.3417> | C-MAPSS | Referencia fundacional de conformal en RUL; tres algoritmos sobre CNN y gradient boosting | Simétrico, partición simple, sin adaptación ni decisión |
| Diao, Zhou, Meng y Wang (2026), *Sensors* 26(7), 2249. <https://doi.org/10.3390/s26072249> (resumen) | C-MAPSS | LSTM cuantílica con penalización por sobreestimación + CQR; transferencia FD001↔FD002 | Sin validación cruzada por motor, sin actualización online ni inspección |
| Lang, Wei, Zhang y Qiu (2026), *IEEE Access* 14. <https://doi.org/10.1109/ACCESS.2026.3751338> (resumen) | C-MAPSS y N-CMAPSS DS02 | Ensamble; calibra cada cota por separado según el desacuerdo del ensamble | "Adaptativo" = según el ensamble, no online; sólo DS02; sin decisión |
| Domingues y Ather (2026), *J. AI, ML and Neural Network* 62. <https://doi.org/10.55529/jaimlnn.62.106.119> | C-MAPSS | Diagnóstico: bajo cambio de régimen, el 95 % de las fallas de cobertura son sobreestimaciones | Motiva el enfoque; no propone solución |
| Leqyanya (2026), ICSIS. <https://doi.org/10.1109/ICSIS70067.2026.11778945> (resumen) | C-MAPSS | Conformal normalizado por fase de degradación | Adapta el ancho por fase, no online |
| Xu, Gao, Wang y Zio (2026), *RESS* 275. <https://doi.org/10.1016/j.ress.2026.112763> (fragmento) | C-MAPSS y baterías | Bayesiano calibrado + conformal | Sin N-CMAPSS ni decisión |
| Busico (2026), *RESS* 278. <https://doi.org/10.1016/j.ress.2026.113634> (sólo metadatos) | ? | Conformal bajo transferencia entre regímenes | **Leer antes del paper**: toca nuestro experimento de clases de vuelo |
| Noot, Martin y Birmelé (2025), IJPHM 16(2). <https://doi.org/10.36001/ijphm.2025.v16i2.4260> | C-MAPSS y N-CMAPSS | Datos censurados; conformal para intervalos | Conformal en N-CMAPSS ya existe; simétrico, sin adaptación ni decisión |
| Basora, Viens, Arias Chao y Olive (2023). arXiv:2302.04730 | N-CMAPSS (subconjunto) | Benchmark de incertidumbre bayesiana y ensambles | Sin garantía conformal, asimetría ni decisión |
| Yan (2026), paquetes en Zenodo, **sin revisión por pares**: <https://doi.org/10.5281/zenodo.21281080>, <https://doi.org/10.5281/zenodo.21330745> | N-CMAPSS (9 subconjuntos) y DS02 | Conformal auditado, piso unilateral para predicción tardía, CV+ por motor en 5 grupos y ACI condicionada por régimen | Por separado aparecen CV+ por motor, ACI y cota unilateral; no el reentrenamiento del pipeline completo, ni ACI de flota con asimetría, ni inspección. Manuscrito enviado a RESS: **vigilar** |

### N-CMAPSS y modelo de motor sano

- Arias Chao, Kulkarni, Goebel y Fink (2021). *Data* 6(1), 5. <https://doi.org/10.3390/data6010005> — el dataset.
- Arias Chao et al. (2022). *RESS* 217, 107961. <https://doi.org/10.1016/j.ress.2021.107961> — híbrido física + aprendizaje profundo.
- **Lövberg (2021).** Annual Conference of the PHM Society 13(1). <https://doi.org/10.36001/phmconf.2021.v13i1.3108> —
  Data Challenge 2021 (N-CMAPSS): red entrenada en datos sanos que predice los sensores desde las condiciones de vuelo y
  usa la diferencia como degradación. **Es casi nuestro modelo de motor sano: citarlo, no presentarlo como aporte.**

### Penalización asimétrica de la sobreestimación

- Saxena, Goebel, Simon y Eklund (2008). Int. Conf. PHM. <https://doi.org/10.1109/PHM.2008.4711414> — C-MAPSS y el score
  asimétrico (sobreestimar es peor).
- Robinson (2026), Diao et al. (2026) y Domingues y Ather (2026), arriba.

### Decisiones de mantenimiento basadas en pronósticos

| Trabajo | Qué hace | Diferencia |
|---|---|---|
| de Pater, Reijns y Mitici (2022), *RESS* 221, 108341. <https://doi.org/10.1016/j.ress.2022.108341> | Alarmas con factor de seguridad sobre el RUL de una CNN; programación entera | Margen fijo, no una cota calibrada |
| Mitici, de Pater, Barros y Zeng (2023), *RESS* 234, 109199. <https://doi.org/10.1016/j.ress.2023.109199> | Distribución del RUL (MC dropout) en planificación; 53 % menos costo que por tiempo | Incertidumbre aproximada, sin cobertura garantizada |
| Lee y Mitici (2023), *RESS* 230, 108908. <https://doi.org/10.1016/j.ress.2022.108908> | Aprendizaje por refuerzo sobre pronósticos probabilísticos | Sin conformal |
| de Pater y Mitici (2026), *RESS* 277, 113184. <https://doi.org/10.1016/j.ress.2026.113184> | Programación estocástica multietapa | Sin conformal |
| AlAllawi (2026), SSRN (preprint). <https://doi.org/10.2139/ssrn.7682391> | Control de riesgo conformal a nivel decisión; acota la tasa de fallas no planificadas | C-MAPSS, sin CV+ por motor ni actualización de flota. **El más cercano en "conformal → decisión"** |

### Métodos base

- Romano, Patterson y Candès (2019). Conformalized Quantile Regression. arXiv:1905.03222.
- Barber, Candès, Ramdas y Tibshirani (2021). Predictive inference with the jackknife+. *Annals of Statistics* 49(1). arXiv:1905.02928 (CV+).
- Gibbs y Candès (2021). Adaptive Conformal Inference Under Distribution Shift. arXiv:2106.00170.

## Brecha

En los trabajos revisados **no encontramos** una propuesta que combine:

1. calibración conformal asimétrica dentro de una validación cruzada por motor que **reentrena el pipeline completo**
   (incluido el modelo de motor sano que genera los residuos), para calibrar con los 60 motores de desarrollo;
2. una **actualización adaptativa de flota** que incorpora la vida real de cada motor cuando falla;
3. una **política de inspección** derivada de la cota inferior, con umbral elegido en desarrollo y evaluada en los 39
   motores oficiales contra una vida fija por clase de vuelo (inspecciones tardías y vida desperdiciada);
4. todo sobre **N-CMAPSS** (perfiles de vuelo reales, 9 subconjuntos), con un experimento de generalización entre clases de vuelo.

## Afirmaciones a evitar

- "La contribución principal es la capa conformal asimétrica" (Robinson 2026).
- Cualquier "first": conformal en N-CMAPSS (Noot 2025, Lang 2026), asimetría (Robinson 2026), conformal → decisión
  (AlAllawi 2026), conformal adaptativo en pronóstico (Yan 2026).
- Presentar el modelo de motor sano con residuos como aporte (Lövberg 2021).
- "Garantía" de cobertura: con dependencia dentro del motor y con ACI, la cobertura es empírica.
- Comparar el RMSE de 6,6 vuelos con otros trabajos sobre N-CMAPSS: particiones y recortes difieren.

## Pendiente antes del paper completo

- Leer Busico (2026), el texto completo de Diao, Lang y AlAllawi, y el dataset exacto de Mitici et al. (2023).
- Revisar si el manuscrito de Yan sobre N-CMAPSS ya salió en RESS.
