"""Corre el pipeline completo de punta a punta.

    python scripts/run_all.py              # versión completa (9 subconjuntos, ~1 h con GPU)
    python scripts/run_all.py --sample     # modo rápido: 2 subconjuntos y pocos motores, minutos
    python scripts/run_all.py --only eda report      # sólo algunas etapas
    python scripts/run_all.py --skip tft             # p. ej. si el TFT se entrena en Colab

Etapas, en orden:
  prepare    pasada 1 sobre los .h5: integridad, calidad, agregados por vuelo, muestras
  health     modelo de "motor sano" (condiciones de vuelo -> sensores)
  residuals  pasada 2: residuos por vuelo (medido - esperado)
  features   tabla de modelado y partición por motor
  eda        figuras, tablas y hallazgos del análisis exploratorio
  catboost   baseline CatBoost con cuantiles
  tft        Temporal Fusion Transformer (GPU si hay)
  conformal  intervalos conformales adaptativos y métricas
  line2      experimento: vuelos cortos -> medianos/largos
  line3      experimento: detección temprana con change points
  report     arma reports/informe_final.html con los resultados reales
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ncmapss.config import get_paths, load_config  # noqa: E402

STAGES = ["prepare", "health", "residuals", "features", "eda", "catboost", "tft",
          "conformal", "line2", "line3", "report"]


def run_stage(name: str, cfg: dict, paths, mode: str, workers: int) -> None:
    if name == "prepare":
        from src.ncmapss import prepare
        prepare.run(cfg, paths, mode, workers)
    elif name == "health":
        from src.ncmapss import health
        health.train(cfg, paths, mode)
    elif name == "residuals":
        from src.ncmapss import health
        health.residuals(cfg, paths, mode, workers)
    elif name == "features":
        from src.ncmapss import features
        features.run(cfg, paths)
    elif name == "eda":
        from src.ncmapss import eda
        eda.run(cfg, paths)
    elif name == "catboost":
        from src.ncmapss.models import catboost_model
        catboost_model.run(cfg, paths, mode)
    elif name == "tft":
        from src.ncmapss.models import tft_model
        tft_model.run(cfg, paths, mode)
    elif name == "conformal":
        from src.ncmapss import uncertainty
        uncertainty.run(cfg, paths)
    elif name == "line2":
        from src.ncmapss import line2_transfer
        line2_transfer.run(cfg, paths, mode)
    elif name == "line3":
        from src.ncmapss import line3_changepoint
        line3_changepoint.run(cfg, paths)
    elif name == "report":
        from src.ncmapss.report import build
        build.run(cfg, paths, mode)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true", help="modo rápido con pocos motores")
    ap.add_argument("--only", nargs="+", choices=STAGES, help="correr sólo estas etapas")
    ap.add_argument("--skip", nargs="+", choices=STAGES, default=[], help="saltear estas etapas")
    ap.add_argument("--workers", type=int, default=4, help="procesos para leer los .h5 en paralelo")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("lightning").setLevel(logging.WARNING)
    mode = "sample" if args.sample else "full"
    cfg = load_config()
    paths = get_paths(cfg, mode)
    logging.info("Modo %s · datos en %s", mode.upper(), paths.raw)

    stages = [s for s in (args.only or STAGES) if s not in args.skip]
    for s in stages:
        t = time.time()
        logging.info("▶ %s", s)
        run_stage(s, cfg, paths, mode, args.workers)
        logging.info("✔ %s (%.1f s)", s, time.time() - t)


if __name__ == "__main__":
    main()
