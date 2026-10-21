"""Configuración del proyecto: lee config.yaml y resuelve las rutas.

La carpeta de datos sale de la variable de entorno NCMAPSS_DATA_DIR (o de un
archivo .env en la raíz). Así cada integrante usa su propia ruta sin tocar código.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Paths:
    raw: Path          # carpeta con los .h5 originales
    processed: Path    # parquet intermedios (fuera del repo)
    outputs: Path      # métricas y figuras (dentro del repo)
    models: Path       # modelos entrenados (ignorado por git)
    reports: Path


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_paths(cfg: dict, mode: str = "full") -> Paths:
    """mode = 'full' o 'sample'; cada modo guarda sus resultados por separado."""
    raw = Path(os.environ.get("NCMAPSS_DATA_DIR", cfg["data"]["default_dir"]))
    p = Paths(
        raw=raw,
        processed=raw / "processed" / mode,
        outputs=ROOT / "outputs" / mode,
        models=ROOT / "outputs" / mode / "models",
        reports=ROOT / "reports",
    )
    for d in (p.processed, p.outputs, p.models, p.reports):
        d.mkdir(parents=True, exist_ok=True)
    return p


def dataset_name(path: Path) -> str:
    """'N-CMAPSS_DS08a-009.h5' -> 'DS08a'."""
    return path.stem.split("_")[1].split("-")[0]
