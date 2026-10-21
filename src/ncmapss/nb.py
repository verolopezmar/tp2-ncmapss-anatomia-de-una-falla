"""Ayudas para los notebooks: muestran figuras, tablas y lecturas guardadas por el pipeline."""
from __future__ import annotations

import json
import re

import pandas as pd
from IPython.display import HTML, Markdown, display

from .config import get_paths, load_config
from .viz import to_plotly

_CACHE: dict = {}


def load(name: str = "eda", mode: str = "full") -> dict:
    paths = get_paths(load_config(), mode)
    path = paths.outputs / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Corré antes: python scripts/run_all.py" + (" --sample" if mode == "sample" else ""))
    _CACHE[name] = json.loads(path.read_text(encoding="utf-8"))
    return _CACHE[name]


def _plain(html: str) -> str:
    return re.sub(r"<i>|</i>", "_", re.sub(r"<b>|</b>", "**", html))


def show(fig_id: str, data: dict | None = None) -> None:
    """Dibuja la figura `fig_id` y debajo su lectura (calculada desde los datos)."""
    d = data or _CACHE.get("eda")
    to_plotly(d["figures"][fig_id]).show()
    if fig_id in d.get("insights", {}):
        display(Markdown("> **Lectura.** " + _plain(d["insights"][fig_id])))


def table(table_id: str, data: dict | None = None) -> pd.DataFrame:
    d = data or _CACHE.get("eda")
    t = d["tables"][table_id]
    return pd.DataFrame(t["rows"], columns=t["columns"])


def checks(data: dict | None = None) -> pd.DataFrame:
    d = data or _CACHE.get("eda")
    return pd.DataFrame([{"verificación": c["name"], "resultado": "✔" if c["ok"] else "✘", "detalle": c["detail"]} for c in d["checks"]])


def facts(data: dict | None = None) -> pd.DataFrame:
    d = data or _CACHE.get("eda")
    return pd.Series(d["facts"], name="valor").to_frame()


def decisions(data: dict | None = None) -> pd.DataFrame:
    d = data or _CACHE.get("eda")
    return pd.DataFrame(d["decisions"]).rename(columns={"finding": "hallazgo del EDA", "decision": "decisión de modelado"})


def html(s: str) -> None:
    display(HTML(s))
