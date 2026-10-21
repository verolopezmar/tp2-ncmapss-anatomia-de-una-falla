"""Utilidades para las figuras Plotly con la estética de "Anatomía de una Falla".

Las figuras se guardan como dicts {data, layout} en JSON. Los colores se escriben como
variables CSS (p. ej. "var(--hud)"): el informe HTML los traduce con la paleta de la
página (igual que compartir.html). Para verlas en un notebook, `to_plotly` reemplaza
esas variables por los colores reales leídos de la hoja de estilos.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

CSS_PATH = Path(__file__).parent / "report" / "assets" / "anatomia.css"

COMP_COLOR = {"fan": "var(--c-fan)", "LPC": "var(--c-lpc)", "HPC": "var(--c-hpc)",
              "HPT": "var(--c-hpt)", "LPT": "var(--c-lpt)"}
FC_COLOR = {1: "var(--hud-2)", 2: "var(--hud)", 3: "var(--c-hpc)"}
SPLIT_COLOR = {"dev": "var(--hud)", "test": "var(--ti)"}
DIVERGING = [[0, "var(--hud)"], [0.5, "var(--mid)"], [1, "var(--red)"]]
HEAT = [[0, "var(--surface-2)"], [0.35, "var(--amber)"], [0.7, "var(--orange)"], [1, "var(--red)"]]
RUL_SCALE = [[0, "var(--red)"], [0.25, "var(--orange)"], [0.5, "var(--amber)"], [1, "var(--hud)"]]


def rnd(x, nd: int = 3):
    """Convierte arrays/series a listas redondeadas (JSON más liviano)."""
    a = np.asarray(x)
    if a.dtype.kind in "iub":
        return a.tolist()
    if a.dtype.kind == "f":
        return np.round(a, nd).tolist()
    return [str(v) for v in a.tolist()]


def fig(data: list[dict], **layout) -> dict:
    return {"data": data, "layout": layout}


def subplot_axes(rows: int, cols: int = 1, vgap: float = 0.06, hgap: float = 0.08,
                 shared_x: bool = True) -> dict:
    """Layout de subgráficos sin pasar por plotly.subplots (que valida colores)."""
    lay = {}
    h = (1 - vgap * (rows - 1)) / rows
    w = (1 - hgap * (cols - 1)) / cols
    n = rows * cols
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c + 1
            sfx = "" if i == 1 else str(i)
            y1 = 1 - r * (h + vgap)
            x0 = c * (w + hgap)
            xa = {"anchor": f"y{sfx}", "domain": [round(x0, 4), round(x0 + w, 4)]}
            if shared_x and rows > 1:
                last_in_col = (rows - 1) * cols + c + 1
                if i != last_in_col:
                    xa.update(matches=f"x{'' if last_in_col == 1 else last_in_col}", showticklabels=False)
            lay[f"xaxis{sfx}"] = xa
            lay[f"yaxis{sfx}"] = {"anchor": f"x{sfx}", "domain": [round(y1 - h, 4), round(y1, 4)]}
    lay["_n"] = n
    return lay


def axis_ref(i: int) -> tuple[str, str]:
    return ("x" if i == 1 else f"x{i}"), ("y" if i == 1 else f"y{i}")


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=_default), encoding="utf-8")


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ── Visualización en notebooks ────────────────────────────────────────────────
def css_tokens() -> dict[str, str]:
    css = CSS_PATH.read_text(encoding="utf-8")
    root = css[css.index(":root{"):]
    root = root[: root.index("}")]
    return dict(re.findall(r"(--[\w-]+):\s*([^;]+);", root))


def to_plotly(spec: dict):
    """Figura del informe -> plotly.graph_objects.Figure con la paleta resuelta."""
    import plotly.graph_objects as go
    tok = css_tokens()

    def res(o):
        if isinstance(o, str):
            return re.sub(r"var\((--[\w-]+)\)", lambda m: tok.get(m.group(1), "#888"), o)
        if isinstance(o, list):
            return [res(x) for x in o]
        if isinstance(o, dict):
            return {k: res(v) for k, v in o.items()}
        return o

    layout = {k: v for k, v in res(spec.get("layout", {})).items() if not k.startswith("_")}
    layout.setdefault("template", "plotly_white")
    layout.setdefault("font", {"family": "Inter, sans-serif", "color": tok.get("--ink-2")})
    return go.Figure(data=res(spec["data"]), layout=layout)
