"""Compara las cifras publicadas en compartir.html (Parte I) con las que recalcula el EDA actual.

    python scripts/compare_parte1.py

Lee window.EDA.facts embebido en compartir.html y outputs/full/eda.json, y muestra lado a lado
las cifras con el mismo nombre, marcando las que difieren.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def old_facts() -> dict:
    s = (ROOT / "compartir.html").read_text(encoding="utf-8")
    for body in re.findall(r"<script[^>]*>(.*?)</script>", s, flags=re.S):
        if body.lstrip().startswith("window.EDA"):
            return json.loads(body[body.index("=") + 1:].strip().rstrip(";"))["facts"]
    raise SystemExit("No se encontró window.EDA en compartir.html")


def main() -> None:
    new = json.loads((ROOT / "outputs" / "full" / "eda.json").read_text(encoding="utf-8"))["facts"]
    old = old_facts()
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"{'cifra':<18}{'Parte I':>20}{'Parte II':>20}   ")
    for k in old:
        a, b = old[k], new.get(k, "—")
        flag = "" if a == b else "  ≠"
        print(f"{k:<18}{a:>20}{b:>20}{flag}")


if __name__ == "__main__":
    main()
