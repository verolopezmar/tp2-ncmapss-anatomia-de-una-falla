"""Formato de números en español (separador de miles '.', decimal ',')."""
from __future__ import annotations

import math


def num(v: float, d: int = 0) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    s = f"{v:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def pct(v: float, d: int = 0) -> str:
    """v en tanto por uno -> '31 %'."""
    return f"{num(100 * v, d)} %"


def millions(v: float, d: int = 1) -> str:
    return f"{num(v / 1e6, d)} M"


def signed(v: float, d: int = 0) -> str:
    return ("+" if v >= 0 else "−") + num(abs(v), d)
