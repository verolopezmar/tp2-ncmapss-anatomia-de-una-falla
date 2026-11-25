"""Demo del modelo final: de los .h5 crudos a una decisión de inspección, para un motor de prueba.

    python scripts/predecir_motor.py --motor DS02_u11 --vuelos 20 40 60
    python scripts/predecir_motor.py --motor DS02_u11 --cada 10
    python scripts/predecir_motor.py --listar

Para cada vuelo pedido sólo se usa la historia hasta ese vuelo: los segundos crudos pasan por los 5 modelos
de motor sano, se arman las variables, predicen los 5 CatBoost y CV+ da el intervalo. El RUL real se
muestra al final, sólo para comparar: nunca entra al modelo. Con --verificar se contrasta el resultado con
las predicciones guardadas por la etapa cv_fit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catboost import CatBoostRegressor  # noqa: E402

from src.ncmapss import io  # noqa: E402
from src.ncmapss.config import dataset_name, get_paths, load_config  # noqa: E402
from src.ncmapss.cvplus import CVPlus  # noqa: E402
from src.ncmapss.features import build, sensor_names  # noqa: E402
from src.ncmapss.health import _load_model, predict  # noqa: E402
from src.ncmapss.prepare import KEY, _combine_flight_aggs, _partial_flight_aggs  # noqa: E402


def leer_motor(uid: str, cfg: dict, paths, K: int) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Devuelve los agregados por vuelo y, para cada grupo, los residuos medios por vuelo."""
    ds, unit = uid.split("_u")[0], int(uid.split("_u")[1])
    path = next(f for f in paths.raw.glob(cfg["data"]["file_glob"]) if dataset_name(f) == ds)
    names = io.variable_names(path)
    models = [_load_model(paths, tag=f"_cv{k}") for k in range(K)]
    partials, sums = [], [[] for _ in range(K)]
    for df in io.iter_chunks(path, "test", cfg["data"]["chunk_rows"], {unit}):
        partials.append(_partial_flight_aggs(df, names))
        for k, (m, ck) in enumerate(models):
            res = pd.DataFrame(df[ck["outputs"]].to_numpy() - predict(m, ck, df[ck["inputs"]].to_numpy()),
                               columns=[f"res_{c}_mean" for c in ck["outputs"]])
            res[KEY] = df[KEY].to_numpy()
            sums[k].append(res.groupby(KEY).agg(["sum", "size"]))
    flights = _combine_flight_aggs(partials, names).assign(uid=uid)
    out = []
    for k in range(K):
        a = pd.concat(sums[k]).groupby(level=[0, 1]).sum()
        means = pd.DataFrame({c: a[c]["sum"] / a[c]["size"] for c in a.columns.levels[0]}).reset_index()
        out.append(flights.merge(means, on=KEY, validate="1:1"))
    return flights, out


def predecir(tablas: list[pd.DataFrame], vuelo: int, sensors, feats, models, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Una predicción por modelo de grupo, cortando la historia en `vuelo`."""
    q = []
    for t, m in zip(tablas, models):
        hist = t[t["cycle"] <= vuelo].drop(columns=["RUL"]).assign(RUL=0)      # el RUL no está disponible
        f, _ = build(hist, sensors, cfg)
        q.append(np.sort(np.asarray(m.predict(f.sort_values("cycle").iloc[[-1]][feats]))[0]))
    q = np.array(q)
    return q[:, 0][None, :], q[:, 1], q[:, 2][None, :]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motor", help="identificador, por ejemplo DS02_u11")
    ap.add_argument("--vuelos", nargs="+", type=int, help="vuelos a evaluar")
    ap.add_argument("--cada", type=int, help="evaluar cada N vuelos")
    ap.add_argument("--listar", action="store_true", help="mostrar los motores de prueba disponibles")
    ap.add_argument("--verificar", action="store_true", help="comparar con las predicciones guardadas")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = load_config()
    paths = get_paths(cfg, "full")
    res = json.loads((paths.outputs / "cvplus.json").read_text(encoding="utf-8"))
    preds = pd.read_parquet(paths.outputs / "cvplus_preds.parquet")
    units = sorted(preds.loc[preds["role"] == "test", "uid"].unique())
    if a.listar or not a.motor:
        print("Motores de prueba:", ", ".join(units))
        return
    if a.motor not in units:
        raise SystemExit(f"{a.motor} no es un motor de prueba. Usá --listar.")

    K = cfg["cvplus"]["folds"]
    cc, rc = cfg["conformal"], cfg["results"]
    H = res["inspection"]["by_tau"][str(cfg["cvplus"]["inspection"]["lead_flights"])]["cota inferior"]["H"]
    edges = rc["rul_buckets"] if res["chosen"] == "cvplus_tramos" else None
    oof = preds[(preds["role"] != "test") & (preds["cycle"] >= rc["eval_min_cycle"])]
    cal = CVPlus(oof, K, edges)
    feats = json.loads((paths.processed / "features.json").read_text())
    sensors = sensor_names(paths)
    models = [CatBoostRegressor().load_model(str(paths.models / f"cvplus_fold{k}.cbm")) for k in range(K)]

    _, tablas = leer_motor(a.motor, cfg, paths, K)
    ultimo = int(tablas[0]["cycle"].max())
    vuelos = a.vuelos or list(range(a.cada or 10, ultimo + 1, a.cada or 10))
    real = tablas[0].set_index("cycle")["RUL"]
    saved = preds[(preds["uid"] == a.motor)].pivot(index="cycle", columns="fold", values="q50")
    print(f"\nMotor {a.motor} · {K} modelos · se ordena inspección si la cota inferior ≤ {H:g} vuelos\n")
    print(f"{'vuelo':>6}{'RUL predicho':>14}{'intervalo':>16}{'decisión':>22}{'RUL real':>10}")
    dif = []
    for v in vuelos:
        if v not in real.index or v < rc["eval_min_cycle"]:
            continue
        q05, q50, q95 = predecir(tablas, v, sensors, feats, models, cfg)
        lo, hi = cal.bounds(q05, np.array([q50.mean()]), q95, cc["alpha_over"], cc["alpha_under"])
        dec = "ORDENAR INSPECCIÓN" if lo[0] <= H else "seguir volando"
        print(f"{v:>6}{q50.mean():>14.1f}{f'[{lo[0]:.1f}, {hi[0]:.1f}]':>16}{dec:>22}{int(real[v]):>10}")
        dif.append(np.abs(q50 - saved.loc[v, list(range(K))].to_numpy()).max())
    if a.verificar and dif:
        print(f"\nDiferencia máxima con las predicciones guardadas: {max(dif):.6f} vuelos")


if __name__ == "__main__":
    main()
