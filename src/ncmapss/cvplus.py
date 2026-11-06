"""CatBoost validado por motor con CV+ y política de inspección.

Dos etapas, sobre los datos reales:

cv_fit   Los 60 motores de desarrollo se reparten en K grupos, estratificados por subconjunto. Para cada
         grupo k se repite TODO lo que depende de los motores: modelo sano, residuos de todos los vuelos,
         variables y CatBoost multicuantil. Así, ningún motor del grupo k influye en su propia predicción
         (tampoco a través de los residuos), y cada motor de desarrollo aporta puntajes de calibración.

cv_eval  1) Elige el método y los umbrales SÓLO con motores de desarrollo y guarda esa elección.
         2) Recién después evalúa en los 39 motores de prueba oficiales.
         3) Da intervalos de confianza remuestreando MOTORES, no vuelos.

CV+ (Barber et al., 2021) con puntajes CQR (Romano et al., 2019), para el vuelo x de prueba:
    L(x) = cuantil inferior de {q05_{-k(i)}(x) − s_lo_i}  con s_lo_i = q05_{-k(i)}(x_i) − y_i
    U(x) = cuantil superior de {q95_{-k(i)}(x) + s_hi_i}  con s_hi_i = y_i − q95_{-k(i)}(x_i)
donde k(i) es el grupo del vuelo i. Los vuelos de un motor no son intercambiables entre sí, por lo que la
garantía teórica es aproximada: por eso se mide la cobertura real y su incertidumbre por motor.

Política de inspección: se ordena la inspección en el primer vuelo con cota ≤ H. Se necesitan τ vuelos
para ejecutarla. Es TARDÍA si en ese momento quedan menos de τ vuelos reales; la vida desperdiciada son los
vuelos que quedaban después de esos τ. H se elige en desarrollo y se evalúa una sola vez en prueba.
"""
from __future__ import annotations

import gc
import json
import logging
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from . import metrics as M
from .config import Paths
from .features import build, sensor_names
from .health import _sample_with_roles, fit, residuals
from .models.common import hardware
from .splits import load_splits
from .uncertainty import Calibrator
from .viz import save_json

log = logging.getLogger(__name__)
ROWS = ["uid", "ds", "role", "cycle", "Fc", "RUL", "eol"]


# ════════════════════════════════════════════════════════════════════════════
#  Etapa 1: ajuste por grupos
# ════════════════════════════════════════════════════════════════════════════
def make_folds(splits: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Reparte los motores de desarrollo en k grupos, estratificados por subconjunto."""
    rng = np.random.default_rng(seed)
    rows, offset = [], 0
    for _, g in splits[splits["role"] != "test"].sort_values("uid").groupby("ds", sort=True):
        uids = g["uid"].to_numpy()[rng.permutation(len(g))]
        rows += [(u, (offset + i) % k) for i, u in enumerate(uids)]
        offset += len(uids)
    return pd.DataFrame(rows, columns=["uid", "fold"])


def _catboost(cfg: dict, iterations: int) -> CatBoostRegressor:
    cc = cfg["catboost"]
    return CatBoostRegressor(
        loss_function="MultiQuantile:alpha=" + ",".join(str(a) for a in cc["quantiles"]), iterations=iterations,
        learning_rate=cc["learning_rate"], depth=cc["depth"], random_seed=cfg["seed"], verbose=False,
        allow_writing_files=False, thread_count=-1)


def pinball(y: np.ndarray, q: np.ndarray, quantiles: list[float]) -> np.ndarray:
    """Pérdida cuantil sumada por iteración candidata; y: n · q: n × cuantiles."""
    e = y[:, None] - q
    return np.maximum(np.array(quantiles) * e, (np.array(quantiles) - 1) * e).sum()


def run_fit(cfg: dict, paths: Paths, mode: str, workers: int = 1) -> None:
    """Un mismo número de iteraciones para los K modelos, elegido por pinball fuera de grupo en desarrollo."""
    vc, qs = cfg["cvplus"], cfg["catboost"]["quantiles"]
    K, step, top = vc["folds"], vc["iteration_step"], vc["max_iterations"]
    splits = load_splits(paths)
    folds = make_folds(splits, K, cfg["seed"])
    folds.to_parquet(paths.outputs / "cvplus_folds.parquet")
    sensors = sensor_names(paths)
    grid = np.arange(step, top + 1, step)
    loss, n_rows, models, tables, info = np.zeros(len(grid)), 0, {}, {}, []
    for k in range(K):
        t0 = time.time()
        tag = f"_cv{k}"
        hold = set(folds.loc[folds["fold"] == k, "uid"])
        res_path = paths.processed / f"flights{tag}.parquet"
        if not res_path.exists():
            s = _sample_with_roles(paths, splits)
            base = s[(s["split"] == "dev") & (s["hs"] == 1)]
            h = fit(cfg, paths, mode, base[~base["uid"].isin(hold)], base[base["uid"].isin(hold)], tag=tag)
            names = json.loads((paths.processed / "variable_names.json").read_text())
            log.info("  grupo %d · modelo sano: %d motores · R² mediano en el grupo apartado %.5f", k, h["train_units"],
                     float(np.median([h["r2_val"][c] for c in names["X_s"]])))
            del s, base                    # libera la muestra de segundos antes de la pasada sobre los .h5
            gc.collect()
            residuals(cfg, paths, mode, workers, tag=tag)
        t, feats = build(pd.read_parquet(res_path), sensors, cfg)
        t = t.merge(splits[["uid", "role"]], on="uid", validate="m:1")
        t["eol"] = t.groupby("uid")["cycle"].transform("max")
        fit_units = set(folds.loc[folds["fold"] != k, "uid"])
        al = t[t["uid"].isin(fit_units)]
        model = _catboost(cfg, top)
        model.fit(Pool(al[feats], al["RUL"]))
        ho = t[t["uid"].isin(hold) & (t["cycle"] >= cfg["results"]["eval_min_cycle"])]
        y = ho["RUL"].to_numpy(dtype=float)
        curve = [pinball(y, np.sort(np.asarray(q), axis=1), qs) for q in model.staged_predict(ho[feats], eval_period=step)]
        loss += np.asarray(curve)[: len(grid)]
        n_rows += len(y)
        models[k], tables[k] = model, t
        info.append({"fold": k, "motores_apartados": len(hold), "motores_ajuste": len(fit_units), "segundos": round(time.time() - t0, 1)})
        log.info("  grupo %d ajustado en %.0f s", k, time.time() - t0)
    best = int(grid[int(np.argmin(loss))])
    log.info("  iteraciones comunes elegidas en desarrollo: %d (pinball %.4f)", best, loss.min() / n_rows / len(qs))
    preds = []
    for k in range(K):
        hold = set(folds.loc[folds["fold"] == k, "uid"])
        t = tables[k]
        models[k].shrink(best)
        models[k].save_model(str(paths.models / f"cvplus_fold{k}.cbm"))
        out = t[t["uid"].isin(hold) | (t["role"] == "test")].sort_values(["uid", "cycle"]).reset_index(drop=True)
        q = np.sort(np.asarray(models[k].predict(out[feats])), axis=1)
        preds.append(out[ROWS].assign(fold=k, q05=q[:, 0], q50=q[:, 1], q95=q[:, 2]))
    pd.concat(preds, ignore_index=True).to_parquet(paths.outputs / "cvplus_preds.parquet")
    (paths.outputs / "cvplus_fit.json").write_text(json.dumps(
        {"folds": info, "iteraciones": best, "curva_pinball": {"iter": grid.tolist(), "loss": (loss / n_rows / len(qs)).round(5).tolist()},
         "feature_count": len(feats), "hardware": hardware()}, indent=1, ensure_ascii=False), encoding="utf-8")


# ════════════════════════════════════════════════════════════════════════════
#  Intervalos CV+
# ════════════════════════════════════════════════════════════════════════════
def _rank_low(n: int, a: float) -> int:
    return int(np.floor(a * (n + 1)))


def _rank_high(n: int, a: float) -> int:
    return int(np.ceil((1 - a) * (n + 1)))


def _order(values: np.ndarray, r: int, side: str) -> np.ndarray:
    """Estadístico de orden r (desde 1) por fila; sin suficientes datos el intervalo no acota."""
    n = values.shape[1]
    if r < 1 or r > n:
        return np.full(len(values), -np.inf if side == "lo" else np.inf)
    return np.partition(values, r - 1, axis=1)[:, r - 1]


class CVPlus:
    """Puntajes fuera de grupo, globales (edges=None) o por tramo de RUL predicho."""

    def __init__(self, oof: pd.DataFrame, K: int, edges: list[float] | None, exclude: int | None = None):
        self.K, self.edges = K, edges
        o = oof if exclude is None else oof[oof["fold"] != exclude]
        self.sets: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
        b = self._bin(o["q50"].to_numpy())
        for bin_id in np.unique(b):
            g = o[b == bin_id]
            self.sets[int(bin_id)] = {int(k): ((f["q05"] - f["RUL"]).to_numpy(), (f["RUL"] - f["q95"]).to_numpy())
                                      for k, f in g.groupby("fold")}

    def _bin(self, q50: np.ndarray) -> np.ndarray:
        return np.zeros(len(q50), dtype=int) if not self.edges else np.digitize(q50, self.edges[1:-1])

    def bounds(self, q05: np.ndarray, q50: np.ndarray, q95: np.ndarray, a_over: float, a_under: float):
        """q05/q95: n × K, una columna por modelo de grupo (o una sola columna para evaluar en desarrollo)."""
        lo, hi = np.empty(len(q50)), np.empty(len(q50))
        bins = self._bin(q50)
        for b in np.unique(bins):
            m = bins == b
            key = int(b) if int(b) in self.sets else min(self.sets, key=lambda j: abs(j - int(b)))
            sets = self.sets[key]
            if q05.shape[1] == 1:   # un único modelo (motor de desarrollo apartado)
                L = np.hstack([q05[m] - s[0][None, :] for s in sets.values()])
                U = np.hstack([q95[m] + s[1][None, :] for s in sets.values()])
            else:
                L = np.hstack([q05[m, f][:, None] - sets[f][0][None, :] for f in sets])
                U = np.hstack([q95[m, f][:, None] + sets[f][1][None, :] for f in sets])
            n = L.shape[1]
            ao, au = min(max(a_over, 1 / (n + 1)), 0.999), min(max(a_under, 1 / (n + 1)), 0.999)
            lo[m], hi[m] = _order(L, _rank_low(n, ao), "lo"), _order(U, _rank_high(n, au), "hi")
        return np.maximum(lo, 0), hi


def _test_matrices(preds: pd.DataFrame, K: int, min_cycle: int):
    """Pasa las predicciones de prueba a una fila por vuelo y una columna por modelo de grupo."""
    t = preds[(preds["role"] == "test") & (preds["cycle"] >= min_cycle)]
    wide = t.pivot(index=["uid", "cycle"], columns="fold", values=["q05", "q50", "q95"]).sort_index()
    keys = t.drop_duplicates(["uid", "cycle"]).set_index(["uid", "cycle"]).loc[wide.index, ["ds", "Fc", "RUL", "eol"]].reset_index()
    q = {n: wide[n][list(range(K))].to_numpy() for n in ("q05", "q50", "q95")}
    keys["q50"] = q["q50"].mean(1)
    keys["q05"], keys["q95"] = q["q05"].mean(1), q["q95"].mean(1)
    return keys, q


def _dev_pseudotest(oof: pd.DataFrame, K: int, edges, a_over, a_under) -> pd.DataFrame:
    """Cada grupo hace de prueba: se lo acota con su modelo y puntajes de los demás grupos."""
    frames = []
    for k in range(K):
        g = oof[oof["fold"] == k].copy()
        cal = CVPlus(oof, K, edges, exclude=k)
        g["lo"], g["hi"] = cal.bounds(g[["q05"]].to_numpy(), g["q50"].to_numpy(), g[["q95"]].to_numpy(), a_over, a_under)
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def _choose_method(results: dict[str, dict], a_over: float, target: float) -> str:
    """Regla fijada de antemano: respetar la tolerancia a sobreestimar, luego acercarse al objetivo, luego el más angosto."""
    return min(results, key=lambda m: (max(0.0, results[m]["miss_over"] - a_over) > 0.005,
                                       round(abs(results[m]["coverage"] - target), 3), results[m]["width"]))


def _fleet_aci(test: pd.DataFrame, q: dict, cal: CVPlus, cc: dict, orders: list) -> tuple[pd.DataFrame, dict]:
    """ACI de flota sobre CV+: α se corrige cuando cada motor falla (mismo esquema del pipeline base)."""
    pos = test.groupby("uid").indices
    acc = []
    first = None
    for i, order in enumerate(orders):
        lo, hi = np.empty(len(test)), np.empty(len(test))
        ao, au = cc["alpha_over"], cc["alpha_under"]
        for uid in order:
            ix = pos[uid]
            L, U = cal.bounds(q["q05"][ix], test["q50"].to_numpy()[ix], q["q95"][ix], ao, au)
            lo[ix], hi[ix] = L, U
            y = test["RUL"].to_numpy()[ix]
            ao = float(np.clip(ao + cc["aci_gamma"] * (cc["alpha_over"] - np.mean(y < L)), 0, 0.999))
            au = float(np.clip(au + cc["aci_gamma"] * (cc["alpha_under"] - np.mean(y > U)), 0, 0.999))
        acc.append(M.interval_metrics(test["RUL"], lo, hi))
        if i == 0:
            first = (lo, hi)
    spread = pd.DataFrame(acc)
    return test.assign(lo=first[0], hi=first[1]), {**spread.mean().to_dict(), "coverage_sd": float(spread["coverage"].std())}


# ════════════════════════════════════════════════════════════════════════════
#  Incertidumbre por motor
# ════════════════════════════════════════════════════════════════════════════
def engine_sums(df: pd.DataFrame, nasa: dict, p: str = "q50", lo: str | None = "lo", hi: str | None = "hi") -> pd.DataFrame:
    d = df[p] - df["RUL"]
    out = pd.DataFrame({"uid": df["uid"], "n": 1.0, "se": d ** 2, "ae": d.abs(),
                        "nasa": np.where(d < 0, np.exp(-d / nasa["a_under"]) - 1, np.exp(d / nasa["a_over"]) - 1)})
    if lo:
        out["cov"] = ((df["RUL"] >= df[lo]) & (df["RUL"] <= df[hi])).astype(float)
        out["over"] = (df[lo] > df["RUL"]).astype(float)
        out["width"] = df[hi] - df[lo]
    return out.groupby("uid").sum().sort_index()


def _ratio(s: np.ndarray, idx: np.ndarray, col: int, n: np.ndarray) -> np.ndarray:
    return s[idx, col].sum(1) / n[idx].sum(1)


def bootstrap_ci(sums: dict[str, pd.DataFrame], B: int, seed: int, pairs: list[tuple[str, str]]) -> dict:
    """Percentil 2,5–97,5 % remuestreando motores; los pares comparan el mismo remuestreo."""
    rng = np.random.default_rng(seed)
    units = next(iter(sums.values())).index
    idx = rng.integers(0, len(units), (B, len(units)))
    stats, out = {}, {}
    for name, df in sums.items():
        df = df.loc[units]
        n, st = df["n"].to_numpy(), {}
        for c in df.columns.drop("n"):
            v = _ratio(df.to_numpy(), idx, df.columns.get_loc(c), n)
            st["rmse" if c == "se" else c] = np.sqrt(v) if c == "se" else v
        stats[name] = st
        out[name] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in st.items()}
    for a, b in pairs:
        out[f"{a} - {b}"] = {k: {"dif_ic95": [float(np.percentile(stats[a][k] - stats[b][k], 2.5)),
                                              float(np.percentile(stats[a][k] - stats[b][k], 97.5))]}
                             for k in stats[a] if k in stats[b]}
    return out


# ════════════════════════════════════════════════════════════════════════════
#  Política de inspección
# ════════════════════════════════════════════════════════════════════════════
def _trigger_table(df: pd.DataFrame, bound: str, H: float, tau: int) -> pd.DataFrame:
    """Una fila por motor: si la orden llegó tarde y cuánta vida se desperdició."""
    f = df.sort_values(["uid", "cycle"])
    hit = f[f[bound] <= H].groupby("uid", sort=True).first()["RUL"]
    units = f.groupby("uid", sort=True).size().index
    rul = hit.reindex(units)
    late = rul.isna() | (rul < tau)
    waste = (rul.fillna(0) - tau).clip(lower=0)
    return pd.DataFrame({"late": late.astype(float), "waste": waste, "rul_at_order": rul}, index=units)


def _sweep(df: pd.DataFrame, bound: str, grid: np.ndarray, tau: int) -> pd.DataFrame:
    rows = []
    for H in grid:
        r = _trigger_table(df, bound, float(H), tau)
        rows.append({"H": float(H), "late": r["late"].mean(), "waste": r["waste"].mean()})
    return pd.DataFrame(rows)


def _pick_H(sweep: pd.DataFrame, max_late: float) -> float:
    """El H más bajo (menos vida desperdiciada) que respeta la tasa de tardías; si ninguno, la menor tasa."""
    ok = sweep[sweep["late"] <= max_late]
    return float(ok["H"].min()) if len(ok) else float(sweep.sort_values(["late", "H"])["H"].iloc[0])


def _fixed_life(dev: pd.DataFrame, max_late: float, tau: int) -> dict[int, int]:
    """Línea de base sin modelo: inspeccionar en un vuelo fijo por clase, elegido con los motores de desarrollo."""
    e = dev.drop_duplicates("uid")[["uid", "Fc", "eol"]]
    return {int(fc): int(next((N for N in range(int(g["eol"].max()), 0, -1) if ((g["eol"] - N) < tau).mean() <= max_late), 1))
            for fc, g in e.groupby("Fc")}


def _fixed_eval(test: pd.DataFrame, life: dict[int, int], tau: int) -> pd.DataFrame:
    e = test.drop_duplicates("uid").set_index("uid").sort_index()
    rul = e["eol"] - e["Fc"].map(life)
    return pd.DataFrame({"late": (rul < tau).astype(float), "waste": (rul - tau).clip(lower=0), "rul_at_order": rul})


def _policy_ci(rows: pd.DataFrame, B: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rows), (B, len(rows)))
    return {c: [float(np.percentile(rows[c].to_numpy()[idx].mean(1), q)) for q in (2.5, 97.5)] for c in ("late", "waste")}


def _paired_ci(a: pd.DataFrame, b: pd.DataFrame, B: int, seed: int) -> dict:
    """IC 95 % de a − b remuestreando los mismos motores en ambas políticas."""
    a, b = a.sort_index(), b.sort_index()
    idx = np.random.default_rng(seed).integers(0, len(a), (B, len(a)))
    return {c: [float(np.percentile((a[c].to_numpy() - b[c].to_numpy())[idx].mean(1), q)) for q in (2.5, 97.5)] for c in ("late", "waste")}


def inspection(cfg: dict, dev: pd.DataFrame, test: pd.DataFrame, lo_test: np.ndarray, B: int) -> dict:
    pc = cfg["cvplus"]["inspection"]
    grid = np.arange(*pc["h_grid"])
    primary, max_late = pc["lead_flights"], pc["max_late_rate"]
    t = test.assign(lower=lo_test)
    out = {"lead_flights": primary, "max_late_rate": max_late, "by_tau": {}, "curves": {}}
    for tau in pc["sensitivity_lead_flights"]:
        res, chosen, per_engine = {}, {}, {}
        for name, col in {"mediana": "q50", "cota inferior": "lo"}.items():
            sw = _sweep(dev, col, grid, tau)
            H = _pick_H(sw, max_late)
            chosen[name] = H
            r = _trigger_table(t, "q50" if name == "mediana" else "lower", H, tau)
            per_engine[name] = r
            dv = sw.loc[sw["H"] == H].iloc[0]
            res[name] = {"H": H, "dev_late": float(dv["late"]), "dev_waste": float(dv["waste"]), "late": float(r["late"].mean()),
                         "late_engines": int(r["late"].sum()), "waste": float(r["waste"].mean()),
                         "ci": _policy_ci(r, B, cfg["seed"])}
            if tau == primary:
                out["curves"][name] = _sweep(t, "q50" if name == "mediana" else "lower", grid, tau).to_dict("records")
                out["curves"][f"{name} (desarrollo)"] = sw.to_dict("records")
        life = _fixed_life(dev, max_late, tau)
        fx = _fixed_eval(test, life, tau)
        res["vida fija por clase"] = {"H": None, "life": life, "late": float(fx["late"].mean()), "late_engines": int(fx["late"].sum()),
                                      "waste": float(fx["waste"].mean()), "ci": _policy_ci(fx, B, cfg["seed"])}
        res["cota inferior - vida fija"] = _paired_ci(per_engine["cota inferior"], fx, B, cfg["seed"])
        res["cota inferior - mediana"] = _paired_ci(per_engine["cota inferior"], per_engine["mediana"], B, cfg["seed"])
        out["by_tau"][str(tau)] = res
    return out


# ════════════════════════════════════════════════════════════════════════════
#  Etapa 2: evaluación
# ════════════════════════════════════════════════════════════════════════════
def run_eval(cfg: dict, paths: Paths) -> dict:
    cc, rc, ns, vc = cfg["conformal"], cfg["results"], cfg["nasa_score"], cfg["cvplus"]
    K, B = vc["folds"], vc["bootstrap"]
    a_o, a_u, target = cc["alpha_over"], cc["alpha_under"], 1 - cc["alpha"]
    preds = pd.read_parquet(paths.outputs / "cvplus_preds.parquet")
    oof = preds[(preds["role"] != "test") & (preds["cycle"] >= rc["eval_min_cycle"])].copy()
    candidates = {"cvplus": None, "cvplus_tramos": rc["rul_buckets"]}

    # 1) Decisiones con motores de desarrollo: se guardan antes de tocar la prueba
    dev_res, dev_frames = {}, {}
    for name, edges in candidates.items():
        g = _dev_pseudotest(oof, K, edges, a_o, a_u)
        dev_frames[name] = g
        dev_res[name] = M.summary(g["RUL"], g["q50"], g["lo"], g["hi"], ns)
    chosen = _choose_method(dev_res, a_o, target)
    decision = {"metodo_elegido": chosen, "desarrollo": dev_res, "regla": "tolerancia a sobreestimar, cercanía a la cobertura objetivo, menor ancho"}
    save_json(decision, paths.outputs / "cvplus_decision.json")
    log.info("Método elegido en desarrollo: %s · %s", chosen, dev_res[chosen])

    # 2) Prueba
    test, q = _test_matrices(preds, K, rc["eval_min_cycle"])
    y = test["RUL"]
    rows, sums, frames = {}, {}, {}
    for name, edges in candidates.items():
        lo, hi = CVPlus(oof, K, edges).bounds(q["q05"], test["q50"].to_numpy(), q["q95"], a_o, a_u)
        frames[name] = test.assign(lo=lo, hi=hi)
        rows[name] = M.summary(y, test["q50"], lo, hi, ns)
        sums[name] = engine_sums(frames[name], ns)
    rng = np.random.default_rng(cfg["seed"])
    units = sorted(test["uid"].unique())
    orders = [rng.permutation(units) for _ in range(cc["fleet_permutations"])]
    frames["cvplus_aci"], aci = _fleet_aci(test, q, CVPlus(oof, K, candidates[chosen]), cc, orders)
    rows["cvplus_aci"] = {**M.summary(y, test["q50"], frames["cvplus_aci"]["lo"], frames["cvplus_aci"]["hi"], ns), **aci}
    sums["cvplus_aci"] = engine_sums(frames["cvplus_aci"], ns)

    # Referencia: el pipeline publicado, recalculado en esta corrida
    base = pd.read_parquet(paths.outputs / "preds_catboost.parquet")
    base = base[base["cycle"] >= rc["eval_min_cycle"]]
    tft = pd.read_parquet(paths.outputs / "preds_tft.parquet")
    common = test[["uid", "cycle"]]
    base_cal = base[base["role"] == "calib"]
    tft_cal = tft[(tft["role"] == "calib") & (tft["cycle"] >= rc["eval_min_cycle"])]
    for name, frame, cal_df in (("base_catboost", base, base_cal), ("base_tft", tft, tft_cal)):
        te = frame[frame["role"] == "test"].merge(common, on=["uid", "cycle"]).sort_values(["uid", "cycle"]).reset_index(drop=True)
        mond = Calibrator(cal_df, rc["rul_buckets"])
        lo, hi = mond.bounds(te, a_o, a_u)
        frames[name] = te.assign(lo=lo, hi=hi)
        rows[name] = M.summary(te["RUL"], te["q50"], lo, hi, ns)
        sums[name] = engine_sums(frames[name], ns)
    ci = bootstrap_ci(sums, B, cfg["seed"], [(chosen, "base_catboost"), ("base_tft", "base_catboost"), ("base_tft", chosen)])

    # 3) Por subconjunto, por tramo y por clase
    use = frames[chosen]
    per_ds = {ds: M.summary(g["RUL"], g["q50"], g["lo"], g["hi"], ns) for ds, g in use.groupby("ds")}
    per_fc = {int(fc): M.summary(g["RUL"], g["q50"], g["lo"], g["hi"], ns) for fc, g in use.groupby("Fc")}
    buckets = M.by_bucket(use, rc["rul_buckets"], p="q50").to_dict("records")

    # 4) Política de inspección: umbrales de desarrollo, evaluación en prueba
    policy = inspection(cfg, dev_frames[chosen], test, frames[chosen]["lo"].to_numpy(), B)

    out = {"chosen": chosen, "development": dev_res, "test": rows, "ci": ci, "per_ds": per_ds, "per_fc": per_fc, "buckets": buckets,
           "inspection": policy, "target": target, "alpha_over": a_o, "alpha_under": a_u, "units_dev": int(oof["uid"].nunique()),
           "units_test": len(units), "bootstrap": B, "fit": json.loads((paths.outputs / "cvplus_fit.json").read_text(encoding="utf-8"))}
    save_json(out, paths.outputs / "cvplus.json")
    frames[chosen].to_parquet(paths.outputs / "cvplus_intervals.parquet")
    log.info("CV+ evaluado: %s", paths.outputs / "cvplus.json")
    return out
