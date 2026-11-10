#!/usr/bin/env python3
"""A/B factorial limpio del efecto de `workload_scale`.

El PR #14 cambió DOS cosas a la vez: (a) quitó la categórica `workload_scale`
de CAT_FEATURES y (b) quitó el filtro de filas `unknown` de clean() (+20 filas
de intel-base). Este script separa ambos efectos con un factorial 2×2 a N
controlada:

    celda  feature workload_scale   filas unknown    N
    A      ON                       excluidas        clean() - 20   (config antigua)
    B      OFF                      excluidas        clean() - 20   (aísla la feature)
    C      OFF                      incluidas        clean()        (config actual)
    D      ON                       incluidas        clean()        (aísla las filas)

N se calcula en cada ejecución (572 filas tras clean() desde que se descartan las
ejecuciones x86 emuladas en ARM; antes, 588).

Nota de diseño: LogLinearServiceRegressor no consume categóricas (su matriz de
diseño son solo ejes log-numéricos), por lo que entre A↔B y entre D↔C debe dar
resultados IDÉNTICOS — el script lo verifica empíricamente. Cualquier cambio de
LogLinear entre configs es atribuible únicamente a las +20 filas.

Uso: python3 scripts/ab_workload_scale.py   (escribe output/ab_workload_scale.txt)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_baseline as tb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "output" / "ab_workload_scale.txt"

# La lista actual de train_baseline NO incluye workload_scale (post PR #14).
CAT_WITHOUT = list(tb.CAT_FEATURES)
CAT_WITH = ["service_key", "workload_scale"] + [c for c in CAT_WITHOUT if c != "service_key"]

CELLS = {
    # nombre: (con_feature, incluir_unknown)
    "A (feature ON,  sin unknown)": (True, False),
    "B (feature OFF, sin unknown)": (False, False),
    "C (feature OFF, con unknown)": (False, True),
    "D (feature ON,  con unknown)": (True, True),
}


def evaluate_cell(df, with_feature: bool, log):
    tb.CAT_FEATURES = CAT_WITH if with_feature else CAT_WITHOUT
    num = tb.usable_num_features(df)
    X, y_log = tb.prepare_xy(df, num)
    y_true = df[tb.TARGET].to_numpy(dtype=float)

    run_group = (
        df["hw_hostname_label"].astype(str)
        + "|"
        + df["service_key"].astype(str)
        + "|"
        + df["workload_scale"].astype(str)
    )
    schemes = {
        "random 5-fold": (KFold(5, shuffle=True, random_state=0), None),
        "interpolación (host,servicio,escala)": (GroupKFold(5), run_group),
        "por HOST (HW no visto)": (
            GroupKFold(df["hw_hostname_label"].nunique()),
            df["hw_hostname_label"],
        ),
    }
    results = {}
    for sname, (cv, groups) in schemes.items():
        log(f"\n  --- {sname} ---")
        log(f"  {'modelo':20s} {'MAE_log':>8s} {'R2_log':>7s} {'MdAPE%':>7s}")
        for mname, model in tb.make_models(num).items():
            pred_log = cross_val_predict(model, X, y_log, cv=cv, groups=groups, n_jobs=1)
            pred = np.expm1(pred_log)
            mae = mean_absolute_error(y_log, pred_log)
            r2 = r2_score(y_log, pred_log)
            mdape = float(np.median(np.abs(pred - y_true) / np.maximum(y_true, 1e-9)) * 100)
            results[(sname, mname)] = (mae, r2, mdape)
            log(f"  {mname:20s} {mae:8.3f} {r2:7.3f} {mdape:7.1f}")
    return results


def main():
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    df_all = tb.clean(tb.load())
    n_unknown = int((df_all["workload_scale"] == "unknown").sum())
    log(f"Filas tras clean(): {len(df_all)} (de ellas {n_unknown} con escala 'unknown')")

    all_results = {}
    n_rows = {}
    for cell, (with_feature, include_unknown) in CELLS.items():
        df = df_all if include_unknown else df_all[df_all["workload_scale"] != "unknown"]
        df = df.reset_index(drop=True)
        n_rows[cell] = len(df)
        log(
            f"\n{'=' * 70}\nCELDA {cell} — {len(df)} filas, feature "
            f"workload_scale {'ON' if with_feature else 'OFF'}"
        )
        all_results[cell] = evaluate_cell(df, with_feature, log)

    # Verificación: LogLinear debe ser idéntico dentro de cada par a N constante
    # (no consume categóricas). Si difiere, la implementación cambió y el
    # razonamiento de atribución de este A/B ya no vale.
    log(f"\n{'=' * 70}\nVERIFICACIÓN — LogLinear es insensible a la feature categórica:")
    for a, b in [
        ("A (feature ON,  sin unknown)", "B (feature OFF, sin unknown)"),
        ("D (feature ON,  con unknown)", "C (feature OFF, con unknown)"),
    ]:
        n = n_rows[a]
        for model in ["LogLinear_Svc_Full", "LogLinear_Svc_Phys"]:
            diffs = [
                abs(all_results[a][(s, model)][i] - all_results[b][(s, model)][i])
                for s in {k[0] for k in all_results[a]}
                for i in range(3)
            ]
            verdict = "SÍ" if max(diffs) < 1e-9 else f"NO (max diff {max(diffs):.3g})"
            log(f"  N={n} {model:20s} idéntico ON/OFF: {verdict}")

    # Resumen de atribución sobre el esquema central (por HOST)
    log(f"\n{'=' * 70}\nRESUMEN — GroupKFold por HOST (esquema central del TFG):")
    log(f"  {'celda':30s} {'N':>4s} {'modelo':20s} {'R2_log':>7s} {'MdAPE%':>7s}")
    for cell in CELLS:
        for model in ["LogLinear_Svc_Full", "RandomForest", "HistGBoost"]:
            _, r2, mdape = all_results[cell][("por HOST (HW no visto)", model)]
            log(f"  {cell:30s} {n_rows[cell]:4d} {model:20s} {r2:7.3f} {mdape:7.1f}")

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nGuardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
