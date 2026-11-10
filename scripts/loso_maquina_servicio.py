#!/usr/bin/env python3
"""Apartar a la vez la máquina y el servicio (hardware y software no conocidos).

En cada parte se aparta una pareja (máquina h, servicio s): el modelo se entrena sin ninguna fila de
h y sin ninguna fila de s, y predice las ejecuciones de s en h. Como cada fila pertenece a una sola
pareja, todas reciben una predicción de un modelo que no ha visto ni su máquina ni su servicio. Es el
caso más exigente: del servicio solo se conoce su clase, y de la máquina, sus características y su
calibración.

Mismas variables, limpieza y modelos que train_baseline.py. Los LogLinear no tienen ecuación para el
servicio apartado y devuelven la mediana del entrenamiento, igual que Dummy.

Uso: python3 scripts/loso_maquina_servicio.py   (escribe output/loso_maquina_servicio.txt)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_baseline as tb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "output" / "loso_maquina_servicio.txt"
DESGLOSE = ("Dummy(median)", "RandomForest", "HistGBoost")


def main() -> None:
    df = tb.clean(tb.load()).reset_index(drop=True)
    num = tb.usable_num_features(df)
    X, y_log = tb.prepare_xy(df, num)
    y_true = df[tb.TARGET].to_numpy(dtype=float)
    hosts = df["hw_hostname_label"].to_numpy()
    svcs = df["service_key"].to_numpy()
    parejas = sorted(set(zip(hosts, svcs)))
    models = tb.make_models(num)

    lines = [
        f"Dataset: {len(df)} filas | {df['hw_hostname_label'].nunique()} máquinas | "
        f"{df['service_key'].nunique()} servicios | {len(parejas)} parejas (máquina, servicio) = partes",
        "",
        "=== Apartando a la vez la máquina y el servicio ===",
        f"  {'modelo':18s} {'MAE_log':>8s} {'R2_log':>7s} {'MdAPE%':>7s}",
    ]
    errores = {}
    for name, est in models.items():
        pred_log = np.full(len(df), np.nan)
        for h, s in parejas:
            test = (hosts == h) & (svcs == s)
            train = (hosts != h) & (svcs != s)
            model = clone(est).fit(X[train], y_log[train])
            pred_log[test] = model.predict(X[test])
        assert not np.isnan(pred_log).any()
        pred = np.expm1(pred_log)
        err = np.abs(pred - y_true) / np.maximum(y_true, 1e-9) * 100
        errores[name] = err
        lines.append(
            f"  {name:18s} {mean_absolute_error(y_log, pred_log):8.3f} "
            f"{r2_score(y_log, pred_log):7.3f} {np.median(err):7.1f}"
        )

    clase = df.groupby("service_key")[["svc_resource_profile", "svc_parallelism"]].first()
    lines += ["", "MdAPE % por servicio apartado (en todas las máquinas):"]
    lines.append(f"{'servicio':20s} {'clase':22s} " + " ".join(f"{n:>14s}" for n in DESGLOSE))
    for s in sorted(df["service_key"].unique()):
        c = f"{clase.loc[s, 'svc_resource_profile']}/{clase.loc[s, 'svc_parallelism']}"
        m = svcs == s
        lines.append(f"{s:20s} {c:22s} " + " ".join(f"{np.median(errores[n][m]):13.1f}%" for n in DESGLOSE))

    lines += ["", "MdAPE % por máquina apartada (todos sus servicios):"]
    lines.append(f"{'máquina':36s} " + " ".join(f"{n:>14s}" for n in DESGLOSE))
    for h in sorted(df["hw_hostname_label"].unique()):
        m = hosts == h
        lines.append(f"{h:36s} " + " ".join(f"{np.median(errores[n][m]):13.1f}%" for n in DESGLOSE))

    mejora = {n: int(sum(np.median(errores[n][svcs == s]) < np.median(errores["Dummy(median)"][svcs == s])
                         for s in sorted(df["service_key"].unique()))) for n in DESGLOSE[1:]}
    lines += ["", "Servicios en los que el modelo mejora a la mediana: "
              + ", ".join(f"{n} {k}/{df['service_key'].nunique()}" for n, k in mejora.items())]
    lines += ["", "MdAPE = error porcentual absoluto mediano, en segundos reales; MAE_log y R2_log, en log1p(s)."]

    text = "\n".join(lines) + "\n"
    print(text)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text, encoding="utf-8")
    print(f"Guardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
