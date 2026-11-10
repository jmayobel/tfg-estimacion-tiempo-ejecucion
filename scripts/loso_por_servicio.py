#!/usr/bin/env python3
"""Error por servicio al apartar cada servicio entero (software no conocido).

Es el esquema «GroupKFold por SERVICIO» de train_baseline.py, desglosado por servicio: en
cada parte se aparta un servicio completo y se predice solo con su clase (perfil de
recursos, paralelismo, tipo de magnitud, GPU), las características de la máquina, su
calibración y el tamaño de carga. El MdAPE global de cada modelo tiene que coincidir con
el de ese esquema en output/train_results.txt.

Se evalúan Dummy (referencia), RandomForest y HistGBoost. Los LogLinear no aparecen: sin
filas del servicio apartado caen al mismo valor que Dummy.

Uso: python3 scripts/loso_por_servicio.py   (escribe output/loso_por_servicio.txt)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_diagnostics as md  # noqa: E402
import train_baseline as tb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "output" / "loso_por_servicio.txt"
MODELS = ("Dummy(median)", "RandomForest", "HistGBoost")


def main() -> None:
    df = tb.clean(tb.load())
    X, y = md.get_xy(df)
    y_true = np.expm1(y)
    services = sorted(df["service_key"].unique())
    models = {name: est for name, est in tb.make_models().items() if name in MODELS}

    lines = [f"Dataset: {len(df)} filas | {len(services)} servicios | modelos: {list(models)}"]
    per_service = {}
    for name, est in models.items():
        pred = np.full(len(df), np.nan)
        for s in services:
            train = (df["service_key"] != s).to_numpy()
            test = np.flatnonzero(~train)
            # service_key se queda como columna: en la prueba su valor no se ha visto
            model = clone(est).fit(X[train], y[train])
            pred[test] = model.predict(X.iloc[test])
        err = np.abs(np.expm1(pred) - y_true) / y_true * 100
        per_service[name] = df.assign(err=err).groupby("service_key")["err"].median()
        lines.append(f"{name}: MdAPE global {np.median(err):.1f} %")

    clase = df.groupby("service_key")[["svc_resource_profile", "svc_parallelism"]].first()
    lines.append("")
    lines.append(f"{'servicio':20s} {'clase':22s} " + " ".join(f"{n:>14s}" for n in models))
    for s in services:
        c = f"{clase.loc[s, 'svc_resource_profile']}/{clase.loc[s, 'svc_parallelism']}"
        lines.append(f"{s:20s} {c:22s} " + " ".join(f"{per_service[n][s]:13.1f}%" for n in models))
    lines.append("")
    lines.append("MdAPE = error porcentual absoluto mediano, en segundos reales.")

    text = "\n".join(lines) + "\n"
    print(text)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text, encoding="utf-8")
    print(f"Guardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
