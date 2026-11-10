#!/usr/bin/env python3
"""¿Ayuda tener otros servicios de la misma clase en el entrenamiento?

Complementa a loso_por_servicio.py. Solo la clase cpu/multi_thread tiene varios servicios
(ds-7zip, ffmpeg, sysbench). Cada uno se aparta y se predice entrenando con 0, 1 o 2 de
los otros servicios de su clase; los servicios de las demás clases entran siempre. Si el
error bajara de forma consistente al añadir servicios de la misma clase, medir más
servicios por clase permitiría predecir un servicio nuevo a partir de su clase.

Con 3 servicios y una sola clase, el resultado es orientativo, no concluyente.

Uso: python3 scripts/vecinos_clase.py   (escribe output/vecinos_clase.txt)
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_diagnostics as md  # noqa: E402
import train_baseline as tb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "output" / "vecinos_clase.txt"
CLASE = "cpu/multi_thread"
MODELS = ("Dummy(median)", "RandomForest", "HistGBoost")


def main() -> None:
    df = tb.clean(tb.load())
    X, y = md.get_xy(df)
    y_true = np.expm1(np.asarray(y))
    svc = df["service_key"].to_numpy()
    clase = (df["svc_resource_profile"] + "/" + df["svc_parallelism"]).to_numpy()
    grupo = sorted(set(svc[clase == CLASE]))
    models = {name: est for name, est in tb.make_models().items() if name in MODELS}

    lines = [
        f"Dataset: {len(df)} filas | clase {CLASE}: {grupo}",
        "MdAPE % del servicio apartado según qué otros servicios de su clase hay en el entrenamiento",
        "",
        f"{'apartado':10s} {'misma clase en entrenamiento':30s} " + " ".join(f"{n:>14s}" for n in models),
    ]
    for s in grupo:
        otros = [o for o in grupo if o != s]
        test = np.flatnonzero(svc == s)
        for k in range(len(otros) + 1):
            for vecinos in combinations(otros, k):
                fuera = sorted(set(otros) - set(vecinos))
                train = (svc != s) & ~np.isin(svc, fuera)
                errores = []
                for est in models.values():
                    pred = clone(est).fit(X[train], y[train]).predict(X.iloc[test])
                    err = np.abs(np.expm1(pred) - y_true[test]) / y_true[test] * 100
                    errores.append(np.median(err))
                etiqueta = ", ".join(vecinos) if vecinos else "(ninguno)"
                lines.append(
                    f"{s:10s} {etiqueta:30s} " + " ".join(f"{e:13.1f}%" for e in errores)
                )
        lines.append("")
    lines.append("La fila con los dos servicios de la clase reproduce loso_por_servicio.py.")

    text = "\n".join(lines) + "\n"
    print(text)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text, encoding="utf-8")
    print(f"Guardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
