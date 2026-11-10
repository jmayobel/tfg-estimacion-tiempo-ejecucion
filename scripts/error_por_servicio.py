#!/usr/bin/env python3
"""Error por servicio con máquinas conocidas (RQ1).

Usa el mismo esquema que train_baseline.py para las máquinas conocidas: GroupKFold(5) con
grupos por máquina, servicio y tamaño de carga, de modo que las repeticiones de una misma
medición nunca quedan repartidas entre entrenamiento y prueba. Calcula el MdAPE de cada servicio
para el modelo propuesto (LogLinear_Svc_Full) y para HistGBoost, el mejor de los modelos de
comparación. El MdAPE global de cada modelo debe coincidir con el de train_baseline.py
(11,1 % y 22,6 %).

Mismas variables, limpieza y modelos que train_baseline.py.

Uso: python3 scripts/error_por_servicio.py   (escribe output/error_por_servicio.txt)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_baseline as tb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "output" / "error_por_servicio.txt"
MODELOS = ("LogLinear_Svc_Full", "HistGBoost")


def mdape(y, p):
    return float(np.median(np.abs(p - y) / np.maximum(y, 1e-9)) * 100)


def main() -> None:
    df = tb.clean(tb.load()).reset_index(drop=True)
    num = tb.usable_num_features(df)
    X, y_log = tb.prepare_xy(df, num)
    y_true = df[tb.TARGET].to_numpy(dtype=float)
    models = tb.make_models(num)
    grupos = (
        df["hw_hostname_label"].astype(str)
        + "|"
        + df["service_key"].astype(str)
        + "|"
        + df["workload_scale"].astype(str)
    )

    pred = {}
    for name in MODELOS:
        p_log = cross_val_predict(models[name], X, y_log, cv=GroupKFold(5), groups=grupos, n_jobs=1)
        pred[name] = np.expm1(p_log)

    lines = [
        f"Dataset: {len(df)} filas | {df['hw_hostname_label'].nunique()} máquinas | "
        f"{df['service_key'].nunique()} servicios",
        "Esquema: GroupKFold(5) por máquina, servicio y tamaño de carga (máquinas conocidas).",
        "",
        f"  {'servicio':20s} {'n':>4s} " + " ".join(f"{m:>19s}" for m in MODELOS),
    ]
    for s in sorted(df["service_key"].unique()):
        m = (df["service_key"] == s).to_numpy()
        lines.append(f"  {s:20s} {m.sum():4d} " + " ".join(
            f"{mdape(y_true[m], pred[n][m]):18.1f}%" for n in MODELOS))
    lines.append(f"  {'Todos':20s} {len(df):4d} " + " ".join(
        f"{mdape(y_true, pred[n]):18.1f}%" for n in MODELOS))
    lines.append("")
    lines.append("Referencia: train_baseline.py, mismo esquema, LogLinear_Svc_Full 11.1 %, HistGBoost 22.6 %.")
    lines.append("MdAPE = error porcentual absoluto mediano, en segundos.")

    text = "\n".join(lines) + "\n"
    print(text)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text, encoding="utf-8")
    print(f"Guardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
