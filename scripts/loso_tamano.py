#!/usr/bin/env python3
"""¿Predice el modelo tamaños de carga que no ha visto?

Aparta un tamaño de carga entero (small, medium, large o xlarge) en todas las máquinas y lo
predice con lo demás. Se aparta por valor de carga, no por etiqueta: para cada servicio se toma
el valor de ese tamaño y van a prueba todas las filas con ese valor (también las de la recogida
antigua, sin etiqueta, que coincidan) y a entrenamiento todas las demás. Así el valor apartado no
aparece en el entrenamiento en ninguna máquina, a diferencia del esquema por celda, que aparta una
combinación de máquina, servicio y tamaño pero deja ese mismo valor en las otras máquinas.

- small apartado: extrapolar hacia tamaños menores.
- medium y large apartados: interpolar entre tamaños medidos.
- xlarge apartado: extrapolar hacia tamaños mayores.

Mismas variables, limpieza y modelos que train_baseline.py.

Uso: python3 scripts/loso_tamano.py   (escribe output/loso_tamano.txt)
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
OUT_PATH = ROOT / "output" / "loso_tamano.txt"
SIZES = ("small", "medium", "large", "xlarge")


def mdape(y, p):
    return float(np.median(np.abs(p - y) / np.maximum(y, 1e-9)) * 100)


def main() -> None:
    df = tb.clean(tb.load()).reset_index(drop=True)
    num = tb.usable_num_features(df)
    X, y_log = tb.prepare_xy(df, num)
    y_true = df[tb.TARGET].to_numpy(dtype=float)
    models = tb.make_models(num)
    svc = df["service_key"].to_numpy()
    val = df["workload_value"].to_numpy(dtype=float)

    # valor de cada tamaño para cada servicio (es el mismo en todas las máquinas)
    tabla = df[df["workload_scale"].isin(SIZES)].groupby(["service_key", "workload_scale"])["workload_value"]
    assert (tabla.nunique() == 1).all(), "un tamaño con varios valores en un servicio"
    valor = tabla.first().to_dict()

    lines = [f"Dataset: {len(df)} filas | {df['hw_hostname_label'].nunique()} máquinas | "
             f"{df['service_key'].nunique()} servicios", ""]
    resumen = {}
    por_servicio = {}
    for s in SIZES:
        test = np.array([valor.get((k, s)) == v for k, v in zip(svc, val)])
        train = ~test
        lines.append(f"=== Apartado {s}: {test.sum()} filas de prueba, {train.sum()} de entrenamiento ===")
        lines.append(f"  {'modelo':18s} {'MAE_log':>8s} {'R2_log':>7s} {'MdAPE%':>7s}")
        for name, est in models.items():
            model = clone(est).fit(X[train], y_log[train])
            p_log = model.predict(X[test])
            e = mdape(y_true[test], np.expm1(p_log))
            resumen[(s, name)] = e
            lines.append(f"  {name:18s} {mean_absolute_error(y_log[test], p_log):8.3f} "
                         f"{r2_score(y_log[test], p_log):7.3f} {e:7.1f}")
            if name == "LogLinear_Svc_Full":
                d = df[test].assign(err=np.abs(np.expm1(p_log) - y_true[test]) / y_true[test] * 100)
                por_servicio[s] = d.groupby("service_key")["err"].median()
        lines.append("")

    lines.append("Resumen MdAPE % (filas: modelo; columnas: tamaño apartado)")
    lines.append(f"  {'modelo':18s} " + " ".join(f"{s:>8s}" for s in SIZES))
    for name in models:
        lines.append(f"  {name:18s} " + " ".join(f"{resumen[(s, name)]:8.1f}" for s in SIZES))
    lines.append("")
    lines.append("LogLinear_Svc_Full, MdAPE % por servicio y tamaño apartado")
    servicios = sorted(df["service_key"].unique())
    lines.append(f"  {'servicio':20s} " + " ".join(f"{s:>8s}" for s in SIZES))
    for k in servicios:
        lines.append(f"  {k:20s} " + " ".join(
            f"{por_servicio[s].get(k, float('nan')):8.1f}" for s in SIZES))
    lines.append("")
    lines.append("Referencia: esquema por celda (train_baseline.py), LogLinear_Svc_Full 11.1 %, HistGBoost 22.6 %.")
    lines.append("MdAPE = error porcentual absoluto mediano, en segundos; MAE_log y R2_log, en log1p(s).")

    text = "\n".join(lines) + "\n"
    print(text)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text, encoding="utf-8")
    print(f"Guardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
