#!/usr/bin/env python3
"""Por qué algunos intervalos bootstrap del exponente de la memoria son tan anchos.

Repite el bootstrap por máquina de bootstrap_exponents.py (misma semilla, mismas réplicas, mismo
modelo LogLinear_Svc_Phys) y, para cada servicio y réplica, guarda además la desviación típica del
logaritmo de la memoria que usa el escalador del modelo.

busqueda-caminos y sysbench solo se midieron en las seis máquinas x86, y cuatro de ellas tienen casi
la misma memoria (unos 15,5 GB). En las réplicas en las que solo salen esas cuatro, la memoria apenas
varía; al deshacer la estandarización (exponente = coeficiente / desviación típica) se divide entre un
número casi nulo y el exponente se dispara. El script da el intervalo con todas las réplicas (tiene que
coincidir con output/bootstrap_exponents.txt) y sin las réplicas con la memoria casi constante
(desviación típica del logaritmo < 0,05).

Uso: python3 scripts/bootstrap_ram.py   (escribe output/bootstrap_ram.txt)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_baseline as tb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "output" / "bootstrap_ram.txt"
B = 1000
SEED = 0
UMBRAL = 0.05
RAM = "log_hw_ram_gb"


def ajuste(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """{servicio: (exponente de la memoria, desviación típica del log de la memoria)}."""
    X, y_log = tb.prepare_xy(df, tb.usable_num_features(df))
    m = tb.LogLinearServiceRegressor(alpha=1.0, use_calibration=False).fit(X, y_log)
    out = {}
    for svc, info in m.models.items():
        feats = info["feature_names"]
        if RAM not in feats:
            out[svc] = (0.0, float("nan"))  # como bootstrap_exponents.py: sin la variable, 0
            continue
        i = feats.index(RAM)
        sigma = float(info["scaler"].scale_[i])
        w = float(info["model"].coef_[i])
        out[svc] = (-w / sigma if sigma > 1e-6 else 0.0, sigma)
    return out


def main() -> None:
    df = tb.clean(tb.load())
    hosts = sorted(df["hw_hostname_label"].unique())
    services = sorted(df["service_key"].unique())
    by_host = {h: df[df["hw_hostname_label"] == h] for h in hosts}
    rng = np.random.default_rng(SEED)
    datos = {s: [] for s in services}
    for _ in range(B):
        draw = rng.choice(hosts, size=len(hosts), replace=True)
        df_b = pd.concat([by_host[h] for h in draw], ignore_index=True)
        estimable = {s for s, g in df_b.groupby("service_key")["hw_hostname_label"] if g.nunique() >= 2}
        for s, (br, sigma) in ajuste(df_b).items():
            if s in estimable:
                datos[s].append((br, sigma))

    maquinas = df.groupby("service_key")["hw_hostname_label"].nunique()
    lines = [
        f"Dataset: {len(df)} filas | bootstrap por máquina, B={B}, semilla={SEED}, IC percentil 95 %",
        f"Réplicas con la memoria casi constante: desviación típica de log(RAM) < {UMBRAL}",
        "",
        f"  {'servicio':20s} {'máq.':>4s} {'B_ef':>5s} {'IC b_r (todas)':>22s} "
        f"{'casi const.':>11s} {'IC b_r (sin ellas)':>22s}",
    ]
    for s in services:
        arr = np.array(datos[s], dtype=float)
        br, sigma = arr[:, 0], arr[:, 1]
        lo, hi = np.percentile(br, [2.5, 97.5])
        casi = np.nan_to_num(sigma, nan=np.inf) < UMBRAL
        if (~casi).sum() >= 50:
            lo2, hi2 = np.percentile(br[~casi], [2.5, 97.5])
            sin = f"[{lo2:8.3f};{hi2:8.3f}]"
        else:
            sin = "insuficiente"
        lines.append(f"  {s:20s} {maquinas[s]:4d} {len(br):5d} [{lo:8.3f};{hi:8.3f}] "
                     f"{casi.sum():5d} ({casi.mean() * 100:4.1f} %) {sin:>22s}")
    lines += ["", "b_r = exponente de la memoria (signo como en output/bootstrap_exponents.txt)."]

    text = "\n".join(lines) + "\n"
    print(text)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text, encoding="utf-8")
    print(f"Guardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
