#!/usr/bin/env python3
"""Intervalos bootstrap (por host) para los exponentes de Amdahl.

Los exponentes de LogLinear_Svc_Phys se estiman con n=8 puntos de hardware, así
que su incertidumbre real viene de QUÉ hosts hay, no de cuántas filas. El
remuestreo correcto es el bootstrap por clúster: se sortean 8 hosts CON
reemplazo, se refitea el modelo por servicio y se repite B veces. IC = percentil
2.5–97.5 de la distribución resultante.

Advertencias honestas (para la memoria):
  * Con 8 clústeres el bootstrap es best-effort; los IC son aproximados y
    tienden a ser anchos — eso ES el resultado (n pequeña en el eje hardware).
  * pytorch-gpu se excluye de la tabla principal: es GPU-bound y proyectarlo
    sobre ejes de CPU da coeficientes sin sentido físico (se reporta aparte).
  * En cada réplica, un servicio puede quedarse con <2 hosts distintos (p. ej.
    pytorch-gpu solo existe en 2); esas réplicas se descartan para ese servicio
    y se reporta el nº efectivo de réplicas.

Uso: python3 scripts/bootstrap_exponents.py   (escribe output/bootstrap_exponents.txt)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_baseline as tb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "output" / "bootstrap_exponents.txt"

B = 1000
SEED = 0
GPU_BOUND = {"pytorch-gpu"}
# (clave en coeficientes, etiqueta, signo con el que se reporta)
EXPONENTS = [
    ("log_hw_cpu_cores_logical", "bc (cores)", -1.0),
    ("log_hw_ram_gb", "br (RAM)", -1.0),
    ("log_workload_value", "bw (carga)", +1.0),
]


def fit_exponents(df) -> dict[str, dict[str, float]]:
    """Ajusta LogLinear_Svc_Phys y devuelve {servicio: {exponente: valor}}."""
    X, y_log = tb.prepare_xy(df, tb.usable_num_features(df))
    m = tb.LogLinearServiceRegressor(alpha=1.0, use_calibration=False)
    m.fit(X, y_log)
    out = {}
    for svc, coefs in m.get_physical_coefficients().items():
        out[svc] = {label: sign * coefs.get(key, 0.0) for key, label, sign in EXPONENTS}
    return out


def main():
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    df = tb.clean(tb.load())
    hosts = sorted(df["hw_hostname_label"].unique())
    services = sorted(df["service_key"].unique())
    log(f"Dataset: {len(df)} filas | {len(hosts)} hosts | {len(services)} servicios")
    log(f"Bootstrap por HOST: B={B}, semilla={SEED}, IC percentil 95%")

    point = fit_exponents(df)

    rng = np.random.default_rng(SEED)
    by_host = {h: df[df["hw_hostname_label"] == h] for h in hosts}
    # {servicio: {exponente: [valores]}}
    dist = {s: {label: [] for _, label, _ in EXPONENTS} for s in services}

    import pandas as pd

    for _ in range(B):
        draw = rng.choice(hosts, size=len(hosts), replace=True)
        df_b = pd.concat([by_host[h] for h in draw], ignore_index=True)
        # un servicio solo es estimable si la réplica conserva >=2 hosts distintos
        estimable = {
            s for s, g in df_b.groupby("service_key")["hw_hostname_label"] if g.nunique() >= 2
        }
        exps = fit_exponents(df_b)
        for s, vals in exps.items():
            if s in estimable:
                for label, v in vals.items():
                    dist[s][label].append(v)

    log(f"\n{'=' * 100}")
    log("EXPONENTES FÍSICOS CON IC BOOTSTRAP 95% (por host) — LogLinear_Svc_Phys")
    log("Modelo: log(T) ~ C - bc*log(cores) - br*log(RAM) + bw*log(workload)")
    log(f"{'=' * 100}")
    hdr = f"  {'servicio':22s} {'B_ef':>5s}"
    for _, label, _ in EXPONENTS:
        hdr += f" | {label:^26s}"
    log(hdr)
    sub = f"  {'':22s} {'':>5s}"
    for _ in EXPONENTS:
        sub += f" | {'punto  [IC 2.5–97.5]':^26s}"
    log(sub)
    log("-" * 100)

    md_rows = []  # tabla lista para pegar en la memoria
    for s in services:
        main_table = s not in GPU_BOUND
        n_eff = len(next(iter(dist[s].values())))
        row = f"  {s:22s} {n_eff:5d}"
        md = f"| `{s}` |"
        for _, label, _ in EXPONENTS:
            vals = np.array(dist[s][label])
            p = point[s][label]
            if len(vals) < 50:
                row += f" | {p:6.3f}  [insuficiente]     "
                md += f" {p:.3f} (IC n/d) |"
                continue
            lo, hi = np.percentile(vals, [2.5, 97.5])
            row += f" | {p:6.3f}  [{lo:7.3f},{hi:7.3f}] "
            md += f" {p:.3f} [{lo:.3f}, {hi:.3f}] |"
        if not main_table:
            row += "   <- GPU-bound, fuera de la tabla principal"
        log(row)
        md_rows.append((main_table, md))

    log("\nLectura: si el IC de bc/br cruza 0, el signo del efecto no está resuelto")
    log("con 8 puntos de hardware. bw se estima con variación intra-host (rejilla de")
    log("escalas), por eso sus IC son mucho más estrechos que los de bc/br.")

    log("\n--- Tabla markdown (memoria) ---")
    log("| servicio | bc (cores) [IC 95%] | br (RAM) [IC 95%] | bw (carga) [IC 95%] |")
    log("|---|---|---|---|")
    for main_table, md in md_rows:
        if main_table:
            log(md)
    log("\nExcluido de la tabla principal (GPU-bound, ejes de CPU sin sentido físico):")
    for main_table, md in md_rows:
        if not main_table:
            log(md)

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nGuardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
