#!/usr/bin/env python3
"""Simulación few-shot: ¿se puede predecir un SERVICIO nuevo con un fingerprint?

El esquema zero-shot (GroupKFold por SERVICIO) fracasa (~85-89% MdAPE): un
servicio sin filas es un punto aislado en el espacio de workloads. Este script
simula el protocolo pragmático de la literatura (PARIS, Ernest): ejecutar el
servicio nuevo SOLO a escala pequeña en UN host de referencia (el *fingerprint*)
y predecir desde ahí el resto de celdas (otras escalas y otros hardwares).

Protocolo (leave-one-service-out, few-shot):
  * fingerprint del servicio s = sus filas `small` en el host de referencia
    (intel-base, el baseline cloud).
  * test = todas las demás filas de s (mismo host a otras escalas + resto de
    hosts a todas las escalas).
  * el resto de servicios entra completo en el entrenamiento.

Escalera de predictores (cada peldaño añade un ingrediente — la atribución
importa más que el mejor número):
  B0  zero-shot ML (HGB sin fingerprint)      — el fracaso conocido, de referencia
  B1  constante: T = T_fp                     — solo el fingerprint
  B2  + ley de carga: T_fp·(w/w_fp)^bw̄        — extrapola la escala (Ernest)
  B3  + transferencia hardware:
      B2·(cores_h/cores_ref)^(−bc̄)·(ram_h/ram_ref)^(−br̄)
  ML  few-shot: HGB / RF con el fingerprint añadido al entrenamiento

Los exponentes pooled (bw̄, bc̄, br̄) son la MEDIANA de los exponentes por
servicio de LogLinear_Svc_Phys ajustado SOLO con los servicios de entrenamiento
(sin s → sin fuga; sin pytorch-gpu → exponentes de CPU sin sentido).

pytorch-gpu se omite como servicio evaluado: no tiene fingerprint posible en el
host de referencia (no corre ahí) y está declarado fuera del alcance CPU.

Uso: python3 scripts/fewshot_fingerprint.py  (escribe output/fewshot_fingerprint.txt)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_baseline as tb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "output" / "fewshot_fingerprint.txt"

REF_HOST = "azure-vm-dataset-creator-vm"
FP_SCALE = "small"
GPU_BOUND = {"pytorch-gpu"}


def mdape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.median(np.abs(y_pred - y_true) / np.maximum(y_true, 1e-9)) * 100)


def pooled_exponents(df_train: pd.DataFrame) -> tuple[float, float, float]:
    """Mediana de (bc, br, bw) por servicio de LogLinear_Svc_Phys sobre df_train."""
    X, y = tb.prepare_xy(df_train, tb.usable_num_features(df_train))
    m = tb.LogLinearServiceRegressor(alpha=1.0, use_calibration=False).fit(X, y)
    coefs = m.get_physical_coefficients()
    bc, br, bw = [], [], []
    for svc, c in coefs.items():
        if svc in GPU_BOUND:
            continue
        bc.append(-c.get("log_hw_cpu_cores_logical", 0.0))
        br.append(-c.get("log_hw_ram_gb", 0.0))
        bw.append(c.get("log_workload_value", 0.0))
    return float(np.median(bc)), float(np.median(br)), float(np.median(bw))


def ml_predict(model_name: str, num, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    Xtr, ytr = tb.prepare_xy(train.reset_index(drop=True), num)
    Xte, _ = tb.prepare_xy(test.reset_index(drop=True), num)
    m = tb.make_models(num)[model_name].fit(Xtr, ytr)
    return np.expm1(m.predict(Xte))


def main():
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    df = tb.clean(tb.load())
    num = tb.usable_num_features(df)
    services = [s for s in sorted(df["service_key"].unique()) if s not in GPU_BOUND]
    log(f"Dataset: {len(df)} filas | ref={REF_HOST} | fingerprint=escala '{FP_SCALE}'")
    log(f"Servicios evaluados: {services} (pytorch-gpu omitido: sin fingerprint en ref)")

    per_svc = []
    all_true, all_pred = {}, {}  # por método, para el pooled global
    methods = ["B0-zeroshot", "B1-const", "B2-carga", "B3-hw", "FS-HGB", "FS-RF"]
    for m in methods:
        all_true[m], all_pred[m] = [], []
    split_data = {m: {"same": ([], []), "cross": ([], [])} for m in ("B3-hw", "FS-HGB")}

    for s in services:
        rows_s = df[df["service_key"] == s]
        is_fp = (rows_s["hw_hostname_label"] == REF_HOST) & (rows_s["workload_scale"] == FP_SCALE)
        fp = rows_s[is_fp]
        if fp.empty:
            log(f"\n[{s}] SIN fingerprint en {REF_HOST} — omitido")
            continue
        test = rows_s.drop(fp.index)
        train_pool = df.drop(rows_s.index)

        bc, br, bw = pooled_exponents(train_pool)
        t_fp = float(fp[tb.TARGET].median())
        w_fp = float(fp["workload_value"].median())
        cores_ref = float(fp["hw_cpu_cores_logical"].iloc[0])
        ram_ref = float(fp["hw_ram_gb"].iloc[0])

        y = test[tb.TARGET].to_numpy(float)
        w = test["workload_value"].to_numpy(float)
        ch = test["hw_cpu_cores_logical"].to_numpy(float)
        rh = test["hw_ram_gb"].to_numpy(float)

        preds = {
            "B0-zeroshot": ml_predict("HistGBoost", num, train_pool, test),
            "B1-const": np.full(len(test), t_fp),
            "B2-carga": t_fp * (w / w_fp) ** bw,
            "B3-hw": (
                t_fp * (w / w_fp) ** bw * (ch / cores_ref) ** (-bc) * (rh / ram_ref) ** (-br)
            ),
            "FS-HGB": ml_predict("HistGBoost", num, pd.concat([train_pool, fp]), test),
            "FS-RF": ml_predict("RandomForest", num, pd.concat([train_pool, fp]), test),
        }

        errs = {m: mdape(y, p) for m, p in preds.items()}
        per_svc.append((s, len(fp), len(test), (bc, br, bw), errs))
        for m in methods:
            all_true[m].append(y)
            all_pred[m].append(preds[m])

        same = (test["hw_hostname_label"] == REF_HOST).to_numpy()
        for m in split_data:
            split_data[m]["same"][0].append(y[same])
            split_data[m]["same"][1].append(preds[m][same])
            split_data[m]["cross"][0].append(y[~same])
            split_data[m]["cross"][1].append(preds[m][~same])

        log(
            f"\n[{s}] fp={len(fp)} filas (T_fp={t_fp:.0f}s, w_fp={w_fp:g}) | "
            f"test={len(test)} | exponentes pooled bc={bc:.2f} br={br:.2f} bw={bw:.2f}"
        )
        for m in methods:
            log(f"    {m:18s} MdAPE={errs[m]:6.1f}%")

    log(f"\n{'=' * 78}")
    log("RESUMEN — MdAPE por servicio (test = todo menos el fingerprint)")
    hdr = f"  {'servicio':20s}" + "".join(f" {m:>11s}" for m in methods)
    log(hdr)
    for s, nfp, nte, _, errs in per_svc:
        log(f"  {s:20s}" + "".join(f" {errs[m]:11.1f}" for m in methods))
    log("  " + "-" * (20 + 12 * len(methods)))
    med = {m: float(np.median([e[4][m] for e in per_svc])) for m in methods}
    pooled = {m: mdape(np.concatenate(all_true[m]), np.concatenate(all_pred[m])) for m in methods}
    log(f"  {'MEDIANA de servicios':20s}" + "".join(f" {med[m]:11.1f}" for m in methods))
    log(f"  {'POOLED (filas)':20s}" + "".join(f" {pooled[m]:11.1f}" for m in methods))

    log("\nDESGLOSE (pooled) — extrapolar escala en el mismo host vs transferir a otro host:")
    for m, d in split_data.items():
        e_same = mdape(np.concatenate(d["same"][0]), np.concatenate(d["same"][1]))
        e_cross = mdape(np.concatenate(d["cross"][0]), np.concatenate(d["cross"][1]))
        log(
            f"  {m:14s} mismo host (otras escalas): {e_same:5.1f}%   "
            f"otros hosts: {e_cross:5.1f}%"
        )

    log("\nLeyenda: B0 = zero-shot (sin fingerprint, el fracaso conocido). B1-B3 = analítico")
    log("anclado al fingerprint (constante → +ley de carga bw̄ → +transferencia hardware")
    log("bc̄/br̄, exponentes pooled de los servicios de entrenamiento). FS = ML few-shot")
    log("(fingerprint añadido al entrenamiento).")

    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nGuardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
