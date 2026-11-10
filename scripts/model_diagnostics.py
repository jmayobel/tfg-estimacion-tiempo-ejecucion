#!/usr/bin/env python3
"""Deep diagnostics for the execution-time regressor: what works, what to add.

Produces a console report + figures in output/diagnostics/:
  * permutation importance for predicting an UNSEEN host (does it use real HW?)
  * per-host leave-one-host-out error (which hardware is underrepresented?)
  * per-service interpolation error (which workloads are hard?)
  * LEARNING CURVE: unseen-host error vs number of training hosts
    (the key evidence for whether adding more hosts will help)
  * collinearity of hardware features (can effects be separated?)
  * predicted-vs-actual, residuals, and per-service scaling curves
"""

from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score

from train_baseline import (
    load,
    clean,
    make_ordinal_encoder,
    prepare_xy,
    LogLinearServiceRegressor,
    CAT_FEATURES,
    NUM_FEATURES,
    TARGET,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)


class DropConstantColumns(BaseEstimator, TransformerMixin):
    """Drop columns that are constant or all-NaN within the *training* fold.

    HistGradientBoosting's per-fold binning crashes (sliding_window_view on a
    single distinct value) on tiny subsets — e.g. the learning curve trains on
    2 hosts where a hardware feature like cores/RAM has just one value. A constant
    feature carries no information for the tree anyway, so dropping it is a no-op
    for predictions and makes the small-subset fits robust.
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.keep_ = [j for j in range(X.shape[1]) if np.unique(X[~np.isnan(X[:, j]), j]).size >= 2]
        if not self.keep_:  # safety: never drop everything
            self.keep_ = list(range(X.shape[1]))
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]


def hgb_pipeline():
    pre = ColumnTransformer(
        [
            ("cat", make_ordinal_encoder(), CAT_FEATURES),
            ("num", "passthrough", NUM_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            ("drop", DropConstantColumns()),
            ("m", HistGradientBoostingRegressor(random_state=0, max_iter=500, learning_rate=0.05)),
        ]
    )


def mdape(y, p):
    return float(np.median(np.abs(p - y) / np.maximum(y, 1e-9)) * 100)


def get_xy(df):
    return prepare_xy(df, NUM_FEATURES)


def per_host_loho(df):
    """Leave-one-host-out error for BOTH models backing the memoir's central table.

    The white-box vs black-box contrast is argued per host (isolated hosts are where
    the structural model wins), so this must emit the log-linear column too — not just
    HistGBoost. The pooled row doubles as a consistency check: it has to reproduce the
    'GroupKFold por HOST' figures of train_baseline.py.
    """
    print("\n=== Error por HOST (leave-one-host-out) — qué hardware es más difícil ===")
    models = {
        "LogLinear_Svc_Full": lambda: LogLinearServiceRegressor(alpha=1.0, use_calibration=True),
        "HistGBoost": hgb_pipeline,
    }
    hosts = sorted(df["hw_hostname_label"].unique())
    X, y = get_xy(df)
    y_s = np.expm1(y)
    oof = {name: np.full(len(df), np.nan) for name in models}

    print(f"  {'host':34s} {'n':>4s} | " + " | ".join(f"{n:>27s}" for n in models))
    print(
        f"  {'':34s} {'':>4s} | "
        + " | ".join(f"{'MdAPE':>9s} {'R2_log':>8s} {'MAE_log':>8s}" for _ in models)
    )
    rows = []
    for h in hosts:
        mask_tr = (df.hw_hostname_label != h).to_numpy()
        idx_te = np.flatnonzero(~mask_tr)
        cells = {}
        for name, make in models.items():
            fitted = make().fit(X[mask_tr], y[mask_tr])
            p = fitted.predict(X.iloc[idx_te])
            oof[name][idx_te] = p
            cells[name] = (
                mdape(y_s[idx_te], np.expm1(p)),
                r2_score(y[idx_te], p),
                mean_absolute_error(y[idx_te], p),
            )
        rows.append((h, len(idx_te), cells))
        print(
            f"  {h:34s} {len(idx_te):4d} | "
            + " | ".join(
                f"{cells[n][0]:8.1f}% {cells[n][1]:8.2f} {cells[n][2]:8.3f}" for n in models
            )
        )

    print(f"  {'-' * 34} {'-' * 4} | " + " | ".join("-" * 27 for _ in models))
    print(
        f"  {'media (peso igual por host)':34s} {'':4s} | "
        + " | ".join(
            f"{np.mean([r[2][n][0] for r in rows]):8.1f}% {'':8s} "
            f"{np.mean([r[2][n][2] for r in rows]):8.3f}"
            for n in models
        )
    )
    print(
        f"  {'agregado por filas (pooled)':34s} {len(df):4d} | "
        + " | ".join(
            f"{mdape(y_s, np.expm1(oof[n])):8.1f}% {r2_score(y, oof[n]):8.3f} "
            f"{mean_absolute_error(y, oof[n]):8.3f}"
            for n in models
        )
    )
    print(
        "  El pooled debe coincidir con el bloque 'GroupKFold por HOST' de "
        "train_baseline.py;\n  la media con peso igual es la que no infrarrepresenta "
        "los hosts difíciles."
    )
    return rows


def per_service_interp(df):
    print("\n=== Error por SERVICIO (interpolación, OOF por grupo) ===")
    X, y = get_xy(df)
    grp = (
        df.hw_hostname_label.astype(str)
        + "|"
        + df.service_key.astype(str)
        + "|"
        + df.workload_scale.astype(str)
    )
    pred = cross_val_predict(hgb_pipeline(), X, y, cv=GroupKFold(5), groups=grp, n_jobs=1)
    df = df.assign(_pred=np.expm1(pred), _true=df[TARGET])
    for s, g in df.groupby("service_key"):
        print(f"  {s:22s} n={len(g):3d}  MdAPE={mdape(g._true.values, g._pred.values):5.1f}%")
    return df


def learning_curve(df):
    print("\n=== CURVA DE APRENDIZAJE: error en host no visto vs nº de hosts de entrenamiento ===")
    hosts = sorted(df["hw_hostname_label"].unique())
    pts = []
    for k in range(2, len(hosts)):
        errs = []
        for train_hosts in itertools.combinations(hosts, k):
            test_hosts = [h for h in hosts if h not in train_hosts]
            tr = df[df.hw_hostname_label.isin(train_hosts)]
            for th in test_hosts:
                te = df[df.hw_hostname_label == th]
                Xtr, ytr = get_xy(tr)
                Xte, yte = get_xy(te)
                m = hgb_pipeline().fit(Xtr, ytr)
                errs.append(mdape(np.expm1(yte), np.expm1(m.predict(Xte))))
        pts.append((k, float(np.mean(errs)), float(np.std(errs))))
        print(
            f"  {k} hosts entrenando -> MdAPE medio en host no visto: "
            f"{np.mean(errs):5.1f}% (±{np.std(errs):.1f})"
        )
    return pts


def feature_importance(df):
    print("\n=== Importancia de features (permutación, para host no visto) ===")
    hosts = sorted(df["hw_hostname_label"].unique())
    feats = CAT_FEATURES + NUM_FEATURES
    imp = np.zeros(len(feats))
    for h in hosts:
        tr, te = df[df.hw_hostname_label != h], df[df.hw_hostname_label == h]
        Xtr, ytr = get_xy(tr)
        Xte, yte = get_xy(te)
        m = hgb_pipeline().fit(Xtr, ytr)
        r = permutation_importance(
            m, Xte, yte, n_repeats=10, random_state=0, scoring="r2", n_jobs=1
        )
        imp += r.importances_mean
    imp /= len(hosts)
    order = np.argsort(imp)[::-1]
    ranked = [(feats[i], imp[i]) for i in order]
    for f, v in ranked[:15]:
        print(f"  {f:26s} {v:+.3f}")
    return ranked


def collinearity(df):
    print("\n=== Colinealidad entre features de hardware numéricas ===")
    hw_num = [c for c in NUM_FEATURES if c.startswith("hw_")]
    # one row per host
    per_host = df.groupby("hw_hostname_label")[hw_num].first()
    corr = per_host.corr()
    high = []
    for i in range(len(hw_num)):
        for j in range(i + 1, len(hw_num)):
            c = corr.iloc[i, j]
            if abs(c) > 0.9:
                high.append((hw_num[i], hw_num[j], c))
    if high:
        print("  Pares casi colineales (|r|>0.9) — el modelo no puede separar sus efectos:")
        for a, b, c in high:
            print(f"    {a} ~ {b}: r={c:+.2f}")
    else:
        print("  Sin pares con |r|>0.9.")
    return corr


# ---------- figures ----------
def fig_pred_vs_actual(df_pred):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df_pred._true, df_pred._pred, s=18, alpha=0.5)
    lim = [df_pred._true.min() * 0.8, df_pred._true.max() * 1.2]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tiempo medido (s)")
    ax.set_ylabel("Tiempo predicho (s)")
    fig.tight_layout()
    fig.savefig(OUT / "pred_vs_actual.png", dpi=120)
    plt.close(fig)


def fig_learning_curve(pts):
    k = [p[0] for p in pts]
    m = [p[1] for p in pts]
    s = [p[2] for p in pts]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(k, m, yerr=s, marker="o", capsize=4)
    ax.set_xlabel("nº de hosts en entrenamiento")
    ax.set_ylabel("MdAPE en host NO visto (%)")
    ax.set_title("¿Ayuda añadir más hosts? (curva de aprendizaje)")
    ax.set_xticks(k)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "learning_curve.png", dpi=120)
    plt.close(fig)


def fig_importance(ranked):
    labels = {
        "workload_value": "Tamaño de la carga",
        "svc_gpu_kind": "GPU requerida por el servicio",
        "svc_workload_kind": "Tipo de carga",
        "svc_resource_profile": "Perfil de recursos",
        "service_key": "Identidad del servicio",
        "service_size": "Tamaño del servicio",
        "svc_secondary_profile": "Perfil secundario",
        "calib_gflops_st": "Calibración FLOPS (1 hilo)",
        "svc_parallelism": "Paralelismo del servicio",
        "calib_gflops_mt": "Calibración FLOPS (multihilo)",
        "calib_parallel_speedup": "Aceleración paralela medida",
        "calib_pyloop_mops": "Calibración bucle Python",
        "calib_mem_gbps": "Calibración de memoria",
        "hw_cpu_max_ghz": "Frecuencia máxima de CPU",
        "hw_cpu_base_ghz": "Frecuencia base de CPU",
        "hw_cpu_cores_logical": "Núcleos lógicos",
        "hw_cpu_cores_physical": "Núcleos físicos",
        "hw_cpu_l1d_kb": "Caché L1 de datos",
        "hw_cpu_l2_kb": "Caché L2",
        "hw_cpu_l3_kb": "Caché L3",
        "hw_ram_gb": "Memoria RAM",
    }
    top = ranked[:10][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([labels.get(t[0], t[0]) for t in top], [t[1] for t in top])
    ax.set_xlabel("Importancia por permutación (caída de $R^2$)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=120)
    plt.close(fig)


def fig_scaling(df):
    # Selección representativa sin paneles vacíos: CPU, compilación, E/S y vídeo.
    svcs = ["sysbench", "ds-build", "ds-fio", "ffmpeg"]
    svcs = [s for s in svcs if s in df.service_key.unique()]
    host_labels = {
        "azure-vm-dataset-creator-vm": "Intel base",
        "azure-vm-dataset-creator-vm-2core": "2 núcleos",
        "azure-vm-dataset-creator-vm-amd": "AMD",
        "azure-vm-dataset-creator-vm-arm": "ARM",
        "azure-vm-dataset-creator-vm-mem": "Memoria",
        "desktop-i9-rtx4060ti": "Windows (i9)",
        "macbook-air-m4": "Mac M4",
        "windows-rtx5070ti": "Windows (i7)",
    }
    all_hosts = sorted(df.hw_hostname_label.unique())
    palette = plt.get_cmap("tab10").colors
    host_colors = {host: palette[i % len(palette)] for i, host in enumerate(all_hosts)}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, s in zip(axes.ravel(), svcs):
        g = df[df.service_key == s]
        for host, gh in g.groupby("hw_hostname_label"):
            gh = gh.groupby("workload_value")[TARGET].median().sort_index()
            ax.plot(
                gh.index,
                gh.values,
                marker="o",
                ms=4,
                color=host_colors[host],
                label=host_labels.get(host, host),
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        # Solo las potencias de 10: las etiquetas intermedias se solapan.
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(s)
        ax.set_xlabel("Magnitud de carga")
        ax.set_ylabel("Tiempo (s)")
        ax.grid(alpha=0.3)
    by_label = {}
    for ax in axes.ravel():
        handles, labels = ax.get_legend_handles_labels()
        by_label.update(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(OUT / "scaling_curves.png", dpi=120)
    plt.close(fig)


def main():
    df = clean(load())
    print(
        f"\nDataset modelado: {len(df)} filas | {df.hw_hostname_label.nunique()} hosts "
        f"| {df.service_key.nunique()} servicios"
    )
    ranked = feature_importance(df)
    per_host_loho(df)
    df_pred = per_service_interp(df)
    pts = learning_curve(df)
    collinearity(df)

    fig_pred_vs_actual(df_pred)
    fig_learning_curve(pts)
    fig_importance(ranked)
    fig_scaling(df)
    print(f"\nFiguras guardadas en {OUT}/")


if __name__ == "__main__":
    main()
