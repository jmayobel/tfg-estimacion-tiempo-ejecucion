#!/usr/bin/env python3
"""Train & evaluate execution-time regression on the combined multi-host dataset.

Consolidates every datasets/<host>/ml_dataset.csv, cleans, and benchmarks a few
regressors on log1p(execution_time_s) using HONEST grouped validation:

  * random K-fold        — optimistic upper bound (rows of the same run leak)
  * GroupKFold by host   — "can it predict an UNSEEN machine?" (the real test)
  * GroupKFold by service— "can it predict an UNSEEN workload?"

Only features knowable *before* running are used (hardware specs, service
profile, workload size). Per-run telemetry (cpu_usage, memory, ...) is excluded
because it is an outcome of the run, not an a-priori input — using it would be
leakage for the "estimate before executing" use case.
"""

from __future__ import annotations

import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent

CAT_FEATURES = [
    "service_key",
    "hw_arch",
    "hw_cpu_vendor",
    "hw_os_family",
    "hw_gpu_kind",
    "hw_gpu_vendor",
    "hw_has_gpu",
    "svc_resource_profile",
    "svc_secondary_profile",
    "svc_parallelism",
    "svc_workload_kind",
    "svc_gpu_kind",
]
# Measured machine-capability indices (scripts/calibrate.py). Every host carries them in
# all its rows: runs recorded before calibration existed take the value measured later on
# the same host (datasets/<host>/host_overrides.json, applied by build_ml_dataset.py).
CALIB_COLUMNS = [
    "calib_gflops_st",
    "calib_gflops_mt",
    "calib_mem_gbps",
    "calib_pyloop_mops",
    "calib_parallel_speedup",
]
NUM_FEATURES = [
    "hw_cpu_cores_logical",
    "hw_cpu_cores_physical",
    "hw_cpu_base_ghz",
    "hw_cpu_max_ghz",
    "hw_cpu_l1d_kb",
    "hw_cpu_l2_kb",
    "hw_cpu_l3_kb",
    "hw_ram_gb",
    "workload_value",
    "service_size",
    *CALIB_COLUMNS,
]
# Dropped on purpose: hw_gpu_memory_gb / hw_gpu_count / num_param are present for
# at most one host pattern (single distinct value globally) — no predictive
# variance with 5 hosts, and they make HistGBoost's per-fold binning crash.
# GPU presence is still captured by the categorical hw_gpu_kind / hw_has_gpu.
TARGET = "execution_time_s"
# Services whose time is set by external servers (ping/DNS to 8.8.8.8, HTTP to test.k6.io)
# and barely changes between hosts. iperf3 is NOT here: client and server are two
# containers on the same host, so its time depends on the machine like any local load.
NETWORK_BOUND = {"k6", "network-tools"}
# stress-ng: 5-19s runs dominated by container start/teardown overhead rather than
# the stress workload itself -> no usable hardware-scaling signal (yields nonsense
# Amdahl exponents). Excluded from modeling.
DEGENERATE_SERVICES = {"stress-ng"}
# Images published only for linux/amd64 (checked on Docker Hub). On an ARM host Docker
# runs them under emulation (Rosetta/QEMU), so the measured time is the emulator's, not
# the machine's. The Azure ARM VM skipped them via SKIP_SERVICES, but macbook-air-m4
# ran them emulated -> drop those rows instead of teaching the model emulator overhead.
X86_ONLY_SERVICES = {"busqueda-caminos", "sysbench"}
ARM_ARCHS = {"arm64", "aarch64"}
# A service must be measured on at least this many hosts. The central evaluation holds
# out one whole host, so a service seen on only two (pytorch-gpu: the two GPU hosts)
# leaves a single host to learn from and no hardware variation to extrapolate with.
MIN_HOSTS_PER_SERVICE = 3


def load() -> pd.DataFrame:
    frames = []
    for p in sorted(glob.glob(str(ROOT / "datasets/*/ml_dataset.csv"))):
        df = pd.read_csv(p)
        df["__source"] = Path(p).parent.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame, min_hosts: int = MIN_HOSTS_PER_SERVICE) -> pd.DataFrame:
    n0 = len(df)
    df = df[df[TARGET].notna()]
    df = df[~df["service_key"].isin(NETWORK_BOUND)]
    df = df[~df["service_key"].isin(DEGENERATE_SERVICES)]
    # macbook-air-m4 ran ollama natively (Metal GPU) instead of in-container; every
    # other host ran it containerized on CPU. Different measurement regime -> those
    # rows are not comparable under service_key='ollama' and would teach the model
    # a phantom hardware effect.
    n_native = int((df["service_name"] == "ollama-native").sum())
    df = df[df["service_name"] != "ollama-native"]
    emulated = df["service_key"].isin(X86_ONLY_SERVICES) & df["hw_arch"].isin(ARM_ARCHS)
    n_emulated = int(emulated.sum())
    df = df[~emulated]
    df = df[df["num_samples"].fillna(0) >= 3]
    # Drop services measured on too few hosts (see MIN_HOSTS_PER_SERVICE). Self-heals
    # once enough hosts run the service.
    hosts_per_svc = df.groupby("service_key")["hw_hostname_label"].nunique()
    few_hosts = sorted(hosts_per_svc[hosts_per_svc < min_hosts].index)
    n_few = int(df["service_key"].isin(few_hosts).sum())
    df = df[~df["service_key"].isin(few_hosts)]
    print(
        f"  cleaning: {n0} -> {len(df)} rows "
        f"(dropped network-bound, degenerate={sorted(DEGENERATE_SERVICES)}, "
        f"hosts<{min_hosts}={few_hosts} ({n_few} rows), ollama-native={n_native}, "
        f"x86-emulated-on-ARM={n_emulated}, NaN target, <3 samples)"
    )
    return df.reset_index(drop=True)


def prepare_xy(df: pd.DataFrame, num: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Feature matrix (categoricals as str) + log1p-transformed target."""
    X = df[CAT_FEATURES + list(num)].copy()
    for c in CAT_FEATURES:
        X[c] = X[c].astype(str)
    y_log = np.log1p(df[TARGET].to_numpy(dtype=float))
    return X, y_log


def usable_num_features(df: pd.DataFrame) -> list[str]:
    """Keep numeric features present with >=2 distinct non-null values.

    Drops columns that are globally constant or (almost) all-NaN — e.g. calib_*
    when only one host has been calibrated yet — which would otherwise break
    HistGBoost's per-fold binning and add no signal.
    """
    keep = []
    for c in NUM_FEATURES:
        if c in df.columns and df[c].nunique(dropna=True) >= 2:
            keep.append(c)
    return keep


class LogLinearServiceRegressor(BaseEstimator, RegressorMixin):
    """A log-linear Ridge regressor fit independently per service.

    Transforms hardware and workload features to log-space to model power-laws
    governing performance (e.g. empirical Amdahl's Law / Roofline limits):
        log(time) ~ log(workload) - beta2*log(cores) - beta3*log(RAM)

    Fitting per-service allows each workload to learn its own scaling exponent
    (e.g., parallelizable workloads will have beta2 closer to 1, sequential closer to 0).

    Missing calibration: with use_calibration=True, each service's calibrated model is
    fit only on rows whose calibration is complete, and a model without calibration is
    fit on all rows; rows lacking calibration are predicted with the latter. Missing
    values are never replaced by a placeholder: a placeholder such as 1.0 reads as a
    machine of 1 GFLOPS and the linear model extrapolates far outside the data.
    """

    def __init__(self, alpha=1.0, use_calibration=True):
        self.alpha = alpha
        self.use_calibration = use_calibration
        self.models: dict[str, dict] = {}
        self.fallback_models: dict[str, dict] = {}
        # Physical scaling axes we want to log-transform
        self.log_features = ["hw_cpu_cores_logical", "hw_ram_gb", "workload_value"]
        # Calibration features (also logged as scaling coefficients)
        self.calib_features = [
            "calib_gflops_st",
            "calib_gflops_mt",
            "calib_mem_gbps",
            "calib_pyloop_mops",
        ]

    def _transform(self, df: pd.DataFrame, with_calibration: bool) -> pd.DataFrame:
        df_trans = pd.DataFrame(index=df.index)
        cols = self.log_features + (self.calib_features if with_calibration else [])
        for col in cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                df_trans[f"log_{col}"] = np.log(np.maximum(vals, 1e-3))
        return df_trans

    def _has_calibration(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self.calib_features if c in df.columns]
        if not cols:
            return np.zeros(len(df), dtype=bool)
        return df[cols].apply(pd.to_numeric, errors="coerce").notna().all(axis=1).to_numpy()

    def _fit_one(self, X_s: pd.DataFrame, y_s: np.ndarray, with_calibration: bool) -> dict:
        X_trans = self._transform(X_s, with_calibration)
        # Fit and store pipeline components manually to keep scales and weights accessible
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        m = Ridge(alpha=self.alpha)
        m.fit(sc.fit_transform(imp.fit_transform(X_trans)), y_s)
        return {
            "imputer": imp,
            "scaler": sc,
            "model": m,
            "feature_names": list(X_trans.columns),
            "with_calibration": with_calibration,
        }

    @staticmethod
    def _predict_one(m_info: dict, X_trans: pd.DataFrame) -> np.ndarray:
        X_scaled = m_info["scaler"].transform(m_info["imputer"].transform(X_trans))
        return m_info["model"].predict(X_scaled)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> LogLinearServiceRegressor:
        # models: the one whose coefficients are reported (calibrated if possible);
        # fallback_models: same service without calibration, for rows that lack it.
        self.models = {}
        self.fallback_models = {}
        self.global_median_y = float(np.median(y))
        y = np.asarray(y)
        calibrated = self._has_calibration(X)

        for s in X["service_key"].unique():
            idx = (X["service_key"] == s).to_numpy()
            if idx.sum() < 2:
                continue
            phys = self._fit_one(X.loc[idx], y[idx], with_calibration=False)
            idx_cal = idx & calibrated
            if self.use_calibration and idx_cal.sum() >= 2:
                self.models[s] = self._fit_one(X.loc[idx_cal], y[idx_cal], with_calibration=True)
                self.fallback_models[s] = phys
            else:
                self.models[s] = phys
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.full(len(X), self.global_median_y)
        calibrated = self._has_calibration(X)
        for s in X["service_key"].unique():
            idx = (X["service_key"] == s).to_numpy()
            if s not in self.models:
                continue
            m_info = self.models[s]
            if m_info["with_calibration"]:
                use_cal = idx & calibrated
                use_phys = idx & ~calibrated
                if use_cal.any():
                    X_t = self._transform(X.loc[use_cal], with_calibration=True)
                    preds[use_cal] = self._predict_one(m_info, X_t)
                if use_phys.any():
                    X_t = self._transform(X.loc[use_phys], with_calibration=False)
                    preds[use_phys] = self._predict_one(self.fallback_models[s], X_t)
            else:
                X_t = self._transform(X.loc[idx], with_calibration=False)
                preds[idx] = self._predict_one(m_info, X_t)
        return preds

    def get_physical_coefficients(self) -> dict[str, dict[str, float]]:
        """De-standardize Ridge coefficients to physical exponents (beta = w_std / sigma)."""
        coefs = {}
        for s, m_info in self.models.items():
            model = m_info["model"]
            scaler = m_info["scaler"]
            feats = m_info["feature_names"]
            s_coefs = {}
            for i, feat in enumerate(feats):
                sigma = scaler.scale_[i]
                w_std = model.coef_[i]
                s_coefs[feat] = float(w_std / sigma) if sigma > 1e-6 else 0.0
            coefs[s] = s_coefs
        return coefs


def make_ordinal_encoder() -> OrdinalEncoder:
    """Ordinal encoder for tree models, robust to unseen/missing categories."""
    return OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2
    )


def make_models(num=None):
    cat = list(CAT_FEATURES)
    num = list(NUM_FEATURES) if num is None else list(num)

    # Tree models: ordinal-encode cats, pass numerics through (HGB handles NaN).
    ord_enc = ColumnTransformer(
        [
            ("cat", make_ordinal_encoder(), cat),
            ("num", "passthrough", num),
        ]
    )
    # Linear model: one-hot + scale + impute.
    lin_pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
                num,
            ),
        ]
    )
    rf_pre = ColumnTransformer(
        [
            ("cat", make_ordinal_encoder(), cat),
            ("num", SimpleImputer(strategy="median"), num),
        ]
    )

    return {
        "Dummy(median)": Pipeline([("pre", rf_pre), ("m", DummyRegressor(strategy="median"))]),
        "Ridge": Pipeline([("pre", lin_pre), ("m", Ridge(alpha=1.0))]),
        "LogLinear_Svc_Full": LogLinearServiceRegressor(alpha=1.0, use_calibration=True),
        "LogLinear_Svc_Phys": LogLinearServiceRegressor(alpha=1.0, use_calibration=False),
        "RandomForest": Pipeline(
            [
                ("pre", rf_pre),
                ("m", RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1)),
            ]
        ),
        "HistGBoost": Pipeline(
            [
                ("pre", ord_enc),
                (
                    "m",
                    HistGradientBoostingRegressor(random_state=0, max_iter=500, learning_rate=0.05),
                ),
            ]
        ),
    }


def evaluate(df: pd.DataFrame):
    num = usable_num_features(df)
    dropped = [c for c in NUM_FEATURES if c not in num]
    if dropped:
        print(f"  (features numéricas descartadas por constantes/NaN: {dropped})")
    X, y_log = prepare_xy(df, num)
    y_true = df[TARGET].to_numpy(dtype=float)

    run_group = (
        df["hw_hostname_label"].astype(str)
        + "|"
        + df["service_key"].astype(str)
        + "|"
        + df["workload_scale"].astype(str)
    )
    schemes = {
        "random 5-fold (optimista, fuga de réplicas)": (
            KFold(5, shuffle=True, random_state=0),
            None,
        ),
        "GroupKFold por (host,servicio,escala) — INTERPOLACIÓN honesta": (GroupKFold(5), run_group),
        "GroupKFold por HOST (HW no visto)": (
            GroupKFold(df["hw_hostname_label"].nunique()),
            df["hw_hostname_label"],
        ),
        "GroupKFold por SERVICIO (no visto)": (
            GroupKFold(df["service_key"].nunique()),
            df["service_key"],
        ),
    }
    models = make_models(num)

    for sname, (cv, groups) in schemes.items():
        print(f"\n=== {sname} ===")
        print(f"  {'modelo':16s} {'MAE_log':>8s} {'R2_log':>7s} {'MdAPE%':>7s}")
        for mname, model in models.items():
            try:
                pred_log = cross_val_predict(model, X, y_log, cv=cv, groups=groups, n_jobs=1)
            except Exception as exc:
                print(f"  {mname:16s} (falló: {type(exc).__name__})")
                continue
            pred = np.expm1(pred_log)
            mae_log = mean_absolute_error(y_log, pred_log)
            r2 = r2_score(y_log, pred_log)
            mdape = float(np.median(np.abs(pred - y_true) / np.maximum(y_true, 1e-9)) * 100)
            print(f"  {mname:16s} {mae_log:8.3f} {r2:7.3f} {mdape:7.1f}")


def main():
    print("Cargando datasets...")
    df = load()
    print(
        f"  {len(df)} filas, {df['__source'].nunique()} hosts: "
        f"{sorted(df['__source'].unique())}"
    )
    df = clean(df)
    print(
        f"  servicios: {df['service_key'].nunique()} | hosts: {df['hw_hostname_label'].nunique()}"
        f" | target {df[TARGET].min():.0f}-{df[TARGET].max():.0f}s"
    )
    evaluate(df)
    print(
        "\nLeyenda: MAE_log = error medio en log1p(s); R2_log = R² en log; "
        "MdAPE% = error porcentual ABSOLUTO mediano en segundos reales."
    )

    # Fit LogLinear_Svc_Phys on the whole dataset to print physical scaling
    # exponents (Amdahl's law).
    print("\n" + "=" * 80)
    print("EXPONENTES FÍSICOS ESTIMADOS POR SERVICIO (Amdahl's Law)")
    print("Modelo log(T) ~ C - bc*log(cores) - br*log(RAM) + bw*log(workload)")
    print("Nota: Un beta_cores cercano a 1.00 indica paralelización perfecta (T ~ 1/cores).")
    print("      Un beta_cores cercano a 0.00 indica rendimiento puramente secuencial.")
    print("=" * 80)

    X, y_log = prepare_xy(df, usable_num_features(df))

    m_phys = LogLinearServiceRegressor(alpha=1.0, use_calibration=False)
    m_phys.fit(X, y_log)
    coefs = m_phys.get_physical_coefficients()

    # Print as a nice table
    print(
        f"  {'servicio':24s} | {'bc (Speedup)':20s} | "
        f"{'br (RAM scale)':20s} | {'bw (workload)':13s}"
    )
    print("-" * 88)
    for s, c_dict in sorted(coefs.items()):
        bc = c_dict.get("log_hw_cpu_cores_logical", 0.0)
        br = c_dict.get("log_hw_ram_gb", 0.0)
        bw = c_dict.get("log_workload_value", 0.0)
        # Invert the sign of cores/ram (negative in the regression: more cores =>
        # less time) so they read as positive speedup exponents: T ~ W^bw / (C^bc * R^br)
        print(f"  {s:24s} | {-bc:20.3f} | {-br:20.3f} | {bw:13.3f}")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()
