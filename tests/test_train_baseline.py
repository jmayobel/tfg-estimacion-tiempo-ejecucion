"""
Unit tests for the cleaning rules and the log-linear model of train_baseline.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# train_baseline imports scikit-learn, which the CI requirements do not install
pytest.importorskip("sklearn")

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from train_baseline import CALIB_COLUMNS, LogLinearServiceRegressor, clean  # noqa: E402


def _rows(specs):
    """Build a minimal dataset; each spec is (host, arch, service)."""
    return pd.DataFrame(
        [
            {
                "hw_hostname_label": host,
                "hw_arch": arch,
                "service_key": service,
                "service_name": service,
                "execution_time_s": 10.0,
                "num_samples": 5,
            }
            for host, arch, service in specs
        ]
    )


class TestCleanEmulatedServices:
    """x86-only images run under emulation on ARM hosts must be dropped."""

    def test_drops_x86_only_services_on_arm(self):
        df = _rows(
            [
                ("intel", "x86_64", "sysbench"),
                ("amd", "x86_64", "sysbench"),
                ("xeon", "x86_64", "sysbench"),
                ("mac", "arm64", "sysbench"),
                ("mac", "arm64", "busqueda-caminos"),
                ("intel", "x86_64", "busqueda-caminos"),
                ("amd", "x86_64", "busqueda-caminos"),
                ("xeon", "x86_64", "busqueda-caminos"),
            ]
        )
        out = clean(df)
        assert set(out["hw_hostname_label"]) == {"intel", "amd", "xeon"}
        assert len(out) == 6

    def test_keeps_multiarch_services_on_arm(self):
        df = _rows(
            [
                ("mac", "arm64", "ds-7zip"),
                ("arm-vm", "aarch64", "ds-7zip"),
                ("intel", "x86_64", "ds-7zip"),
            ]
        )
        assert len(clean(df)) == 3


class TestCleanFewHosts:
    """Holding out a host needs the service measured on at least three hosts."""

    def test_drops_service_on_two_hosts(self):
        df = _rows(
            [
                ("i7", "x86_64", "pytorch-gpu"),
                ("i9", "x86_64", "pytorch-gpu"),
                ("i7", "x86_64", "ds-7zip"),
                ("i9", "x86_64", "ds-7zip"),
                ("mac", "arm64", "ds-7zip"),
            ]
        )
        out = clean(df)
        assert set(out["service_key"]) == {"ds-7zip"}


def _toy(calibrated_hosts, uncalibrated_hosts, seed=0):
    """One service following a power law in cores and in its calibration."""
    rng = np.random.default_rng(seed)
    rows = []
    for host_i, host in enumerate(calibrated_hosts + uncalibrated_hosts):
        cores = 2 ** (host_i + 1)
        gflops = 20.0 * (host_i + 1)
        for w in (1, 2, 4, 8):
            t = 50.0 * w / cores**0.5 / (gflops / 20) ** 0.3 * rng.uniform(0.95, 1.05)
            row = {
                "service_key": "svc",
                "hw_hostname_label": host,
                "hw_cpu_cores_logical": cores,
                "hw_ram_gb": 16.0,
                "workload_value": w,
                "y": np.log1p(t),
            }
            for c in CALIB_COLUMNS:
                row[c] = gflops if host in calibrated_hosts else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


class TestLogLinearMissingCalibration:
    """Rows without calibration use the model without calibration, never a placeholder."""

    def test_uncalibrated_rows_use_model_without_calibration(self):
        df = _toy(["a", "b", "c"], ["d"])
        full = LogLinearServiceRegressor(use_calibration=True).fit(df, df["y"].to_numpy())
        phys = LogLinearServiceRegressor(use_calibration=False).fit(df, df["y"].to_numpy())
        sin = df["calib_gflops_st"].isna().to_numpy()
        np.testing.assert_allclose(full.predict(df)[sin], phys.predict(df)[sin])

    def test_calibrated_model_ignores_uncalibrated_rows(self):
        df = _toy(["a", "b", "c"], ["d"])
        cal = df[df["calib_gflops_st"].notna()]
        full = LogLinearServiceRegressor(use_calibration=True).fit(df, df["y"].to_numpy())
        only_cal = LogLinearServiceRegressor(use_calibration=True).fit(cal, cal["y"].to_numpy())
        np.testing.assert_allclose(full.predict(cal), only_cal.predict(cal))

    def test_predictions_are_finite_and_reasonable(self):
        df = _toy(["a", "b", "c"], ["d"])
        full = LogLinearServiceRegressor(use_calibration=True).fit(df, df["y"].to_numpy())
        pred, real = np.expm1(full.predict(df)), np.expm1(df["y"].to_numpy())
        assert np.all(np.isfinite(pred))
        assert np.all(np.abs(np.log(pred / real)) < np.log(10))  # never orders of magnitude off
