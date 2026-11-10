"""Unit tests for scripts/build_ml_dataset.py."""

import csv
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

import build_ml_dataset  # noqa: E402


def _write_min_csv(path: Path, n_rows: int = 5) -> None:
    """Write a synthetic per-service CSV mimicking the orchestrator output."""
    rows = []
    for i in range(n_rows):
        rows.append(
            {
                "timestamp": f"2026-01-01 00:00:{i:02d}",
                "service_name": "test/image:latest",
                "service_type": "Test",
                "architecture_type": "CPU",
                "hostname": "test-host",
                "cpu_name": "Test CPU",
                "gpu_name": "Test GPU",
                "npu_name": "",
                "cpu_cores": 8,
                "gpu_memory_total": 8192,
                "service_size": 100.0,
                "num_param": "",
                "workload_value": 1000,
                "workload_unit": "ops",
                "execution_time": 42.0,
                "cpu_usage": 10.0 + i,
                "cpu_load_avg_1s": 1.5,
                "memory_usage": 500,
                "memory_avail": 7500,
                "memory_swap": 0,
                "gpu_usage": 0,
                "gpu_memory_usage": 100,
                "gpu_power_usage": 30,
                "gpu_clock_value": 1500,
                "cpu_clock_value": 3000,
                "cpu_temp": "",
                "gpu_temp": 40,
                "npu_usage": "",
                "npu_memory_usage": "",
                "npu_temp": "",
                "disk_read": 0,
                "disk_write": 0,
                "network_receive": 0,
                "network_transmit": 0,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


class TestAggregateRunCsv:
    """aggregate_run_csv reduces a per-service CSV to one summary row."""

    def test_basic_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "stress-ng.csv"
            _write_min_csv(csv_path, n_rows=10)
            row = build_ml_dataset.aggregate_run_csv(csv_path, "stress-ng")
        assert row is not None
        assert row["service_key"] == "stress-ng"
        assert row["num_samples"] == 10
        assert row["execution_time"] == 42.0
        assert row["workload_value"] == 1000
        # 10 to 19 -> mean 14.5, p95 around 18.55
        assert row["cpu_usage_mean"] == 14.5
        assert row["cpu_usage_max"] == 19.0
        assert row["cpu_usage_std"] > 0
        # cpu_temp was empty -> all aggregates None
        assert row["cpu_temp_mean"] is None

    def test_missing_file_returns_none(self):
        result = build_ml_dataset.aggregate_run_csv(Path("/nonexistent.csv"), "x")
        assert result is None

    def test_empty_csv_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "empty.csv"
            csv_path.write_text("timestamp,cpu_usage\n", encoding="utf-8")
            result = build_ml_dataset.aggregate_run_csv(csv_path, "x")
        assert result is None


class TestLoaders:
    """JSON loaders gracefully return {} when files are absent or invalid."""

    def test_load_hw_profile_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert build_ml_dataset.load_hw_profile(Path(tmp)) == {}

    def test_load_hw_profile_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = {"hostname_label": "abc", "cpu_cores_logical": 8}
            (Path(tmp) / "hw_profile.json").write_text(json.dumps(data), encoding="utf-8")
            assert build_ml_dataset.load_hw_profile(Path(tmp)) == data

    def test_load_hw_profile_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "hw_profile.json").write_text("{not valid", encoding="utf-8")
            assert build_ml_dataset.load_hw_profile(Path(tmp)) == {}

    def test_load_run_metadata_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert build_ml_dataset.load_run_metadata(Path(tmp)) == {}

    def test_load_run_metadata_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = {"run_id": "r1", "workload_scale": "small"}
            (Path(tmp) / "run_metadata.json").write_text(json.dumps(data), encoding="utf-8")
            assert build_ml_dataset.load_run_metadata(Path(tmp))["workload_scale"] == "small"

    def test_load_service_profiles_missing(self):
        assert build_ml_dataset.load_service_profiles(Path("/nonexistent.yaml")) == {}

    def test_load_service_profiles_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "p.yaml"
            yaml_path.write_text(
                "services:\n  stress-ng:\n    resource_profile: cpu\n", encoding="utf-8"
            )
            profiles = build_ml_dataset.load_service_profiles(yaml_path)
        assert "stress-ng" in profiles
        assert profiles["stress-ng"]["resource_profile"] == "cpu"


class TestEnrichers:
    """enrich_with_* mutate the row in place adding prefixed keys."""

    def test_enrich_with_hw(self):
        row = {}
        build_ml_dataset.enrich_with_hw(row, {"cpu_cores_logical": 8, "ram_gb": 16})
        assert row["hw_cpu_cores_logical"] == 8
        assert row["hw_ram_gb"] == 16
        # Missing keys become None
        assert row["hw_gpu_memory_gb"] is None

    def test_enrich_with_profile(self):
        row = {}
        build_ml_dataset.enrich_with_profile(
            row, {"resource_profile": "cpu", "is_network_bound": False}
        )
        assert row["svc_resource_profile"] == "cpu"
        assert row["svc_is_network_bound"] is False
        # Missing keys become None
        assert row["svc_parallelism"] is None


class TestCollectRun:
    """collect_run wires everything together for one run directory."""

    def test_collect_run_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "2026-01-01_00-00-00"
            run_dir.mkdir()
            _write_min_csv(run_dir / "stress-ng.csv", n_rows=5)
            # Skipped because of _requests suffix
            _write_min_csv(run_dir / "ollama_requests.csv", n_rows=3)
            (run_dir / "hw_profile.json").write_text(
                json.dumps({"hostname_label": "h", "cpu_cores_logical": 4}), encoding="utf-8"
            )
            (run_dir / "run_metadata.json").write_text(
                json.dumps({"run_id": "r1", "workload_scale": "small"}), encoding="utf-8"
            )

            rows = build_ml_dataset.collect_run(
                run_dir, profiles={"stress-ng": {"resource_profile": "cpu"}}
            )
        assert len(rows) == 1
        r = rows[0]
        assert r["service_key"] == "stress-ng"
        assert r["workload_scale"] == "small"
        assert r["execution_time_s"] == 42.0
        assert "execution_time" not in r  # dropped intentionally
        assert r["hw_hostname_label"] == "h"
        assert r["svc_resource_profile"] == "cpu"


class TestHostOverrides:
    """host_overrides.json keeps per-host corrections across dataset rebuilds."""

    OVERRIDES = {
        "_doc": "ignored",
        "calibration": {"_source": "ignored", "calib_gflops_st": 50.0, "calib_gflops_mt": 150.0},
        "hw": {"_source": "ignored", "ram_gb": 8.0},
    }

    def test_calibration_fills_only_missing(self):
        out = build_ml_dataset.apply_host_overrides({"ram_gb": 24.0}, self.OVERRIDES)
        assert out["calib_gflops_st"] == 50.0
        measured = build_ml_dataset.apply_host_overrides(
            {"calib_gflops_st": 53.1, "calib_gflops_mt": None}, self.OVERRIDES
        )
        assert measured["calib_gflops_st"] == 53.1  # a calibration from the run itself wins
        assert measured["calib_gflops_mt"] == 150.0

    def test_hw_values_are_replaced(self):
        out = build_ml_dataset.apply_host_overrides({"ram_gb": 24.0}, self.OVERRIDES)
        assert out["ram_gb"] == 8.0
        assert "_source" not in out and "_doc" not in out

    def test_no_overrides_is_identity(self):
        hw = {"ram_gb": 24.0}
        assert build_ml_dataset.apply_host_overrides(hw, {}) == hw

    def test_main_applies_overrides_next_to_runs_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            for run_id, hw in (
                ("r1", {"ram_gb": 24.0}),
                ("r2", {"ram_gb": 24.0, "calib_gflops_st": 53.1}),
            ):
                run_dir = runs_dir / run_id
                run_dir.mkdir(parents=True)
                _write_min_csv(run_dir / "sysbench.csv", n_rows=4)
                (run_dir / "hw_profile.json").write_text(json.dumps(hw), encoding="utf-8")
            (Path(tmp) / build_ml_dataset.HOST_OVERRIDES_FILENAME).write_text(
                json.dumps(self.OVERRIDES), encoding="utf-8"
            )
            out_csv = Path(tmp) / "ml.csv"
            argv = ["build_ml_dataset.py", "--runs-dir", str(runs_dir), "--output", str(out_csv)]
            with patch.object(sys, "argv", argv):
                assert build_ml_dataset.main() == 0
            df = pd.read_csv(out_csv).set_index("run_id")
        assert df.loc["r1", "calib_gflops_st"] == 50.0
        assert df.loc["r2", "calib_gflops_st"] == 53.1
        assert (df["hw_ram_gb"] == 8.0).all()


DATASETS = Path(__file__).parent.parent / "datasets"


@pytest.mark.parametrize(
    "csv_path", sorted(DATASETS.glob("*/ml_dataset.csv")), ids=lambda p: p.parent.name
)
def test_datasets_keep_calibration(csv_path):
    """Every host dataset carries its calibration in all rows (see host_overrides.json)."""
    df = pd.read_csv(csv_path, usecols=build_ml_dataset.CALIB_FEATURES)
    assert df.notna().all().all(), f"{csv_path.parent.name} has rows without calibration"


class TestWriteAndMain:
    """write_dataset writes a header-ordered CSV; main runs end-to-end."""

    def test_write_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.csv"
            build_ml_dataset.write_dataset(
                [{"run_id": "r1", "service_key": "s", "execution_time_s": 10.0, "x": 1}], out
            )
            assert out.exists()
            with out.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert rows[0]["run_id"] == "r1"
            assert rows[0]["execution_time_s"] == "10.0"

    def test_write_dataset_empty_does_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.csv"
            build_ml_dataset.write_dataset([], out)
            assert not out.exists()

    def test_main_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            run_dir = runs_dir / "r1"
            run_dir.mkdir(parents=True)
            _write_min_csv(run_dir / "sysbench.csv", n_rows=4)

            out_csv = Path(tmp) / "ml.csv"
            profiles_path = Path(tmp) / "profiles.yaml"
            profiles_path.write_text(
                "services:\n  sysbench:\n    resource_profile: cpu\n", encoding="utf-8"
            )

            argv = [
                "build_ml_dataset.py",
                "--runs-dir",
                str(runs_dir),
                "--output",
                str(out_csv),
                "--profiles",
                str(profiles_path),
            ]
            with patch.object(sys, "argv", argv):
                rc = build_ml_dataset.main()
            assert rc == 0
            assert out_csv.exists()
            df = pd.read_csv(out_csv)
            assert len(df) == 1
            assert df.iloc[0]["service_key"] == "sysbench"

    def test_main_missing_runs_dir(self):
        argv = ["build_ml_dataset.py", "--runs-dir", "/no/such/dir"]
        with patch.object(sys, "argv", argv):
            rc = build_ml_dataset.main()
        assert rc == 1
