#!/usr/bin/env python3
"""Assemble the ML-ready dataset from raw per-run output.

For every `output/runs/<run_id>/<service>.csv`, this script:
- aggregates the time-series rows into ONE row of summary statistics
  (mean, p50, p95, max, std) for the numeric telemetry columns;
- joins the hardware features captured in `hw_profile.json`;
- joins the workload-scale label and parametric workload values from
  `run_metadata.json`;
- joins the per-service semantic profile from `service_profiles.yaml`;
- applies the per-host corrections of `host_overrides.json` (next to the
  runs directory, if present): the host's calibration for runs recorded
  before calibration existed, and hardware values that must describe what
  the containers can use (e.g. the memory of Docker Desktop's VM on macOS);
- emits a single CSV at `output/ml_dataset.csv` ready to feed a
  regressor whose target is `execution_time_s`.

Critical: each row in the output corresponds to ONE
(hostname, workload_scale, service) execution. The per-second
Prometheus rows in the input are NOT independent samples — they are
the same workload measured many times — so feeding them to the model
unaggregated would inflate train/test scores via leakage.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("build_ml_dataset")


# Columns whose distribution across the run window we want as model features.
# Each gets aggregated to {col}_mean, {col}_p50, {col}_p95, {col}_max, {col}_std.
TIMESERIES_FEATURES = [
    "cpu_usage",
    "cpu_load_avg_1s",
    "memory_usage",
    "memory_avail",
    "memory_swap",
    "gpu_usage",
    "gpu_memory_usage",
    "gpu_power_usage",
    "gpu_clock_value",
    "cpu_clock_value",
    "cpu_temp",
    "gpu_temp",
    "npu_usage",
    "npu_memory_usage",
    "npu_temp",
    "disk_read",
    "disk_write",
    "network_receive",
    "network_transmit",
]

# Columns that default to 0.0 if not present or empty (rates/utilizations)
ZERO_DEFAULT_FEATURES = {
    "cpu_usage",
    "gpu_usage",
    "gpu_memory_usage",
    "gpu_power_usage",
    "gpu_clock_value",
    "npu_usage",
    "npu_memory_usage",
    "disk_read",
    "disk_write",
    "network_receive",
    "network_transmit",
}

# Per-run constants we want to keep verbatim (read from the first row).
PER_RUN_FIXED = [
    "service_name",
    "service_type",
    "architecture_type",
    "hostname",
    "cpu_name",
    "gpu_name",
    "npu_name",
    "cpu_cores",
    "gpu_memory_total",
    "service_size",
    "num_param",
    "workload_value",
    "workload_unit",
    "execution_time",
]


def aggregate_run_csv(csv_path: Path, service_key: str) -> dict[str, Any] | None:
    """Reduce one per-service CSV to a single row of features + target."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", csv_path, exc)
        return None

    if df.empty:
        logger.warning("Empty CSV skipped: %s", csv_path)
        return None

    row: dict[str, Any] = {
        "service_key": service_key,
        "num_samples": len(df),
    }

    # Carry per-run constants (use the first non-null value).
    for col in PER_RUN_FIXED:
        if col in df.columns:
            series = df[col].dropna()
            row[col] = series.iloc[0] if not series.empty else None
        else:
            row[col] = None

    # Aggregate time-series telemetry.
    for col in TIMESERIES_FEATURES:
        if col not in df.columns:
            default_val = 0.0 if col in ZERO_DEFAULT_FEATURES else None
            for suffix in ("mean", "p50", "p95", "max", "std"):
                row[f"{col}_{suffix}"] = default_val
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            default_val = 0.0 if col in ZERO_DEFAULT_FEATURES else None
            for suffix in ("mean", "p50", "p95", "max", "std"):
                row[f"{col}_{suffix}"] = default_val
            continue
        row[f"{col}_mean"] = float(series.mean())
        row[f"{col}_p50"] = float(series.median())
        row[f"{col}_p95"] = float(series.quantile(0.95))
        row[f"{col}_max"] = float(series.max())
        row[f"{col}_std"] = float(series.std(ddof=0))

    return row


def load_hw_profile(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "hw_profile.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse %s: %s", path, exc)
        return {}


def load_run_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse %s: %s", path, exc)
        return {}


def load_service_profiles(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        logger.warning("service_profiles.yaml not found at %s; profiles empty", path)
        return {}
    if yaml is None:
        logger.error("PyYAML not installed; install it to enable service profiles")
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("services", {}) or {}


# Columns from hw_profile.json we want to expose to the model.
HW_FEATURES_TO_KEEP = [
    "hostname_label",
    "arch",
    "os_family",
    "cpu_vendor",
    "cpu_cores_physical",
    "cpu_cores_logical",
    "cpu_base_ghz",
    "cpu_max_ghz",
    "cpu_l1d_kb",
    "cpu_l2_kb",
    "cpu_l3_kb",
    "ram_gb",
    "has_gpu",
    "gpu_kind",
    "gpu_vendor",
    "gpu_memory_gb",
    "gpu_compute_capability",
    "gpu_count",
]


# Machine-capability calibration fields (scripts/calibrate.py --merge-hw writes
# these into hw_profile.json). Carried verbatim (already named calib_*).
CALIB_FEATURES = [
    "calib_gflops_st",
    "calib_gflops_mt",
    "calib_mem_gbps",
    "calib_pyloop_mops",
    "calib_parallel_speedup",
]


HOST_OVERRIDES_FILENAME = "host_overrides.json"


def load_host_overrides(path: Path | None) -> dict[str, Any]:
    """Read a host's host_overrides.json; {} if absent or unreadable."""
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse %s: %s", path, exc)
        return {}


def apply_host_overrides(hw: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `hw` with the host's corrections applied.

    - "calibration": calibration is a property of the machine, so runs recorded
      before it existed take the value measured later on the same host. It only
      fills missing values; a calibration measured in the run itself is kept.
    - "hw": hardware values that replace the recorded ones, e.g. the memory of
      Docker Desktop's VM on macOS instead of the laptop's total RAM.

    Keys starting with "_" document the source and are ignored.
    """
    if not overrides:
        return hw
    out = dict(hw)
    for key, val in (overrides.get("calibration") or {}).items():
        if key in CALIB_FEATURES and out.get(key) is None:
            out[key] = val
    for key, val in (overrides.get("hw") or {}).items():
        if key in HW_FEATURES_TO_KEEP:
            out[key] = val
    return out


def enrich_with_hw(row: dict[str, Any], hw: dict[str, Any]) -> None:
    for key in HW_FEATURES_TO_KEEP:
        row[f"hw_{key}"] = hw.get(key)
    for key in CALIB_FEATURES:
        row[key] = hw.get(key)


# Columns from service_profiles.yaml we want as model features.
PROFILE_FEATURES_TO_KEEP = [
    "resource_profile",
    "secondary_profile",
    "parallelism",
    "workload_kind",
    "gpu_kind",
    "is_network_bound",
]


def enrich_with_profile(row: dict[str, Any], profile: dict[str, Any]) -> None:
    for key in PROFILE_FEATURES_TO_KEEP:
        row[f"svc_{key}"] = profile.get(key)


def collect_run(
    run_dir: Path,
    profiles: dict[str, dict[str, Any]],
    overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    hw = apply_host_overrides(load_hw_profile(run_dir), overrides or {})
    meta = load_run_metadata(run_dir)
    workload_scale = meta.get("workload_scale", "unknown")
    run_id = meta.get("run_id") or run_dir.name

    rows: list[dict[str, Any]] = []
    for csv_path in sorted(run_dir.glob("*.csv")):
        # Skip non-service CSVs (e.g. ollama per-request timing)
        stem = csv_path.stem
        if stem.endswith("_requests"):
            continue
        if stem in {"all_services"}:
            continue

        row = aggregate_run_csv(csv_path, service_key=stem)
        if row is None:
            continue

        row["run_id"] = run_id
        row["workload_scale"] = workload_scale
        enrich_with_hw(row, hw)
        enrich_with_profile(row, profiles.get(stem, {}))

        # Target — keep the name explicit so it's hard to misuse as a feature.
        if row.get("execution_time") is not None:
            try:
                row["execution_time_s"] = float(row["execution_time"])
            except (TypeError, ValueError):
                row["execution_time_s"] = None
        else:
            row["execution_time_s"] = None
        # Drop the raw column so consumers don't accidentally double-use it.
        row.pop("execution_time", None)

        rows.append(row)
    return rows


def write_dataset(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        logger.error("No rows assembled; nothing to write.")
        return

    # Union of keys; sort puts identifiers/target at the front for readability.
    leading = [
        "run_id",
        "workload_scale",
        "service_key",
        "execution_time_s",
        "num_samples",
    ]
    all_keys: set[str] = set()
    for r in rows:
        all_keys.update(r.keys())
    rest = sorted(k for k in all_keys if k not in leading)
    ordered = [k for k in leading if k in all_keys] + rest

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logger.info("Wrote %d rows to %s", len(rows), out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        default="output/runs",
        help="Directory containing per-run subfolders (default: output/runs).",
    )
    parser.add_argument(
        "--output",
        default="output/ml_dataset.csv",
        help="Path to the assembled ML dataset CSV (default: output/ml_dataset.csv).",
    )
    parser.add_argument(
        "--profiles",
        default="scripts/service_profiles.yaml",
        help="Path to service_profiles.yaml.",
    )
    parser.add_argument(
        "--host-overrides",
        default=None,
        help=(
            f"Path to the host's corrections (default: {HOST_OVERRIDES_FILENAME} next to "
            "--runs-dir, if it exists)."
        ),
    )
    parser.add_argument(
        "--exclude-network-bound",
        action="store_true",
        help=(
            "Drop services flagged is_network_bound=true (iperf3, network-tools, k6) "
            "from the output. Recommended when training a model that should reflect "
            "hardware, not network conditions."
        ),
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help=(
            "Discard rows where the underlying CSV had fewer than this many samples "
            "(default: 3). Filters out clearly broken / aborted runs."
        ),
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        logger.error("Runs directory not found: %s", runs_dir)
        return 1

    profiles = load_service_profiles(Path(args.profiles))
    overrides_path = (
        Path(args.host_overrides)
        if args.host_overrides
        else runs_dir.parent / HOST_OVERRIDES_FILENAME
    )
    overrides = load_host_overrides(overrides_path)
    if overrides:
        logger.info("Applying host corrections from %s", overrides_path)

    all_rows: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        run_rows = collect_run(run_dir, profiles, overrides)
        all_rows.extend(run_rows)

    # Optional filtering.
    before = len(all_rows)
    if args.exclude_network_bound:
        all_rows = [r for r in all_rows if not r.get("svc_is_network_bound")]
    all_rows = [r for r in all_rows if (r.get("num_samples") or 0) >= args.min_samples]
    dropped = before - len(all_rows)
    if dropped:
        logger.info("Filtered out %d rows (network-bound and/or below min-samples).", dropped)

    write_dataset(all_rows, Path(args.output))

    # Brief summary on stdout to make pipeline output more useful.
    if all_rows:
        df = pd.DataFrame(all_rows)
        logger.info(
            "Summary: %d rows | %d unique services | %d unique hosts | scales: %s",
            len(df),
            df["service_key"].nunique(),
            df["hw_hostname_label"].nunique(),
            sorted(df["workload_scale"].dropna().unique().tolist()),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
